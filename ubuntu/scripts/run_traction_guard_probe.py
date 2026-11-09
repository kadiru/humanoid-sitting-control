"""Model/encoder/IMU/depth housing-clearance feedback, physical probe only."""
import argparse,sys,json
from pathlib import Path
import numpy as np
import torch
from humanoid_sitting.shared_frame import level_rotation,apply
from plan_seat_load_targets import clipped_vertices
p=argparse.ArgumentParser(add_help=False);p.add_argument('--guard-gain',type=float,default=2.);p.add_argument('--guard-clearance',type=float,default=.006);p.add_argument('--contact-gap',type=float,default=-.001)
guard,rest=p.parse_known_args();sys.argv=[sys.argv[0]]+rest
import run_traction_transfer_probe as shear
shear.bridge.static_args.contact_gap=guard.contact_gap
class GuardEnv(shear.TractionEnv):
    def __init__(self,*args,**kwargs):
        self.guard_ready=False;self.guard_tick=-1;self.guard_records=[]
        super().__init__(*args,**kwargs)
        self.lift=np.zeros(self.num_envs);self.guard_correction=torch.zeros_like(self.previous);self.guard_ready=True
    def reference(self):
        ref=super().reference()
        if not self.guard_ready:return ref
        if self.t%5==0 and self.guard_tick!=self.t:
            self.guard_tick=self.t;q=self.robot.data.joint_pos.torch.cpu().numpy();torque=self.robot.data.applied_torque.torch.cpu().numpy();gravity=self.robot.data.projected_gravity_b.torch.cpu().numpy();correction=np.zeros_like(q);clear=np.full(self.num_envs,np.nan)
            latest={a['case']:a for a in self.plan_audit}
            for i,a in latest.items():
                if self.near_start[i]<0:continue
                R=level_rotation(gravity[i]);fk=self.model.fk(q[i]);soles=np.concatenate([apply(fk[s+'_ankle_roll_link'],self.model.vertices[s+'_ankle_roll_link']) for s in ['left','right']])@R.T
                origin=np.array([float(self.estimated_root_x[i]),a['origin'][1],-soles[:,2].min()]);n=np.array(a['normal']);d=a['offset'];values=[]
                for side in ['left','right']:
                    for joint in ['hip_pitch','hip_roll']:
                        b=side+'_'+joint+'_link';v=clipped_vertices(apply(fk[b],self.model.vertices[b])@R.T+origin)
                        if len(v):values.append(float((v@n+d).min()))
                clear[i]=min(values) if values else 1.
                desired=np.clip(guard.guard_gain*(guard.guard_clearance-clear[i]),0,.02)
                self.lift[i]=.7*self.lift[i]+.3*desired
                w,condition,J=self.wrench_observer.estimate(q[i],torque[i],gravity[i],fk=fk,return_jacobian=True)
                for side in range(2):
                    if condition[side]<200:correction[i,self.sagittal_ids[side]]=np.clip(np.linalg.solve(J[side].T,[0,-self.lift[i],0]),-.12,.12)
            self.guard_correction=torch.tensor(correction,device=self.device,dtype=torch.float32)
            self.guard_records.append(dict(t=self.t*.02,clearance=clear.copy(),lift=self.lift.copy(),correction=correction.copy()))
        return ref+self.guard_correction
    def close(self):
        super().close();np.savez_compressed(self.out/'clearance_guard.npz',**{k:np.array([x[k] for x in self.guard_records]) for k in self.guard_records[0]});(self.out/'guard_config.json').write_text(json.dumps(vars(guard),indent=2));(self.out/'source'/'run_traction_guard_probe.py').write_text(Path(__file__).read_text())
shear.progressive.ProgressiveEnv=GuardEnv
if __name__=='__main__':shear.progressive.main()
