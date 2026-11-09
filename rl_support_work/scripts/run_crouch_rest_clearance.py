"""Reference-only clearance/load-transfer probes; separate from preserved probes."""
import argparse,sys,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

p=argparse.ArgumentParser(add_help=False)
p.add_argument('--clearance-lift',type=float,default=.015)
p.add_argument('--end-forward',type=float,default=.025)
p.add_argument('--end-gap',type=float,default=.003)
clear_args,rest=p.parse_known_args();sys.argv=[sys.argv[0]]+rest
import run_crouch_rest_balance as balanced
from crouch_rest_motion import continuous_plan,smooth
from build_near_seat_candidates import solve
from humanoid_sitting.shared_frame import level_rotation

probe=balanced.probe
def plan(model,q,gravity,measurement,reference_origin,command,calibration):
    c,qs,a=continuous_plan(model,q,gravity,measurement,reference_origin,command,calibration)
    R=level_rotation(gravity);fk=model.fk(q);origin=np.array(a['origin'])
    feet={s:(R@fk[s+'_ankle_roll_link'][:3,3]+origin,R@fk[s+'_ankle_roll_link'][:3,:3]) for s in ['left','right']}
    n=np.array(a['normal']);d=a['offset'];pivot=np.array(a['pivot']);pitch=a['pitch']
    ids=[model.names.index(s+'_'+j+'_joint') for s in ['left','right'] for j in ['hip_pitch','knee','ankle_pitch']]
    errors=[];roots=[]
    for k,u in enumerate(np.linspace(0,1,25)):
        alpha=smooth(u);angle=pitch*(1-alpha)
        x=origin[0]+((R-Rotation.from_euler('y',angle).as_matrix())@pivot)[0]+clear_args.end_forward*alpha
        gap=a['gap_initial']+(clear_args.end_gap-a['gap_initial'])*alpha+clear_args.clearance_lift*np.sin(np.pi*u)**2
        qi,root,error=solve(model,q,feet,n,d,x,angle,gap=gap,y=origin[1])
        assert error<.003,(float(u),error)
        c[k,ids]=qi[ids]+(1-alpha)*np.array(a['bias']);qs[k]=qi;errors.append(error);roots.append(root.tolist())
    c[0]=command
    a.update(clearance_config=vars(clear_args),rear_shift=clear_args.end_forward,target_root=roots[-1],root_path=roots,ik_error=max(errors),gap_final=clear_args.end_gap)
    return c,qs,a

class ClearanceEnv(balanced.BalancedEnv):
    def close(self):
        super().close();(self.out/'clearance_config.json').write_text(json.dumps(vars(clear_args),indent=2));(self.out/'source'/'run_crouch_rest_clearance.py').write_text(Path(__file__).read_text())

probe.progressive.rolling_plan=plan
probe.progressive.ProgressiveEnv=ClearanceEnv
if __name__=='__main__':probe.progressive.main()
