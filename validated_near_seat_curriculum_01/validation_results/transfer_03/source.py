"""Sensor-planned near-seat transfer validation; never feeds labels to controls."""
import argparse,json,hashlib,shutil
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
import run_resting_posture as base
import run_supported_postures as support
from humanoid_sitting.shared_frame import level_rotation
from humanoid_sitting.supported_surface import observed_support
from humanoid_sitting.contact_aware_reference import measured_level_plane
from build_near_seat_candidates import solve

DT=.02

def main():
    p=argparse.ArgumentParser();p.add_argument('--bank',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--duration',type=float,default=6.);p.add_argument('--seed',type=int,default=57101);p.add_argument('--fixed',action='store_true');p.add_argument('--plane-calibration',default='/data/humanoid-sitting-perception/contact_effort_repair_01/plane_calibration.json');base.add_launcher_args(p);a=p.parse_args();a.mode='fixed';a.depth_check=True;a.enable_cameras=True
    out=a.output;out.mkdir(exist_ok=False);geometry=json.loads((Path(a.bank)/'geometry.json').read_text());bank=dict(np.load(Path(a.bank)/'candidates.npz'));assert bank['valid'].all()
    a.num_envs=len(geometry);a.seconds=20.;cfg=support.make_cfg(a,geometry);cfg.scene.rest_camera=None
    base.write_json(out/'geometry.json',geometry);base.write_json(out/'manifest.json',dict(duration=a.duration,fixed=a.fixed,bank=a.bank,seed=a.seed,scope='Measured depth plane, encoders, IMU, fixed horizontal reference. Ground truth only reset and labels.'))
    shutil.copy2(__file__,out/'source.py')
    with base.launch_simulation(cfg,a):
        raw=base.gym.make('Humanoid-Sit-Learned-G1',cfg=cfg).unwrapped;raw.reset();support.restore(raw,bank);robot=raw.scene['robot'];device=raw.device;model=base.BodyModel(base.MODEL)
        command=torch.tensor(bank['command'],device=device,dtype=torch.float32);start=command.clone();paths=[];audit=[]
        # Refresh simulated encoders/IMU after indexed state writes; log all settling ticks.
        settling=[]
        for tick in range(25):
            raw.step(command);sm,sd,sf=support.metrics(raw)
            settling.append(dict(tick=tick+1,failures={k:v.cpu().tolist() for k,v in sf.items()},slip_ok=sd['no_slip'].cpu().tolist()))
            assert not sm['failure'].any(), (tick,settling[-1])
        base.write_json(out/'initial_settling.json',settling)
        # Camera acquisition itself must not advance physics.
        raw.sim.render_context.update_scene_state(raw.sim.get_physics_step_count());cam=raw.scene['rear_camera'];base.acquire_current_camera(cam)
        depth=cam.data.output['distance_to_image_plane'].torch.cpu().numpy();K=cam.data.intrinsic_matrices.torch.cpu().numpy()
        q=robot.data.joint_pos.torch.cpu().numpy();gravity=robot.data.projected_gravity_b.torch.cpu().numpy()
        for i in range(a.num_envs):
            measurement,diagnostic=observed_support(model,K[i],support.REAR_CAMERA,depth[i],q[i],gravity[i]);assert measurement is not None,(i,diagnostic)
            n,d,origin=measured_level_plane(model,q[i],gravity[i],measurement,bank['root_pose'][i]);raw_n=n.copy();raw_d=d
            calibration=json.loads(Path(a.plane_calibration).read_text());rn=np.array(calibration['reference_normal']);rd=calibration['reference_offset']
            height=.4-(-.4*n[0]+d)/n[2]+(-.4*rn[0]+rd)/rn[2];slope=np.arctan2(n[0],n[2])-np.arctan2(rn[0],rn[2]);n=Rotation.from_euler('y',slope).as_matrix()[:,2];d=-n@np.array([-.4,0.,height])
            R=level_rotation(gravity[i]);fk=model.fk(q[i]);feet={s:(R@fk[s+'_ankle_roll_link'][:3,3]+origin,R@fk[s+'_ankle_roll_link'][:3,:3]) for s in ['left','right']}
            pitch=Rotation.from_matrix(R).as_euler('xyz')[1];commands=[bank['command'][i]];leg=[j for j,name in enumerate(model.names) if any(s in name for s in ['hip_','knee','ankle'])]
            for u in np.linspace(0,1,25)[1:]:
                alpha=u*u*(3-2*u);qi,root,error=solve(model,q[i],feet,n,d,origin[0]*(1-alpha)-.18*alpha,pitch*(1-alpha),y=origin[1]);assert error<.003,(i,float(u),error,origin.tolist(),n.tolist(),d,float(pitch))
                c=bank['command'][i].copy();c[leg]=qi[leg];commands.append(c)
            paths.append(commands);audit.append(dict(case=i,measured_normal=n.tolist(),measured_offset=d,raw_normal=raw_n.tolist(),raw_offset=raw_d,origin=origin.tolist(),initial_pitch=float(pitch),scope='Horizontal x is explicit offline reference assistance; plane and feet from sensors.'))
        paths=torch.tensor(np.array(paths),device=device,dtype=torch.float32);np.savez_compressed(out/'planned_commands.npz',command=paths.cpu().numpy());base.write_json(out/'plan_audit.json',audit)
        np.savez_compressed(out/'startup.npz',**bank)
        flags_all={};good_ticks=torch.zeros(a.num_envs,device=device,dtype=torch.long);held=good_ticks.clone();slip_ticks=good_ticks.clone();traction_ticks=good_ticks.clone();illegal_ticks=good_ticks.clone();max_slip=torch.zeros(a.num_envs,device=device);max_util=max_slip.clone();traces=[];first=[];seen=set()
        with torch.inference_mode():
            for step in range(1000):
                ph=min(24.,(step+1)*DT/a.duration*24);lo=min(23,int(ph));f=ph-lo
                desired=(1-f)*paths[:,lo]+f*paths[:,lo+1]
                if a.fixed:desired=start
                command+=(.1*(desired-command)).clamp(-.003,.003)
                raw.step(command);assert not raw.reset_buf.any();m,d,flags=support.metrics(raw)
                good_ticks+=d['good'];held=torch.where(d['good'],held+1,torch.zeros_like(held));slip_ticks+=~d['no_slip'];traction_ticks+=~d['traction_ok'];illegal_ticks+=~d['contacts_legal']
                loaded=m['force_w'].norm(dim=-1)>.01*m['bodyweight_N'][:,None];slip=torch.where(loaded,m['slip_m_s'],0).max(1).values;max_slip=torch.maximum(max_slip,slip);max_util=torch.maximum(max_util,d['maximum_friction_utilization'])
                for k,v in flags.items():
                    if k not in flags_all:flags_all[k]=torch.zeros_like(v)
                    flags_all[k]|=v
                for i in m['failure'].nonzero().flatten().tolist():
                    if i not in seen:first.append(dict(case=i,t=(step+1)*DT,flags={k:bool(v[i]) for k,v in flags.items()},wrong=float(d['wrong_force_N'][i])));seen.add(i)
                if step%5==4:traces.append(dict(t=(step+1)*DT,body=robot.data.body_pose_w.torch.cpu().numpy().copy(),q=robot.data.joint_pos.torch.cpu().numpy().copy(),command=command.cpu().numpy().copy(),torque=robot.data.applied_torque.torch.cpu().numpy().copy(),seat=d['rest_vertical_load_bw'].cpu().numpy(),foot=d['foot_load_bw'].cpu().numpy(),good=d['good'].cpu().numpy(),traction=d['traction_ok'].cpu().numpy(),slip=slip.cpu().numpy()))
                if step%250==249:print('SECONDS',(step+1)*DT,'good',int(d['good'].sum()),'physical',len(seen),flush=True)
            bad=torch.stack(list(flags_all.values())).any(0);success=~bad&(held>=250);strict=success&(slip_ticks==0)&(traction_ticks==0)&(illegal_ticks==0)
            rows=[dict(case=i,geometry=geometry[i],success=bool(success[i]),strict_transfer=bool(strict[i]),good_ticks=int(good_ticks[i]),final_hold_s=int(held[i])*DT,slip_ticks=int(slip_ticks[i]),traction_bad_ticks=int(traction_ticks[i]),illegal_ticks=int(illegal_ticks[i]),max_slip_m_s=float(max_slip[i]),max_friction_utilization=float(max_util[i]),failures={k:bool(v[i]) for k,v in flags_all.items()}) for i in range(a.num_envs)]
            base.write_json(out/'results.json',dict(rows=rows,successes=int(success.sum()),strict_successes=int(strict.sum()),count=a.num_envs,body_names=robot.body_names,joint_names=robot.joint_names,origins=raw.scene.env_origins.cpu().tolist()));base.write_json(out/'first_failures.json',first)
            np.savez_compressed(out/'trace.npz',**{k:np.stack([x[k] for x in traces]) for k in traces[0]});print('RESULT',int(success.sum()),'strict',int(strict.sum()),a.num_envs,flush=True)
        raw.close()
if __name__=='__main__':main()
