"""Run a measured single-step alignment stage ahead of the original evaluation."""
import json
from pathlib import Path
import sys

import numpy as np


def install_step(output,config):
    import torch
    from scipy.spatial.transform import Rotation
    from environment import FullBodyEnv
    from policy import command
    import run_supported_postures as support
    from standing_rim import TILT_DEG
    from rearstep_kin import MassModel
    from seat_alignment import Alignment, UNITS, correction, points, register
    from controlled_seat_step import BASE_END, GATE, STAGE_TICKS, ease, floor_witness

    metrics=support.metrics
    original_observe=FullBodyEnv._observe

    def run(env):
        d=env.robot.data
        initial=env.previous.clone()
        origins=env.raw.scene.env_origins.cpu().numpy().copy()
        names=env.robot.body_names
        ids=[names.index(side+'_ankle_roll_link') for side in ('left','right')]
        soles=[env.model.vertices[names[i]] for i in ids]
        def local_body():
            v=d.body_pose_w.torch.cpu().numpy().copy()
            v[...,:3]-=origins[:,None,:]
            return v
        def clearance(b):
            w=Rotation.from_quat(b[:,ids,3:].reshape(-1,4)).as_matrix().reshape(len(b),2,3,3)
            return np.column_stack([b[:,i,2]+(w[:,j,2] @ soles[j].T).min(axis=1) for j,i in enumerate(ids)])
        def scan():
            camera=env.raw.scene['rear_scan']
            clock=env.raw.sim.get_physics_step_count()
            env.raw.sim.render_context.update_scene_state(clock)
            support.base.acquire_current_camera(camera)
            assert clock==env.raw.sim.get_physics_step_count()
            return (camera.data.output['distance_to_image_plane'].torch.cpu().numpy()[...,0].copy(),
                    camera.data.intrinsic_matrices.torch.cpu().numpy().copy())
        count=len(initial)
        with np.load(config['steps']) as z:
            paths=z['reference']
            path_data={name:z[name] for name in z.files}
        opts=np.zeros(count,dtype=int)
        pred=np.zeros((count,3))
        valid=np.ones(count,bool)
        if config['mode']=='pilot':
            opts=np.array(config['choices'])
        else:
            mount=support.REAR_CAMERA.copy()
            mount[:3,:3]=(Rotation.from_matrix(mount[:3,:3])*Rotation.from_euler('x',-TILT_DEG,degrees=True)).as_matrix()
            depth,k=scan()
            q=d.joint_pos.torch.cpu().numpy()
            gravity=d.projected_gravity_b.torch.cpu().numpy()
            with np.load(config['templates']) as refs:
                measurements=[register(refs[key],points(depth[i],k[i],mount,q[i],gravity[i],env.model))
                              for i,key in enumerate(config['templates_by_row'])]
            features,valid=map(np.array,zip(*measurements))
            actor=Alignment().eval().to(env.device)
            actor.load_state_dict(torch.load(config['weights'],map_location=env.device,weights_only=True))
            with torch.inference_mode():
                pred=actor(torch.as_tensor(features,device=env.device,dtype=torch.float32)).cpu().numpy()*UNITS
            requested=np.array([correction(row) for row in pred])
            if config['mode']=='aligned':
                # Compare the frozen camera request with physically verified
                # net body displacements. Standing is always an available action.
                options=np.array([0]+config['validated_choices'])
                achieved=np.array([[0.,0.,0.]]+config['validated_displacements'])
                errors=np.square((requested[:,None,:]-achieved[None,:,:])/UNITS).sum(2)
                errors[:,1:]+=.09
                if config.get('feedback'):
                    errors[:,1:]=np.where(abs(requested[:,0:1])>=.012,errors[:,1:],np.inf)
                choice=options[errors.argmin(1)]
                opts=np.where(valid,choice,0)
        cuts=np.array(config.get('cutoffs',[config.get('cutoff',BASE_END-1)]*count))
        staged=torch.as_tensor(paths,device=env.device,dtype=env.previous.dtype)
        q_start=local_body()
        z_start=clearance(q_start)
        phase=np.zeros(count,dtype=int)
        unload_count=np.zeros(count,dtype=int)
        bad=np.zeros(count,bool)
        hip_names=('left_hip_pitch_link','right_hip_pitch_link','left_hip_roll_link','right_hip_roll_link')
        hips=[env.raw.scene['seat_probe_'+s] for s in hip_names]
        trace={name:[] for name in ('body','command','floor_load','clearance','floor_speed','verified_floor',
                                    'slip','flags','hip_N','phase','prestage_power_W','flat_speed')}
        feedback=config.get('feedback',0.)
        if feedback:
            mass=MassModel()
            last_error=np.zeros((count,3))
            speed_filter=np.zeros((count,3))
            trim=np.zeros((count,4))
        for tick in range(STAGE_TICKS):
            target=initial[:,:23].clone()
            enabled=(opts>0)&~bad
            advancing=enabled & ((phase!=GATE)|(unload_count>=3))
            for i in np.flatnonzero(enabled):
                p=phase[i]
                target[i]=staged[opts[i]-1,p]
                if not feedback or p<185:
                    continue
                q=d.joint_pos.torch[i].cpu().numpy()
                up=-d.projected_gravity_b.torch[i].cpu().numpy()
                gyro=d.root_ang_vel_b.torch[i].cpu().numpy()
                f=np.array([1.,0.,0.])-up[0]*up
                f/=np.linalg.norm(f)
                com,feet=mass.estimate(q,np.stack([f,np.cross(up,f),up]))
                frac=np.maximum(path_data['weight'][p],0.)
                frac=frac/max(frac.sum(),.3)
                ref=path_data['com'][p]-frac@path_data['feet'][p]
                error=(com-frac@feet)-ref
                speed_filter[i]=.78*speed_filter[i]+.22*(error-last_error[i])*50 if p>185 else 0.
                last_error[i]=error
                ext=error+.18*speed_filter[i]
                ref_up=path_data['up'][p]
                pitch=np.arcsin(np.clip(ref_up[0],-1,1))-np.arcsin(np.clip(up[0],-1,1))+.09*gyro[1]
                roll=np.arctan2(up[1],up[2])-np.arctan2(ref_up[1],ref_up[2])+.09*gyro[0]
                update=np.array([2.0*ext[0]+.30*pitch,-2.0*ext[1]+.25*roll,
                                 1.10*ext[0]+.75*pitch,-.95*ext[1]+.72*roll])
                w=min(1.,(p-185)/40.)
                if config.get('settle_taper'):
                    w*=1-ease(p,391,427)
                trim[i]+=np.clip(np.clip(feedback*w*update,-.085,.085)-trim[i],-.002,.002)
                for side in range(2):
                    gain=.20+1.3*frac[side]
                    target[i,[15+side,19+side,side,3+side]]+=target.new_tensor(gain*trim[i])
            env.previous[:,:23]=command(env.previous[:,:23],target,env.limit_lower,env.limit_upper)
            env.raw.step(env.previous)
            env.simulator_control_steps+=1
            assert not env.raw.reset_buf.any()
            m,detail,flags=metrics(env.raw)
            loads,speed,verified,active=floor_witness(env,m)
            slip=(speed>.02)&active
            hard=torch.stack(list(flags.values()),dim=1).cpu().numpy()
            bad|=hard.any(1)|slip.any(1)|(~verified).any(1)
            unloaded=(loads[:,0]<.01) & (loads[:,1]>.75)
            unload_count=np.where(unloaded,unload_count+1,0)
            pose=local_body()
            values=dict(body=pose,command=env.previous.cpu().numpy().copy(),floor_load=loads,
                        clearance=clearance(pose),floor_speed=speed,verified_floor=verified,slip=slip,flags=hard,
                        hip_N=torch.stack([s.data.normal_force_matrix_w_history.torch.sum((2,3)).norm(dim=-1).amax(1)
                                           for s in hips],1).cpu().numpy(),phase=phase.copy(),
                        prestage_power_W=(m['torque']*m['joint_velocity']).abs().sum(1).cpu().numpy().copy(),
                        flat_speed=d.root_lin_vel_w.torch.norm(dim=-1).cpu().numpy().copy())
            for name,value in values.items():
                trace[name].append(value)
            phase=np.minimum(phase+advancing,cuts)
        assert np.abs(np.diff(np.concatenate((initial.cpu().numpy()[None],trace['command'])),axis=0)).max()<.0030003
        np.savez_compressed(output/'single_step.npz', **trace,initial_command=initial.cpu().numpy(),
                            names=names, initial_body=q_start, initial_clearance=z_start,options=opts,
                            pred=pred, valid=valid, cutoffs=cuts)

    def observe(env,force=False):
        if force and env.hist is None and not hasattr(env,'single_step_completed'):
            env.single_step_completed=True
            run(env)
        return original_observe(env,force)

    FullBodyEnv._observe=observe


if __name__=='__main__':
    original,flag,profile,launcher=sys.argv[1:5]
    assert flag=='--arm-worker'
    sys.path[:0]=[str(Path(original).parent),str(Path(launcher).parent),str(Path(launcher).parent/'trainer')]
    output=Path(sys.argv[sys.argv.index('--output')+1])
    config=json.loads((output.parent/'step_config.json').read_text())
    install_step(output,config)
    from pelvis_marker_worker import install
    install(Path(original),Path(profile),Path(launcher),sys.argv[5:])
