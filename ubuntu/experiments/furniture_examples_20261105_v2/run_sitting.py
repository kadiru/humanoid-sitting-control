"""Run the preserved best09 teacher on a new mesh, recording every control."""
import os,sys,json,runpy,argparse,importlib.util
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent.parent
read=lambda p:json.loads(Path(p).read_text())
def write(p,d):Path(p).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--variant',required=True);a=parser.parse_args()
    variant=next(v for v in read(HERE/'protocol.json')['variants'] if v['id']==a.variant);assert variant['task']=='sitting'
    case=next(c for c in read(HERE/'cases.json') if c['id']==variant['case']);job=read(HERE/'jobs'/(a.variant+'.json'))
    out=Path(job['arguments'][job['arguments'].index('--output')+1])
    assert os.environ['CUDA_VISIBLE_DEVICES']=='GPU-ab1950b0-01dd-0fab-407f-2d17936b27f4'
    assert torch.cuda.device_count()==1 and '4090' in torch.cuda.get_device_name(0)
    sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'experiments/furniture_perception_repair_20261029_v1'),str(ROOT/'experiments/furniture_validation_20261028_v2'),str(ROOT/'experiments/furniture_test_20261028_v1')]
    capture=load('furniture_aligned_capture',ROOT/'experiments/furniture_perception_repair_20261029_v1/capture.py')
    observer=capture.Observer(variant['intent'])
    import humanoid_sitting.supported_surface as geometry
    geometry.observed_support=observer
    import run_supported_transition as transition
    import run_supported_postures as support
    import run_resting_posture as base
    import surface_map
    from contact_adapter import MeshMetrics
    from scene_adapter import reset_mesh
    surface_map.ASSETS=HERE
    transition.observed_support=observer
    metrics=MeshMetrics(case,support.metrics,support)
    def metric_details(raw):
        m,d,flags=metrics(raw)
        # Original reporting formulas, using the mesh-corrected normals.
        d['foot_load_bw']=m['force_w'][:,5:,2].sum(1)/m['bodyweight_N']
        d['torso_load_bw']=(m['force_w'][:,3:5]*m['normal_w'][:,3:5]).sum((1,2))/m['bodyweight_N']
        return m,d,flags
    support.metrics=metric_details
    original_cfg=support.make_cfg
    def config(args,gs):
        cfg=original_cfg(args,gs);assert len(gs)==1
        cfg.scene.num_envs=1;cfg.scene.rest_camera=None;cfg.scene.rest_floor.spawn.size=(5.,5.,.1)
        cfg.scene.rear_camera.data_types=['rgb','distance_to_image_plane']
        cfg.scene.seat=base.RigidObjectCfg(prim_path='{ENV_REGEX_NS}/Seat',spawn=base.sim.UsdFileCfg(usd_path=case['mesh_usd'],activate_contact_sensors=True),init_state=base.RigidObjectCfg.InitialStateCfg(pos=(0.,0.,0.)))
        cfg.events.reset_seats.func=reset_mesh;return cfg
    support.make_cfg=config
    old_init=transition.TransitionEnv.__init__;records=[];warm=[];capture_installed=False
    def initialize(env,*args,**kwargs):
        nonlocal capture_installed
        raw=args[0] if args else kwargs['raw']
        original_warm_step=raw.step;warm_flags=[]
        def warm_step(*sa,**sk):
            result=original_warm_step(*sa,**sk)
            warm_flags.append(bool(support.metrics(raw)[0]['failure'][0]));return result
        raw.step=warm_step
        try:old_init(env,*args,**kwargs)
        finally:raw.step=original_warm_step
        decision=env.step
        def first_step(*da,**dk):
            nonlocal capture_installed
            if not capture_installed:
                assert env.num_envs==1 and not env.warming.any() and not env.episode_length_buf.any()
                capture_installed=True;original_step=env.raw.step;original_close=env.raw.close
                previous=env.previous.detach().clone();hist=None
                write(out/'furniture_manifest.json',dict(case=case,variant=variant,teacher='preserved_best09',task='sitting',actor_truth_input=False,action_limits_unchanged=True,success_thresholds_unchanged=True))
                capture.save(env,observer,out,0)
                assert len(warm_flags)==25,('WARMUP_COUNT',len(warm_flags))
                warm.append(any(warm_flags))
                write(out/'furniture_startup.json',dict(warm_failure=warm,warm_steps=25,sensor_valid=env.geometry[:,11].tolist(),physics_steps=env.raw.sim.get_physics_step_count()))
                def raw_step(command,*ra,**rk):
                    nonlocal previous,hist
                    data=env.robot.data
                    frame=torch.cat((data.joint_pos.torch[:,:23],data.joint_vel.torch[:,:23]/10,data.applied_torque.torch[:,:23]/data.joint_effort_limits.torch[:,:23],data.projected_gravity_b.torch,data.root_ang_vel_b.torch/3,previous[:,:23],torch.as_tensor(env.geometry,device=env.device)),-1)
                    hist=frame[:,None].repeat(1,3,1) if hist is None else torch.cat((hist[:,1:],frame[:,None]),1)
                    row=dict(q_pre=data.joint_pos.torch.cpu().numpy().copy(),qd_pre=data.joint_vel.torch.cpu().numpy().copy(),previous=previous.cpu().numpy().copy(),command=command.cpu().numpy().copy(),sensor_history=hist.cpu().numpy().copy(),tick=np.array([len(records)]))
                    result=original_step(command,*ra,**rk);previous=command.detach().clone()
                    m,d,flags=support.metrics(env.raw)
                    row.update({'physical_'+k:v.detach().cpu().numpy().copy() for k,v in m.items()})
                    row.update(q=data.joint_pos.torch.cpu().numpy().copy(),body=data.body_pose_w.torch.cpu().numpy().copy(),root_height=d['root_height_m'].cpu().numpy().copy(),wrong_force=d['wrong_force_N'].cpu().numpy().copy(),peak_bw=(d['peak_force_N']/m['bodyweight_N']).cpu().numpy().copy(),failure=m['failure'].cpu().numpy().copy(),legal=d['contacts_legal'].cpu().numpy().copy(),no_slip=d['no_slip'].cpu().numpy().copy())
                    records.append(row)
                    assert len(records)<=1000
                    return result
                def close(*ca,**ck):
                    assert len(records)==1000,('INCOMPLETE_CAPTURE',len(records))
                    np.savez_compressed(out/'furniture_trajectory.npz',**{k:np.stack([r[k] for r in records]) for k in records[0]})
                    write(out/'contact_mapping.json',dict(validated=metrics.validated,frames=metrics.frames))
                    write(out/'furniture_capture.json',dict(samples=1000,physics_steps=env.raw.sim.get_physics_step_count(),body_names=env.robot.body_names,joint_names=env.robot.joint_names,origins=env.raw.scene.env_origins.cpu().tolist(),saved_before_simulator_close=True,teacher_commands_modified=False,full_standing_start=True))
                    return original_close(*ca,**ck)
                env.raw.step=raw_step;env.raw.close=close
            return decision(*da,**dk)
        env.step=first_step
    transition.TransitionEnv.__init__=initialize
    sys.argv=[str(ROOT/'scripts'/job['entry'])]+job['arguments']
    try:runpy.run_path(str(ROOT/'scripts'/job['entry']),run_name='__main__')
    except BaseException as e:
        if records:
            np.savez_compressed(out/'furniture_partial.npz',**{k:np.stack([r[k] for r in records]) for k in records[0]})
        write(out/'capture_failure.json',dict(error=repr(e),saved_active_controls=len(records),warmup_controls=len(warm_flags) if 'warm_flags' in locals() else 25))
        raise
if __name__=='__main__':main()
