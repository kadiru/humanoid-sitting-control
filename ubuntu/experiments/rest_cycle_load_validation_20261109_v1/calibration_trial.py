"""One first-cycle sitting calibration with shadow geometric separation; no PPO."""
import os,sys,json,argparse,hashlib
import signal as deadline_signal
import traceback
import subprocess,time
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
HERE=Path(__file__).resolve().parent; ROOT=HERE.parent.parent
OLD=ROOT/'experiments/furniture_repair_20261106_v1'
PRIOR=ROOT/'experiments/armchair_rest_ppo_20261105_v1'
ASSETS=HERE.parent/'rest_cycle_narrow_support_20261109_v1/assets'
sys.path[1:1]=[str(HERE),str(ROOT/'experiments/rest_cycle_partial_support_20261109_v1'),str(ROOT/'experiments/rest_cycle_contact_20261108_v1'),str(ROOT/'experiments/furniture_load_repair_20261106_v3'),str(PRIOR),str(OLD/'model_v4'),str(OLD),str(ROOT/'scripts'),str(ROOT/'sitting_task/src'),
              str(ROOT/'experiments/furniture_perception_repair_20261029_v1'),
              str(ROOT/'experiments/furniture_validation_20261028_v2'),
              str(ROOT/'experiments/furniture_test_20261028_v1'),
              str(ROOT/'experiments/furniture_load_estimator_20261106_v2')]
read=lambda p:json.loads(Path(p).read_text())
from recording import write


def main():
    from isaaclab.app import add_launcher_args
    parser=argparse.ArgumentParser();parser.add_argument('--name',required=True);add_launcher_args(parser);a=parser.parse_args()
    from humanoid_sitting.shared_frame import level_rotation,BodyModel
    from capture import Observer,save
    from environment import FullBodyEnv
    import run_resting_posture as base
    import run_supported_transition as transition
    import run_supported_postures as support
    import scene_adapter,surface_map
    from contact_recording import RecordedMeshMetrics
    assert os.environ['CUDA_VISIBLE_DEVICES']=='GPU-ab1950b0-01dd-0fab-407f-2d17936b27f4'
    assert torch.cuda.device_count()==1 and '4090' in torch.cuda.get_device_name(0)
    torch.set_num_threads(2)
    jobs=read(ROOT/'experiments/furniture_load_repair_20261106_v3/calibration_jobs.json');job=next(j for j in jobs if j['name']==a.name)
    job=dict(job,closer_m=read(HERE.parent/'rest_cycle_edge_visible_20261108_v1/preflight.json')['closer_m'])
    variant=next(v for v in read(ASSETS/'protocol.json')['variants'] if v['id']=='sit_narrow_support_050_center')
    case=next(c for c in read(ASSETS/'cases.json') if c['id']==variant['case'])
    # JSON integer normals are valid geometry; normalize floating arithmetic.
    original_normal=np.asarray(case['target_normal'])
    case=dict(case,target_normal=np.asarray(case['target_normal'],dtype=np.float64).tolist())
    assert np.array_equal(original_normal,np.asarray(case['target_normal']))
    metadata=read(ROOT/'outputs/full_body_contact_imitation_v4/dataset/sitting/metadata.json')
    metadata.update(cases=1,ids=[a.name])
    reset=np.load(ASSETS/'banks'/variant['id']/'reset_candidates.npz')
    bank={'reset_'+k:reset[k][:,0].copy() for k in ['q','qd','command','root_pose','root_velocity']}
    with np.load(ROOT/'outputs/full_body_contact_imitation_v4/dataset/sitting/bank.npz') as z:
        bank.update(q_post=z['q_post'][:,:1].copy(),reference=np.zeros(1,int),
                    phase_times=z['phase_times'][:1].copy(),admitted=np.ones(1,bool))
    with np.load(ROOT/'outputs/furniture_examples_20261105_v2/sit_modern_coffee_table_01_center/frame_0.npz') as z:
        basis=z['root_rotation']@level_rotation(z['gravity'].astype(float).copy()).T
    bank['reset_root_pose'][0,:3]-=basis@np.array([job['closer_m'],0.,0.])
    observer=Observer(variant['intent']);transition.observed_support=observer
    scene_adapter.HERE=ASSETS;surface_map.ASSETS=ASSETS
    metrics=RecordedMeshMetrics(case,support.metrics,support);support.metrics=metrics
    a.device='cuda:0';a.num_envs=1;a.seed=66302;a.mode='evaluate';a.depth_check=True;a.enable_cameras=True
    support.REAR_CAMERA=np.asarray(read(HERE.parent/'rest_cycle_partial_support_20261109_v1/camera_preflight.json')['camera_in_body'])
    cfg=scene_adapter.make_cfg(a,case);cfg.scene.rear_camera.data_types=['rgb','distance_to_image_plane']
    cfg.scene.rear_camera.width=384;cfg.scene.rear_camera.height=288
    cfg.scene.rear_camera.spawn.horizontal_aperture=39.
    cfg.scene.rear_camera.spawn.focal_length=15.
    assert a.name=='sit_modern_coffee_table_01_center_normal_landing'
    out=ROOT/'outputs'/HERE.name/'narrow_load_calibration';out.mkdir(parents=True,exist_ok=False)
    write(out/'manifest.json',dict(task='sitting',case=case,variant=dict(variant,id=a.name,closer_m=job['closer_m']),
        purpose='Bounded20s geometry-driven load calibration on the admitted narrow fixture; no sensor-qualified stage or stand-up claim',camera=dict(width=384,height=288,horizontal_aperture_mm=39.,focal_length_mm=15.,simulation_sensor_configuration_changed=True),
        estimated_force_controls_action=False,action_limits_unchanged=True,success_thresholds_unchanged=True))
    from observer import SpatialLoadObserver
    import warp as wp
    with base.launch_simulation(cfg,a):
        raw=base.gym.make('Humanoid-Sit-Learned-G1',cfg=cfg).unwrapped;env=FullBodyEnv(raw,a,out,bank,metadata)
        signals=[]
        try:
            write(out/'startup.json',env.startup);save(env,observer,out,0)
            assert not env.startup['warm_failure'][0] and env.startup['sensor_valid'][0]
            write(out/'current_snapshot_binding.json',dict(frame_sha256=hashlib.sha256((out/'frame_0.npz').read_bytes()).hexdigest(),
                previous_command=env.previous[0].cpu().numpy(),physics_step=raw.sim.get_physics_step_count()))
            # Freeze the new current snapshot, then independently recheck the
            # accepted path against it. No old rejected plan or new optimizer.
            check=subprocess.run([str(ROOT/'IsaacLab/.venv/bin/python'),str(HERE/'admit_calibration.py'),str(out)],cwd=ROOT,
                stdout=(out/'current_admission.log').open('x'),stderr=subprocess.STDOUT,timeout=46)
            assert check.returncode==0,'CURRENT_INPUT_PATH_ADMISSION_FAILED'
            admission=read(out/'current_admission.json');assert admission['passed']
            plan=read(HERE.parent/'rest_cycle_measured_support_20261109_v2/construction/plan.json')
            commands=np.asarray(plan['forward']['commands']);desired_path=np.asarray(plan['forward']['desired'])
            inertia=read('/data/humanoid-sitting-perception/crouch_rest_bridge_01/balanced_standing10_01/static_inertial_model.json')
            estimator=SpatialLoadObserver(env.model,inertia)
            index=0;quiet_count=0;started=False
            initial_g=env.robot.data.projected_gravity_b.torch[0].cpu().numpy().copy()
            initial_pitch=float(np.arctan2(initial_g[0],-initial_g[2]))
            with torch.inference_mode():
                for tick in range(1000):
                    data=env.robot.data;obs=env.observation().cpu().numpy().copy();previous=env.previous.cpu().numpy().copy()
                    q=data.joint_pos.torch[0].cpu().numpy().copy();qd=data.joint_vel.torch[0].cpu().numpy().copy()
                    gravity=data.projected_gravity_b.torch[0].cpu().numpy().copy();gyro=data.root_ang_vel_b.torch[0].cpu().numpy().copy()
                    tau=wp.to_torch(env.robot.root_view.get_dof_projected_joint_forces()).cpu().numpy()[0].copy()
                    value,condition,_,_=estimator.estimate(q,tau,gravity)
                    quiet=bool(abs(qd[:23]).max()<=.05 and abs(gyro[1])<=.02 and abs(np.arctan2(gravity[0],-gravity[2])-initial_pitch)<=np.deg2rad(.5))
                    valid=bool(env.geometry[0,11]>.5 and 0<=env.geometry[0,12]<=.21)
                    quiet_count=quiet_count+1 if quiet and valid else 0
                    if not started and quiet_count>=25:started=True
                    # Deliberately a bounded geometric calibration primitive.
                    # Its schedule never asserts contact/transfer/release success.
                    # The shadow estimator does NOT authorize any motion.
                    advance=started and valid and index<len(commands)
                    if advance:
                        desired=desired_path[index].copy();index+=1
                    elif index==len(commands):desired=commands[-1,:23].copy()
                    else:desired=previous[0,:23].copy()
                    env.step(torch.as_tensor(desired,dtype=torch.float32,device=env.device)[None],record=True)
                    env.episode_records[-1].update(obs=obs,previous=previous,q_pre=q[None],qd_pre=qd[None],
                        desired=desired[None].astype(np.float32),native_motor_tau_pre=tau[None],
                        gravity_pre=gravity[None],gyro_pre=gyro[None],
                        estimated_rest_force_pre=(np.zeros(3) if value is None else value[:3])[None],
                        estimate_available_pre=np.array([value is not None]),condition_pre=condition[None])
                    signals.append(dict(phase='calibration_path' if index<len(commands) else 'calibration_hold',
                        path_index=index,geometry_age_s=float(env.geometry[0,12]),geometry_valid=valid,quiet=quiet,
                        raw_max_qd_rad_s=float(abs(qd[:23]).max()),shadow_load_used_by_controller=False,
                        estimated_rest_force=None if value is None else value[:3].tolist()))
                    if tick+1 in (350,650,1000):save(env,observer,out,tick+1)
                    if tick%200==199:print('calibration',tick+1,'path',index,'hold',int(env.held[0]),flush=True)
            env.save_records(out/'trajectory.npz');write(out/'controller_signals.json',signals)
            write(out/'contact_mapping.json',dict(validated=metrics.validated,frames=metrics.frames))
            write(out/'capture_receipt.json',dict(standing_warmup_controls=25,active_controls=1000,physics_steps=raw.sim.get_physics_step_count(),
                path_samples_executed=index,path_samples_expected=len(commands),
                force_sensor_controls_action=False,joint_force_is_not_motor_torque=True,
                motor_axis_torque_not_full_incoming_wrench=True,PPO_updates=0,complete_cycle=False,
                purpose='Loaded calibration only; no return command or timer-asserted stage qualification'))
        except BaseException as e:
            write(out/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),controls=len(env.episode_records)))
            if env.episode_records:
                np.savez_compressed(out/'partial_trajectory.npz',**{k:np.stack([r[k] for r in env.episode_records]) for k in env.episode_records[0]})
            if signals:write(out/'partial_signals.json',signals)
            raise
        finally:
            env.close()

if __name__=='__main__':main()
