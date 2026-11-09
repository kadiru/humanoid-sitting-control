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
ASSETS=HERE/'assets'
sys.path[1:1]=[str(ROOT/'experiments/rest_cycle_partial_support_20261109_v1'),str(ROOT/'experiments/rest_cycle_contact_20261108_v1'),str(ROOT/'experiments/furniture_load_repair_20261106_v3'),str(PRIOR),str(OLD/'model_v4'),str(OLD),str(ROOT/'scripts'),str(ROOT/'sitting_task/src'),
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
    out=ROOT/'outputs'/HERE.name/'camera_and_wrench_capture';out.mkdir(parents=True,exist_ok=False)
    write(out/'manifest.json',dict(task='sitting',case=case,variant=dict(variant,id=a.name,closer_m=job['closer_m']),
        purpose='Standing-only current-camera observation of the dedicated12cm narrow support fixture; no sit/hold/stand claim',camera=dict(width=384,height=288,horizontal_aperture_mm=39.,focal_length_mm=15.,simulation_sensor_configuration_changed=True),
        estimated_force_controls_action=False,action_limits_unchanged=True,success_thresholds_unchanged=True))
    from joint_wrench_recorder import JointWrenchRecorder
    with base.launch_simulation(cfg,a):
        raw=base.gym.make('Humanoid-Sit-Learned-G1',cfg=cfg).unwrapped;env=FullBodyEnv(raw,a,out,bank,metadata)
        recorder=JointWrenchRecorder(env,support)
        try:
            write(out/'startup.json',env.startup);save(env,observer,out,0)
            assert not env.startup['warm_failure'][0] and env.startup['sensor_valid'][0]
            write(out/'current_snapshot_binding.json',dict(frame_sha256=hashlib.sha256((out/'frame_0.npz').read_bytes()).hexdigest(),
                previous_command=env.previous[0].cpu().numpy(),physics_step=raw.sim.get_physics_step_count()))
            recorder.install()
            initial_command=env.previous.clone()
            with torch.inference_mode():
                for tick in range(50):
                    previous=env.previous.cpu().numpy().copy()
                    desired=env.previous[:,:23].clone()
                    env.step(desired,record=True)
                    assert torch.equal(env.previous,initial_command),'STANDING_CAPTURE_CHANGED_COMMAND'
                    env.episode_records[-1].update(previous=previous,desired=desired.cpu().numpy().copy())
            recorder.uninstall();recorder.save(out/'shadow_wrench.npz')
            save(env,observer,out,50)
            np.savez_compressed(out/'standing_calibration.npz',**{k:np.stack([r[k] for r in env.episode_records]) for k in env.episode_records[0]})
            write(out/'capture_receipt.json',dict(standing_controls=50,warmup_controls=25,physics_steps=raw.sim.get_physics_step_count(),
                sitting_controls=0,force_sensor_controls_action=False,joint_force_is_not_motor_torque=True,
                sensor_layout_changed=True,physical_sensor_hardware_validated=False,PPO_updates=0,complete_cycle=False))
        except BaseException as e:
            write(out/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),controls=len(env.episode_records)))
            raise
        finally:
            recorder.uninstall()
            if recorder.rows and not (out/'shadow_wrench.npz').exists():recorder.save(out/'shadow_wrench_partial.npz')
            env.close()

if __name__=='__main__':main()
