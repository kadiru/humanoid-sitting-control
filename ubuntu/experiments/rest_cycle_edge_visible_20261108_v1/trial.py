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
ASSETS=HERE.parent/'rest_cycle_high_support_20261108_v1/assets'
sys.path[1:1]=[str(ROOT/'experiments/rest_cycle_contact_20261108_v1'),str(ROOT/'experiments/furniture_load_repair_20261106_v3'),str(PRIOR),str(OLD/'model_v4'),str(OLD),str(ROOT/'scripts'),str(ROOT/'sitting_task/src'),
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
    from seat_plan import SeatPlan
    from normal_landing import NormalLanding
    from separation import shadow_separation
    from capture import Observer,save
    from environment import FullBodyEnv
    from audit_seat_plan import audit as audit_geometry
    from audit_complete_path import audit as audit_commands
    import run_resting_posture as base
    import run_supported_transition as transition
    import run_supported_postures as support
    import scene_adapter,surface_map
    from contact_recording import RecordedMeshMetrics
    assert os.environ['CUDA_VISIBLE_DEVICES']=='GPU-ab1950b0-01dd-0fab-407f-2d17936b27f4'
    assert torch.cuda.device_count()==1 and '4090' in torch.cuda.get_device_name(0)
    torch.set_num_threads(2)
    jobs=read(ROOT/'experiments/furniture_load_repair_20261106_v3/calibration_jobs.json');job=next(j for j in jobs if j['name']==a.name)
    job=dict(job,closer_m=read(HERE/'preflight.json')['closer_m'])
    variant=next(v for v in read(ASSETS/'protocol.json')['variants'] if v['id']=='sit_high_support_050_center')
    case=next(c for c in read(ASSETS/'cases.json') if c['id']==variant['case'])
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
    cfg=scene_adapter.make_cfg(a,case);cfg.scene.rear_camera.data_types=['rgb','distance_to_image_plane']
    cfg.scene.rear_camera.width=384;cfg.scene.rear_camera.height=288
    cfg.scene.rear_camera.spawn.horizontal_aperture=39.
    cfg.scene.rear_camera.spawn.focal_length=15.
    assert a.name=='sit_modern_coffee_table_01_center_normal_landing'
    out=ROOT/'outputs'/HERE.name/'sit_edge_visible';out.mkdir(parents=True,exist_ok=False)
    write(out/'manifest.json',dict(task='sitting',case=case,variant=dict(variant,id=a.name,closer_m=job['closer_m']),
        purpose='Higher50cm support subtask; return phase requires qualified sensing and separate path admission',camera=dict(width=384,height=288,horizontal_aperture_mm=39.,focal_length_mm=15.,simulation_sensor_configuration_changed=True),
        estimated_force_controls_action=False,action_limits_unchanged=True,success_thresholds_unchanged=True))
    inertia=read('/data/humanoid-sitting-perception/crouch_rest_bridge_01/balanced_standing10_01/static_inertial_model.json')
    shapes=read(ROOT/'experiments/furniture_perception_repair_20261029_v1/render_colliders.json')['shapes']
    with base.launch_simulation(cfg,a):
        raw=base.gym.make('Humanoid-Sit-Learned-G1',cfg=cfg).unwrapped;env=FullBodyEnv(raw,a,out,bank,metadata)
        signals=[]
        try:
            write(out/'startup.json',env.startup);save(env,observer,out,0)
            assert not env.startup['warm_failure'][0] and env.startup['sensor_valid'][0]
            data=env.robot.data;q=data.joint_pos.torch[0].cpu().numpy().copy();g=data.projected_gravity_b.torch[0].cpu().numpy().copy()
            K=raw.scene['rear_camera'].data.intrinsic_matrices.torch[0].cpu().numpy()
            expected_K=np.array([[384.*15./39.,0.,192.],[0.,384.*15./39.,144.],[0.,0.,1.]])
            assert np.max(abs(K-expected_K))<2e-5, ('CAMERA_INTRINSICS_MISMATCH',K.tolist())
            # Freeze current snapshot before the CPU-only cold construction.
            write(out/'current_snapshot_binding.json',dict(frame_sha256=hashlib.sha256((out/'frame_0.npz').read_bytes()).hexdigest(),
                  previous_command=env.previous[0].cpu().numpy(),physics_step=raw.sim.get_physics_step_count()))
            model_start=time.monotonic()
            with (out/'construction.log').open('x') as model_log:
                result=subprocess.run([str(ROOT/'IsaacLab/.venv/bin/python'),str(HERE/'construct.py')],cwd=ROOT,
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'),
                    stdout=model_log,stderr=subprocess.STDOUT,timeout=45.)
            write(out/'construction_process_receipt.json',dict(returncode=result.returncode,wall_seconds=time.monotonic()-model_start))
            assert result.returncode==0,'CURRENT_CAMERA_CONSTRUCTION_EXECUTION_FAILED'
            model_out=out/'model_construction'
            admission=read(model_out/'audit.json')
            assert admission['model_admitted'],'CURRENT_CAMERA_MODEL_NOT_ADMITTED'
            construction=read(model_out/'plan.json');path=read(model_out/'complete.json')
            assert raw.sim.get_physics_step_count()==100,'PHYSICS_ADVANCED_DURING_MODEL_CONSTRUCTION'
            plan=SeatPlan(env.model,inertia,shapes,q,g,env.geometry[0],observer.last[1]['selected_points'],env.previous[0].cpu().numpy())
            assert np.array_equal(q,np.asarray(construction['nodes'][0]['q'])),'RESET_MODEL_STATE_MISMATCH'
            assert read(model_out/'input_binding.json')['sha256']==hashlib.sha256((out/'frame_0.npz').read_bytes()).hexdigest()
            write(out/'fresh_geometry_admission.json',dict(geometry=admission['geometry_audit'],commands=admission['command_audit'],clearance=admission['clearance_audit'],source=str(model_out/'audit.json')))
            teacher=NormalLanding(plan,construction,path)
            from substep_recorder import SubstepRecorder
            recorder=SubstepRecorder(env,support)
            recorder.install()
            with torch.inference_mode():
                for tick in range(1000):
                    obs=env.observation().cpu().numpy().copy();prev=env.previous.cpu().numpy().copy()
                    q=data.joint_pos.torch.cpu().numpy().copy();qd=data.joint_vel.torch.cpu().numpy().copy()
                    gravity=data.projected_gravity_b.torch[0].cpu().numpy().copy();gyro=data.root_ang_vel_b.torch[0].cpu().numpy().copy()
                    separation=shadow_separation(env.model,q[0],env.geometry[0].copy())
                    desired,diagnostics=teacher.command(tick,q[0],qd[0],gravity,gyro,env.geometry[0].copy(),prev[0])
                    diagnostics.update(separation_shadow=separation,load_estimator_used=False)
                    env.step(torch.as_tensor(desired,dtype=torch.float32,device=env.device)[None],record=True)
                    env.episode_records[-1].update(obs=obs,previous=prev,q_pre=q,qd_pre=qd,desired=desired[None].astype(np.float32))
                    signals.append(diagnostics)
                    if tick+1 in (350,650,1000):save(env,observer,out,tick+1)
                    if tick%200==199:print(tick+1,teacher.phase,'hold',int(env.held[0]),flush=True)
            recorder.uninstall()
            recorder.save(out/'substeps.npz')
            env.save_records(out/'trajectory.npz');write(out/'controller_signals.json',signals)
            write(out/'contact_mapping.json',dict(validated=metrics.validated,frames=metrics.frames))
            write(out/'cycle_gate.json',dict(sitting_controls=1000,return_controls=0,full_cycle=False,
                return_admitted=False,reason='Qualified sensor load/unload adapter and return primitive not yet validated; do not substitute contact truth or time for qualification.'))
        except BaseException as e:
            if env.episode_records:
                np.savez_compressed(out/'partial_trajectory.npz',**{k:np.stack([r[k] for r in env.episode_records]) for k in env.episode_records[0]})
            write(out/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),controls=len(env.episode_records)))
            raise
        finally:
            deadline_signal.alarm(0)
            if 'recorder' in locals():
                recorder.uninstall()
                if recorder.rows and not (out/'substeps.npz').exists():recorder.save(out/'substeps_partial.npz')
            env.close()


if __name__=='__main__':main()
