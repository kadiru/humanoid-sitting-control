"""A fresh complete furniture rollout; the repaired estimator is measurement-only."""
import os,sys,json,argparse,hashlib
from pathlib import Path
import numpy as np
import torch
from json_types import encode
from scipy.spatial.transform import Rotation
HERE=Path(__file__).resolve().parent; ROOT=HERE.parent.parent
OLD=ROOT/'experiments/furniture_repair_20261106_v1'
PRIOR=ROOT/'experiments/armchair_rest_ppo_20261105_v1'
ASSETS=ROOT/'experiments/furniture_examples_20261105_v1'
sys.path[1:1]=[str(ROOT/'experiments/furniture_load_repair_20261106_v3'),str(PRIOR),str(OLD/'model_v4'),str(OLD),str(ROOT/'scripts'),str(ROOT/'sitting_task/src'),
              str(ROOT/'experiments/furniture_perception_repair_20261029_v1'),
              str(ROOT/'experiments/furniture_validation_20261028_v2'),
              str(ROOT/'experiments/furniture_test_20261028_v1'),
              str(ROOT/'experiments/furniture_load_estimator_20261106_v2')]
read=lambda p:json.loads(Path(p).read_text())
def write(p,d): p.write_text(json.dumps(d,indent=2,allow_nan=False,default=encode)+'\n')


def main():
    from isaaclab.app import add_launcher_args
    parser=argparse.ArgumentParser();parser.add_argument('--name',required=True);add_launcher_args(parser);a=parser.parse_args()
    from humanoid_sitting.shared_frame import level_rotation,BodyModel
    from seat_plan import SeatPlan
    from legacy_normal_landing import NormalLanding
    from load_estimator import QualifiedLoadEstimator
    from estimator import Estimator
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
    jobs=read(HERE/'calibration_jobs.json');job=next(j for j in jobs if j['name']==a.name)
    variant=next(v for v in read(ASSETS/'protocol.json')['variants'] if v['id']==job['source_case'])
    case=next(c for c in read(ASSETS/'cases.json') if c['id']==variant['case'])
    assert read(Path(job['model_directory'])/(job['source_case']+'.audit.json'))['model_admitted']
    construction=read(Path(job['model_directory'])/(job['source_case']+'.plan.json'))
    path=read(Path(job['model_directory'])/(job['source_case']+'.complete.json'))
    metadata=read(ROOT/'outputs/full_body_contact_imitation_v4/dataset/sitting/metadata.json')
    metadata.update(cases=1,ids=[a.name])
    reset=np.load(ASSETS/'banks'/variant['id']/'reset_candidates.npz')
    bank={'reset_'+k:reset[k][:,0].copy() for k in ['q','qd','command','root_pose','root_velocity']}
    with np.load(ROOT/'outputs/full_body_contact_imitation_v4/dataset/sitting/bank.npz') as z:
        bank.update(q_post=z['q_post'][:,:1].copy(),reference=np.zeros(1,int),
                    phase_times=z['phase_times'][:1].copy(),admitted=np.ones(1,bool))
    with np.load(ROOT/'outputs/furniture_examples_20261105_v2'/variant['id']/'frame_0.npz') as z:
        basis=z['root_rotation']@level_rotation(z['gravity'].astype(float).copy()).T
    bank['reset_root_pose'][0,:3]-=basis@np.array([job['closer_m'],0.,0.])
    observer=Observer(variant['intent']);transition.observed_support=observer
    scene_adapter.HERE=ASSETS;surface_map.ASSETS=ASSETS
    metrics=RecordedMeshMetrics(case,support.metrics,support);support.metrics=metrics
    a.device='cuda:0';a.num_envs=1;a.seed=66301+jobs.index(job);a.mode='evaluate';a.depth_check=True;a.enable_cameras=True
    cfg=scene_adapter.make_cfg(a,case);cfg.scene.rear_camera.data_types=['rgb','distance_to_image_plane']
    out=ROOT/'outputs'/HERE.name/'calibration_demos'/a.name;out.mkdir(parents=True,exist_ok=False)
    write(out/'manifest.json',dict(task='sitting',case=case,variant=dict(variant,id=a.name,closer_m=job['closer_m']),
        purpose='Complete standing-start demonstration candidate and load calibration',
        estimated_force_controls_action=False,action_limits_unchanged=True,success_thresholds_unchanged=True))
    models=[]
    for seed in [66201,66202,66203]:
        state=torch.load(ROOT/f'experiments/furniture_load_estimator_20261106_v2/estimator_{seed}.pt',map_location='cpu',weights_only=True)
        model=Estimator(state['mean'],state['scale']);model.load_state_dict(state);models.append(model.eval())
    def predict(x):
        with torch.no_grad(): values=torch.stack([m(torch.from_numpy(x)[None])[0] for m in models])
        return values.mean(0).numpy(),values.std(0,unbiased=False).numpy()
    with np.load(ROOT/'experiments/furniture_load_repair_20261106_v3/sensor_envelope.npz') as z:
        estimator=QualifiedLoadEstimator(predict,z['low'],z['high'],z['calibration_bound'])
    inertia=read('/data/humanoid-sitting-perception/crouch_rest_bridge_01/balanced_standing10_01/static_inertial_model.json')
    shapes=read(ROOT/'experiments/furniture_perception_repair_20261029_v1/render_colliders.json')['shapes']
    with base.launch_simulation(cfg,a):
        raw=base.gym.make('Humanoid-Sit-Learned-G1',cfg=cfg).unwrapped;env=FullBodyEnv(raw,a,out,bank,metadata)
        signals=[]
        try:
            write(out/'startup.json',env.startup);save(env,observer,out,0)
            assert not env.startup['warm_failure'][0] and env.startup['sensor_valid'][0]
            data=env.robot.data;q=data.joint_pos.torch[0].cpu().numpy().copy();g=data.projected_gravity_b.torch[0].cpu().numpy().copy()
            plan=SeatPlan(env.model,inertia,shapes,q,g,env.geometry[0],observer.last[1]['selected_points'],env.previous[0].cpu().numpy())
            assert np.max(abs(q-np.asarray(construction['nodes'][0]['q'])))<2e-5,'RESET_MODEL_STATE_MISMATCH'
            geometry=audit_geometry(plan,construction);commands=audit_commands(plan,construction,path)
            write(out/'fresh_geometry_admission.json',dict(geometry=geometry,commands=commands))
            if not all(v for k,v in geometry['tests'].items() if k!='time_for_original_hold') or not commands['passed']:
                raise SystemExit(2)
            teacher=NormalLanding(plan,construction,path)
            with torch.inference_mode():
                for tick in range(1000):
                    obs=env.observation().cpu().numpy().copy();prev=env.previous.cpu().numpy().copy()
                    q=data.joint_pos.torch.cpu().numpy().copy();qd=data.joint_vel.torch.cpu().numpy().copy()
                    gravity=data.projected_gravity_b.torch[0].cpu().numpy().copy();gyro=data.root_ang_vel_b.torch[0].cpu().numpy().copy()
                    reading=estimator.read(obs[0,:333])
                    desired,signal=teacher.command(tick,q[0],qd[0],gravity,gyro,env.geometry[0].copy(),prev[0])
                    signal.update(load_state=reading.state,load_reason=reading.reason,
                                  load_mean_bw=None if reading.mean_bw is None else reading.mean_bw.tolist(),
                                  load_lower_bw=None if reading.lower_bw is None else reading.lower_bw.tolist())
                    env.step(torch.as_tensor(desired,dtype=torch.float32,device=env.device)[None],record=True)
                    env.episode_records[-1].update(obs=obs,previous=prev,q_pre=q,qd_pre=qd,desired=desired[None].astype(np.float32))
                    signals.append(signal)
                    if tick+1 in (350,650,1000):save(env,observer,out,tick+1)
                    if tick%200==199:print(tick+1,teacher.phase,'hold',int(env.held[0]),flush=True)
            env.save_records(out/'trajectory.npz');write(out/'controller_signals.json',signals)
            write(out/'contact_mapping.json',dict(validated=metrics.validated,frames=metrics.frames))
        except BaseException as e:
            if env.episode_records:
                np.savez_compressed(out/'partial_trajectory.npz',**{k:np.stack([r[k] for r in env.episode_records]) for k in env.episode_records[0]})
            write(out/'failure.json',dict(error=repr(e),controls=len(env.episode_records)))
            raise
        finally:env.close()


if __name__=='__main__':main()
