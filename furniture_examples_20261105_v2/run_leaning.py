"""Reuse the successful rear-camera back teacher on the frozen furniture starts."""
import os,sys,json,argparse,importlib.util
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent.parent
read=lambda p:json.loads(Path(p).read_text())
def write(p,d):Path(p).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')

def main():
    from isaaclab.app import add_launcher_args
    p=argparse.ArgumentParser();p.add_argument('--variant',required=True);add_launcher_args(p);a=p.parse_args()
    variant=next(v for v in read(HERE/'protocol.json')['variants'] if v['id']==a.variant);assert variant['task']=='leaning'
    case=next(c for c in read(HERE/'cases.json') if c['id']==variant['case'])
    assert os.environ['CUDA_VISIBLE_DEVICES']=='GPU-ab1950b0-01dd-0fab-407f-2d17936b27f4'
    assert torch.cuda.device_count()==1 and '4090' in torch.cuda.get_device_name(0)
    prior=ROOT/'experiments/armchair_rest_ppo_20261105_v1'
    sys.path[:0]=[str(prior),str(ROOT/'scripts'),str(ROOT/'experiments/furniture_perception_repair_20261029_v1'),str(ROOT/'experiments/furniture_validation_20261028_v2'),str(ROOT/'experiments/furniture_test_20261028_v1')]
    from demo import bank_for
    from capture import Observer,save
    from teacher import Teacher
    from environment import FullBodyEnv
    import run_resting_posture as base
    import run_supported_transition as transition
    import run_supported_postures as support
    import scene_adapter,surface_map
    from contact_adapter import MeshMetrics
    scene_adapter.HERE=HERE;surface_map.ASSETS=HERE
    observer=Observer('back');transition.observed_support=observer
    metrics=MeshMetrics(case,support.metrics,support);support.metrics=metrics
    bank,meta=bank_for(variant)
    a.device='cuda:0';a.num_envs=1;a.seed=65101;a.mode='evaluate';a.depth_check=True;a.enable_cameras=True
    cfg=scene_adapter.make_cfg(a,case);cfg.scene.rear_camera.data_types=['rgb','distance_to_image_plane']
    out=ROOT/'outputs'/HERE.name/variant['id'];out.mkdir(parents=True,exist_ok=False)
    write(out/'manifest.json',dict(case=case,variant=variant,profile='rear_back',task='leaning',actor_truth_input=False,action_limits_unchanged=True,success_thresholds_unchanged=True))
    with base.launch_simulation(cfg,a):
        raw=base.gym.make('Humanoid-Sit-Learned-G1',cfg=cfg).unwrapped;env=FullBodyEnv(raw,a,out,bank,meta)
        try:
            write(out/'startup.json',env.startup);save(env,observer,out,0)
            assert not env.startup['warm_failure'][0] and env.startup['sensor_valid'][0]
            try:teacher=Teacher(env,observer,'rear_back')
            except (ValueError,AssertionError) as e:
                write(out/'planning_failure.json',dict(error=repr(e),ticks=env.t,physics_steps=raw.sim.get_physics_step_count()));raise
            write(out/'plan.json',dict(nodes=teacher.nodes,times=teacher.t.tolist(),band=teacher.plan.band.tolist()))
            signals=[]
            with torch.inference_mode():
                for tick in range(1000):
                    obs=env.observation().cpu().numpy().copy();prev=env.previous.cpu().numpy().copy()
                    q=env.robot.data.joint_pos.torch.cpu().numpy().copy();qd=env.robot.data.joint_vel.torch.cpu().numpy().copy()
                    desired,signal=teacher.command(tick);env.step(torch.as_tensor(desired,dtype=torch.float32,device=env.device)[None],record=True)
                    env.episode_records[-1].update(obs=obs,previous=prev,q_pre=q,qd_pre=qd,desired=desired[None].astype(np.float32));signals.append(signal)
                    if tick+1 in (10,350,650,750,1000):save(env,observer,out,tick+1)
                    if tick%200==199:print(tick+1,'alive',bool(env.alive[0]),'hold',int(env.held[0]),flush=True)
            env.save_records(out/'trajectory.npz');np.save(out/'sensor_control.npy',signals)
            write(out/'contact_mapping.json',dict(validated=metrics.validated,frames=metrics.frames))
        finally:env.close()
if __name__=='__main__':main()
