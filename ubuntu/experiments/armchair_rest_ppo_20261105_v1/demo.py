"""One full standing reset and20s demonstration, always saved including failure."""
import argparse,os,sys
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from common import ROOT,read,write
HERE=Path(__file__).resolve().parent
sys.path[1:1]=[str(ROOT/'scripts'),str(ROOT/'experiments/furniture_perception_repair_20261029_v1'),str(ROOT/'experiments/furniture_test_20261028_v1'),str(ROOT/'experiments/furniture_validation_20261028_v2')]
from capture import Observer,save
from teacher import Teacher

def bank_for(variant):
    folder=ROOT/'outputs/full_body_contact_imitation_v4/dataset/leaning';meta=read(folder/'metadata.json');meta.update(cases=1,ids=[variant['id']])
    with np.load(folder/'bank.npz') as z:
        bank={k:z[k][:1].copy() for k in ['reset_q','reset_qd','reset_command','reset_root_pose','reset_root_velocity','phase_times','admitted']}
        bank['q_post']=z['q_post'][:,:1].copy();bank['reference']=np.zeros(1,int)
    bank['reset_root_pose'][0,:2]+=np.array([variant['x'],variant['y']])
    R=Rotation.from_euler('z',variant['yaw_deg'],degrees=True)
    bank['reset_root_pose'][0,3:7]=(R*Rotation.from_quat(bank['reset_root_pose'][0,3:7])).as_quat()
    for k in [slice(0,3),slice(3,6)]:bank['reset_root_velocity'][0,k]=R.apply(bank['reset_root_velocity'][0,k])
    return bank,meta

def main():
    from isaaclab.app import add_launcher_args
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);p.add_argument('--variant',required=True,type=int);p.add_argument('--profile',required=True,choices=['rear_back','rear_pelvis']);add_launcher_args(p);a=p.parse_args()
    import run_resting_posture as base
    import run_supported_postures as support
    import run_supported_transition as transition
    from scene_adapter import make_cfg
    from contact_adapter import MeshMetrics
    from environment import FullBodyEnv
    assert os.environ['CUDA_VISIBLE_DEVICES']=='GPU-ab1950b0-01dd-0fab-407f-2d17936b27f4'
    assert torch.cuda.device_count()==1 and '4090' in torch.cuda.get_device_name(0)
    variant=read(HERE/'protocol.json')['demonstrations']['variants'][a.variant]
    case=next(c for c in read(ROOT/'experiments/furniture_test_20261028_v1/cases.json') if c['id']=='myd1WSucAz_rear_back')
    observer=Observer('back');transition.observed_support=observer
    adapter=MeshMetrics(case,support.metrics,support);support.metrics=adapter
    a.device='cuda:0';a.num_envs=1;a.seed=65001+a.variant;a.mode='evaluate';a.depth_check=True;a.enable_cameras=True
    bank,meta=bank_for(variant);cfg=make_cfg(a,case);cfg.scene.rear_camera.data_types=['rgb','distance_to_image_plane']
    out=a.output;out.mkdir(parents=True,exist_ok=False)
    write(out/'manifest.json',dict(case=case,variant=variant,profile=a.profile,seed=a.seed,action_limits_unchanged=True,success_thresholds_unchanged=True,actor_truth_input=False))
    with base.launch_simulation(cfg,a):
        raw=base.gym.make('Humanoid-Sit-Learned-G1',cfg=cfg).unwrapped;env=FullBodyEnv(raw,a,out,bank,meta)
        try:
            write(out/'startup.json',env.startup);assert not env.startup['warm_failure'][0] and env.startup['sensor_valid'][0]
            save(env,observer,out,0);teacher=Teacher(env,observer,a.profile)
            write(out/'plan.json',dict(nodes=teacher.nodes,times=teacher.t.tolist(),profile=a.profile,band=teacher.plan.band.tolist()))
            signals=[]
            with torch.inference_mode():
                for tick in range(1000):
                    obs=env.observation().cpu().numpy().copy();prev=env.previous.cpu().numpy().copy()
                    q=env.robot.data.joint_pos.torch.cpu().numpy().copy();qd=env.robot.data.joint_vel.torch.cpu().numpy().copy()
                    desired,signal=teacher.command(tick)
                    env.step(torch.as_tensor(desired,dtype=torch.float32,device=env.device)[None],record=True)
                    env.episode_records[-1].update(obs=obs,previous=prev,q_pre=q,qd_pre=qd,desired=desired[None].astype(np.float32))
                    signals.append(signal)
                    if tick+1 in (10,150,350,500,650,750,1000):save(env,observer,out,tick+1)
                    if tick%200==199:print(tick+1,'alive',bool(env.alive[0]),'held',int(env.held[0]),flush=True)
            env.save_records(out/'trajectory.npz');np.save(out/'sensor_control.npy',signals)
            write(out/'contact_mapping.json',dict(validated=adapter.validated,frames=adapter.frames))
        except ValueError as e:
            # A geometric infeasibility is recorded; never silently switch posture.
            write(out/'planning_failure.json',dict(error=repr(e),ticks=env.t));raise
        finally:env.close()
if __name__=='__main__':main()
