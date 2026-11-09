"""Single independent 20-second standing-start box-armchair test, 4090 only."""
import argparse,os,sys,time,json
from pathlib import Path
import numpy as np
import torch
from plan import ROOT,LeanPlan
from wrench import WrenchObserver
HERE=Path(__file__).resolve().parent
CAL=ROOT/'experiments/furniture_perception_repair_20261029_v1'
SRC=ROOT/'experiments/full_body_contact_trainer_4090_step_v5'
ASSETS=ROOT/'experiments/furniture_test_20261028_v1'
CHECK=ROOT/'experiments/furniture_validation_20261028_v2'
sys.path[:0]=[str(SRC),str(ROOT/'scripts'),str(CAL),str(ASSETS),str(CHECK)]
from common import read,write
from capture import Observer,save
CASE='myd1WSucAz_rear_back'

def main():
    from isaaclab.app import add_launcher_args
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--trial',required=True,type=int,choices=[0,1,2])
    add_launcher_args(parser);a=parser.parse_args()
    import run_resting_posture as base
    import run_supported_postures as support
    import run_supported_transition as transition
    from scene_adapter import make_cfg
    from contact_adapter import MeshMetrics
    from environment import FullBodyEnv
    assert os.environ['CUDA_VISIBLE_DEVICES']=='GPU-ab1950b0-01dd-0fab-407f-2d17936b27f4'
    assert torch.cuda.device_count()==1 and '4090' in torch.cuda.get_device_name(0)
    case=next(c for c in read(ASSETS/'cases.json') if c['id']==CASE)
    observer=Observer('back'); transition.observed_support=observer
    adapter=MeshMetrics(case,support.metrics,support);support.metrics=adapter
    a.device='cuda:0';a.num_envs=1;a.seed=64901+a.trial;a.mode='evaluate';a.depth_check=True;a.enable_cameras=True
    folder=ROOT/'outputs/full_body_contact_imitation_v4/dataset/leaning'
    meta=read(folder/'metadata.json');meta.update(cases=1,ids=[CASE])
    with np.load(folder/'bank.npz') as z:
        bank={k:z[k][:1].copy() for k in ['reset_q','reset_qd','reset_command','reset_root_pose','reset_root_velocity','phase_times','admitted']}
        bank['q_post']=z['q_post'][:,:1].copy();bank['reference']=np.zeros(1,int)
    dy=[0.,.015,-.015][a.trial];bank['reset_root_pose'][0,1]+=dy
    cfg=make_cfg(a,case);cfg.scene.rear_camera.data_types=['rgb','distance_to_image_plane']
    out=a.output;out.mkdir(exist_ok=False,parents=True)
    with base.launch_simulation(cfg,a):
        raw=base.gym.make('Humanoid-Sit-Learned-G1',cfg=cfg).unwrapped
        env=FullBodyEnv(raw,a,out,bank,meta); write(out/'startup.json',env.startup)
        try:
            assert not env.startup['warm_failure'][0] and env.startup['sensor_valid'][0]
            info=observer.last[1];q=env.robot.data.joint_pos.torch[0].cpu().numpy()
            imu=env.robot.data.projected_gravity_b.torch[0].cpu().numpy()
            prev=env.previous[0].cpu().numpy().copy()
            plan=LeanPlan(env.model,q,imu,observer.last[0],info['selected_points'],prev)
            nodes,t=plan.solve();assert t[-1]<14.4
            x=np.array([n['cmd'] for n in nodes]);pitch=np.array([n['pitch'] for n in nodes]);gap=np.array([n['gap'] for n in nodes])
            base_target=prev[:23].copy()
            write(out/'plan.json',dict(times=t.tolist(),nodes=nodes,sole_range=plan.foot_bounds.tolist(),target_band=plan.band.tolist()))
            write(out/'manifest.json',dict(case=case,source=str(SRC),seed=a.seed,lateral_start_m=dy,
                contact_bodies=[x[1] for x in support.CONTACTS],actor_truth_input=False,action_limits_unchanged=True,success_thresholds_unchanged=True,
                controller='Camera-selected interior, fixed-foot path, static feedforward, IMU pitch/gyro and rear-camera gap feedback.'))
            signals=[];wrench_log=[]
            wrench=WrenchObserver(plan,nodes)
            bias_samples=[]
            plane_latched=False;trim=0.;filtered_pitch=0.;filtered_load=0.;hold_target=None
            approach=t[next(j for j,n in enumerate(nodes) if n["phase"]=="approach")]
            load_time=t[next(j for j,n in enumerate(nodes) if n["phase"]=="load")]
            optical=0.
            save(env,observer,out,0)
            with torch.inference_mode():
                for tick in range(1000):
                    sec=(tick+1)*.02
                    gravity=env.robot.data.projected_gravity_b.torch[0].cpu().numpy()
                    ang=float(env.robot.data.root_ang_vel_b.torch[0,1].cpu())
                    actual_pitch=np.arctan2(gravity[0],-gravity[2])
                    pref=np.interp(sec,t,pitch);gref=np.interp(sec,t,gap)
                    g=env.geometry[0]
                    # At small optical ranges the depth image sees through the
                    # near clip to a parallel face. Reject this measured jump;
                    # the initial camera plane remains in the fixed-foot plan.
                    if gref<.04 and g[11] and g[1]-gref>.05:plane_latched=True
                    if not plane_latched and g[11] and g[12]<.21:
                        optical=float(np.clip(.9*(g[1]-gref),-.025,.025))
                    idx=int(np.abs(t-sec).argmin())
                    tau=env.robot.data.applied_torque.torch[0].cpu().numpy()[plan.ids].sum(0)
                    if approach-.6<sec<approach+.4:
                        bias_samples.append(wrench.tables[idx][:,0]-tau)
                        wrench.bias=np.median(bias_samples,axis=0)
                    estimated=wrench.estimate(idx,tau)
                    filtered_load+=.06*(estimated[0]/plan.weight-filtered_load)
                    filtered_pitch+=.08*(.72*(actual_pitch-pref)+.12*ang-filtered_pitch)
                    if sec>load_time:
                        # Normal compliance while approaching the loaded target.
                        trim+=np.clip(.18*(.12-filtered_load),-.015,.015)*.02
                        trim=float(np.clip(trim,-.035,.04))
                    fb=np.clip(filtered_pitch+optical+trim,-.085,.085)
                    goal=base_target.copy()
                    for j,ids in enumerate(plan.ids.T):
                        goal[ids]+=np.interp(sec,t,x[:,j])-plan.prev[j]
                    goal[plan.ids[:,2]]+=fb
                    # Stop chasing per-tick IMU jitter once actually loaded.
                    # A constant target removes the observed 3-tick command cycle.
                    if sec>t[-1]+.30 and hold_target is None and .075<filtered_load<.17:
                        hold_target=goal.copy()
                    if hold_target is not None:goal=hold_target
                    wrench_log.append([sec,estimated[0],estimated[1],filtered_load,trim,plane_latched,hold_target is not None])
                    prior=env.previous[:,env.ids]
                    delta=(torch.as_tensor(goal,device=env.device)[None]-prior)/.003
                    env.step(torch.atanh(delta.clamp(-.99999,.99999)),record=True)
                    signals.append([sec,actual_pitch,pref,ang,fb,optical,gref,*g.copy()])
                    if tick+1 in (10,20,150,350,500,650,750,1000):save(env,observer,out,tick+1)
                    if tick%100==99: print(tick+1,'alive',bool(env.alive[0]),'held',int(env.held[0]),flush=True)
            env.save_records(out/'trajectory.npz')
            np.save(out/'sensor_control.npy',signals)
            np.save(out/'wrench_control.npy',wrench_log)
            write(out/'contact_mapping.json',dict(validated=adapter.validated,frames=adapter.frames,
                thresholds=dict(surface_distance_m=.015,perimeter_inset_m=.01),multiple_normal_policy='unvalidated'))
        finally:env.close()

if __name__=='__main__':main()
