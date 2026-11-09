"""One bounded known lean, adding a motor-axis torque readout only.

The original teacher receives its original sensors. New torque signals are
recorded for independent validation and cannot alter the controller or score.
"""
import os,sys,json,argparse,time
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent.parent
read=lambda p:json.loads(Path(p).read_text())
def write(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')

def main():
    from isaaclab.app import add_launcher_args
    p=argparse.ArgumentParser();add_launcher_args(p);a=p.parse_args()
    assert os.environ['CUDA_VISIBLE_DEVICES']=='GPU-ab1950b0-01dd-0fab-407f-2d17936b27f4'
    assert torch.cuda.device_count()==1 and '4090' in torch.cuda.get_device_name(0)
    prior=ROOT/'experiments/armchair_rest_ppo_20261105_v1'
    assets=ROOT/'experiments/furniture_examples_20261105_v1'
    sys.path[1:1]=[str(prior),str(ROOT/'scripts'),str(ROOT/'experiments/furniture_perception_repair_20261029_v1'),
                  str(ROOT/'experiments/furniture_validation_20261028_v2'),str(ROOT/'experiments/furniture_test_20261028_v1')]
    from demo import bank_for
    from capture import Observer,save
    from teacher import Teacher
    from environment import FullBodyEnv
    from contact_recording import RecordedMeshMetrics
    from isaaclab.sensors import JointWrenchSensorCfg
    import run_resting_posture as base
    import run_supported_transition as transition
    import run_supported_postures as support
    import scene_adapter,surface_map
    variant=next(v for v in read(assets/'protocol.json')['variants'] if v['id']=='lean_myd1WSucAz_right')
    case=next(c for c in read(assets/'cases.json') if c['id']==variant['case'])
    scene_adapter.HERE=assets;surface_map.ASSETS=assets
    observer=Observer('back');transition.observed_support=observer
    metrics=RecordedMeshMetrics(case,support.metrics,support);support.metrics=metrics
    bank,metadata=bank_for(variant)
    a.device='cuda:0';a.num_envs=1;a.seed=65101;a.mode='evaluate';a.depth_check=True;a.enable_cameras=True
    cfg=scene_adapter.make_cfg(a,case);cfg.scene.rear_camera.data_types=['rgb','distance_to_image_plane']
    cfg.scene.motor_axis_readout=JointWrenchSensorCfg(prim_path='{ENV_REGEX_NS}/Robot',update_period=0.)
    out=ROOT/'outputs'/HERE.name/'motor_axis_readout';out.mkdir(parents=True,exist_ok=False)
    write(out/'manifest.json',dict(case=case,variant=variant,profile='rear_back',purpose='Validate motor-axis torque observation; no controller change',
                                  raw_wrench_actor_input=False,original_teacher_unchanged=True,success_thresholds_unchanged=True))
    with base.launch_simulation(cfg,a):
        raw=base.gym.make('Humanoid-Sit-Learned-G1',cfg=cfg).unwrapped;env=FullBodyEnv(raw,a,out,bank,metadata)
        signals=[]
        try:
            write(out/'startup.json',env.startup);save(env,observer,out,0)
            assert not env.startup['warm_failure'][0] and env.startup['sensor_valid'][0]
            teacher=Teacher(env,observer,'rear_back')
            write(out/'plan.json',dict(nodes=teacher.nodes,times=teacher.t.tolist(),band=teacher.plan.band.tolist()))
            sensor=raw.scene['motor_axis_readout'];definitions={j['name']:j for j in env.model.joints}
            mapping=[(sensor.body_names.index(definitions[name]['body1'][0]),'XYZ'.index(definitions[name]['axis']))
                     for name in env.model.names]
            write(out/'motor_axis_contract.json',dict(sensor_body_names=sensor.body_names,joint_names=env.model.names,
                mapping=mapping,frame='child-side incoming joint frame',selected_component='torque about original actuated joint axis',
                sign=1,full_reaction_force_exposed_to_actor=False,validation_required=True))
            with torch.inference_mode():
                for tick in range(1000):
                    obs=env.observation().cpu().numpy().copy();previous=env.previous.cpu().numpy().copy()
                    data=env.robot.data;q=data.joint_pos.torch.cpu().numpy().copy();qd=data.joint_vel.torch.cpu().numpy().copy()
                    angular=sensor.data.torque.torch[0].cpu().numpy().copy()
                    motor=np.array([angular[b,axis] for b,axis in mapping],np.float32)
                    assert np.isfinite(motor).all()
                    gyro=data.root_ang_vel_b.torch.cpu().numpy().copy();gravity=data.projected_gravity_b.torch.cpu().numpy().copy()
                    stamp=raw.sim.get_physics_step_count();desired,signal=teacher.command(tick)
                    env.step(torch.as_tensor(desired,dtype=torch.float32,device=env.device)[None],record=True)
                    env.episode_records[-1].update(obs=obs,previous=previous,q_pre=q,qd_pre=qd,
                        desired=desired[None].astype(np.float32),motor_axis_effort_pre=motor[None],
                        motor_axis_physics_step=np.array([stamp]),gravity_pre=gravity,gyro_pre=gyro)
                    signals.append(signal)
                    if tick+1 in (350,650,1000):save(env,observer,out,tick+1)
                    if tick%200==199:print(tick+1,'alive',bool(env.alive[0]),'hold',int(env.held[0]),flush=True)
            env.save_records(out/'trajectory.npz');np.save(out/'sensor_control.npy',signals)
            write(out/'contact_mapping.json',dict(validated=metrics.validated,frames=metrics.frames))
            write(out/'raw_contact_buffers.json',metrics.evidence)
        except BaseException as e:
            if env.episode_records:env.save_records(out/'partial_trajectory.npz')
            write(out/'failure.json',dict(error=repr(e),recorded_controls=len(env.episode_records),physics_steps=raw.sim.get_physics_step_count()))
            raise
        finally:env.close()
if __name__=='__main__':main()
