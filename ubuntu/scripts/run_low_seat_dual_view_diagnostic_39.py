"""One13s current-depth diagnostic; second view never enters control.

24development environments,15600active+600global settling ticks, no complete
20s episode, optimizer, holdout, or repaired-motion claim. The original staged
controller and frozen parent's absolute timing run unchanged through13s.
"""
import sys,json,hashlib,shutil,traceback,datetime
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');OUT=D/'dual_view_diagnostic_39'
argv=json.loads((D/'staged_approach_36/candidate.command.json').read_text())[3:]
k=argv.index('--landing-config');del argv[k:k+2];argv[argv.index('--output')+1]=str(OUT);sys.argv=[__file__]+argv
import run_low_positive_feedback_v2 as harness
from low_seat_staged_controller_36 import install
install(harness,json.loads((D/'normal_landing3s_guarded_09.json').read_text()))
DOWN=np.eye(4);DOWN[:3,:3]=np.array([[0.,1.,0.],[1.,0.,0.],[0.,0.,-1.]]);DOWN[:3,3]=[-.22,0,.55]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    OUT.mkdir(exist_ok=False);(OUT/'source').mkdir()
    a=harness.parent_harness.arguments();cfg,goals,resets=harness.parent_harness.build(a,OUT)
    assert cfg.scene.rest_camera is None
    cfg.scene.diagnostic_down_camera=harness.base.CameraCfg(prim_path='{ENV_REGEX_NS}/Robot/pelvis/DiagnosticDownCamera',width=128,height=96,data_types=['distance_to_image_plane'],update_period=1.e6,
        offset=harness.base.CameraCfg.OffsetCfg(pos=tuple(DOWN[:3,3]),rot=tuple(Rotation.from_matrix(DOWN[:3,:3]).as_quat()),convention='ros'),
        spawn=harness.base.sim.PinholeCameraCfg(focal_length=18.,horizontal_aperture=24.,clipping_range=(.05,4.)))
    sources=[Path(__file__).resolve()]+[Path(__file__).with_name(n) for n in ['low_seat_staged_controller_36.py','low_seat_staged_planner_36.py','low_seat_normal_duration_36.py','low_seat_load_duration_36.py','low_seat_contact_duration_36.py']]
    protocol=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='SHORT PHYSICAL SENSOR DIAGNOSTIC ONLY. Enable the existing code-defined downward mount under a diagnostic name; record current depth without passing it to tracker, actor, reference or policy. Rear-only staged36 control and frozen59117 timing remain unchanged. Camera addition/exposure may introduce rendering differences, which must be reported, never forced equal.',seconds=13.,environments=24,active_ticks=15600,initial_settling_excluded=600,indexed_warmup=0,full20s_episodes=0,optimizer_updates=0,fresh_holdout=False,downward_mount=DOWN.tolist(),capture_from_s=6.,capture_period_s=.2,sources={str(p):sha(p) for p in sources},parent_sha256=sha(a.checkpoint))
    (OUT/'diagnostic_protocol.json').write_text(json.dumps(protocol,indent=2))
    captures=[];sensor_frames=[];parent_actions=[]
    with harness.base.launch_simulation(cfg,a):
        raw=harness.base.gym.make('Humanoid-Sit-Learned-G1',cfg=cfg).unwrapped
        env=harness.FeedbackEnv(raw,a,OUT,goals,resets);clock=raw.sim.get_physics_step_count()
        parent=harness.PosturePPO(np.zeros(414),np.ones(414),0,env.device);parent.load(a.checkpoint)
        for module in [parent.actor,parent.critic]:
            for w in module.parameters():w.requires_grad_(False)
        parent_target=torch.zeros(env.num_envs,3,device=env.device);handover=None
        try:
            with torch.inference_mode():
                for step in range(130):
                    sensor=env.get_observations()['policy'].detach().clone();assert sensor.shape==(24,414);sensor_frames.append(sensor.cpu().numpy())
                    if step==110:
                        handover=harness.parent_decision_steps(env);pa=parent.choose(sensor,False)[0];due=handover==110;parent_target[due]=.008*torch.tanh(pa[due])
                        parent_actions.append(dict(step=110,observation=sensor.cpu().numpy().tolist(),action=pa.cpu().numpy().tolist(),due=due.cpu().numpy().tolist()))
                    env.step(harness.old.mapped(env,parent_target))
                    (OUT/'step_status.json').write_text(json.dumps(dict(completed_control_steps=step+1,active_transitions=env.active_transitions,physics_substeps=raw.sim.get_physics_step_count()-clock)))
                    if (step+1)%2==0 and step>=59:
                        cam=raw.scene['rear_camera'];data=env.robot.data
                        pos,quat=cam._view.get_world_poses()
                        entry=dict(t=round((step+1)*.1,6),rear_depth=cam.data.output['distance_to_image_plane'].torch.cpu().numpy().copy(),rear_K=cam.data.intrinsic_matrices.torch.cpu().numpy().copy(),rear_position=pos.torch.cpu().numpy().copy(),rear_orientation=quat.torch.cpu().numpy().copy(),q=data.joint_pos.torch.cpu().numpy().copy(),gravity=data.projected_gravity_b.torch.cpu().numpy().copy(),body=data.body_pose_w.torch.cpu().numpy().copy(),geometry=env.geometry.copy(),estimated_root_x=env.estimated_root_x.cpu().numpy().copy() if isinstance(env.estimated_root_x,torch.Tensor) else np.asarray(env.estimated_root_x).copy())
                        current_clock=raw.sim.get_physics_step_count()
                        down=raw.scene['diagnostic_down_camera'];harness.base.acquire_current_camera(down)
                        assert raw.sim.get_physics_step_count()==current_clock
                        pos,quat=down._view.get_world_poses()
                        entry.update(down_depth=down.data.output['distance_to_image_plane'].torch.cpu().numpy().copy(),down_K=down.data.intrinsic_matrices.torch.cpu().numpy().copy(),down_position=pos.torch.cpu().numpy().copy(),down_orientation=quat.torch.cpu().numpy().copy())
                        captures.append(entry)
                        if len(captures)%5==0:np.savez_compressed(OUT/'raw_dual_captures.npz',**{k:np.array([x[k] for x in captures]) for k in captures[0]})
                assert env.active_transitions==15600 and not env.episodes and env.warmup_transitions==0
                assert abs((raw.sim.get_physics_step_count()-clock)*raw.sim.get_physics_dt()-13.)<1e-6
                np.savez_compressed(OUT/'raw_dual_captures.npz',**{k:np.array([x[k] for x in captures]) for k in captures[0]})
                np.savez_compressed(OUT/'actual_sensor_frames.npz',sensors=np.array(sensor_frames))
                (OUT/'parent_actions.json').write_text(json.dumps(parent_actions))
                (OUT/'diagnostic_accounting.json').write_text(json.dumps(dict(elapsed_s=13.,physics_substeps=raw.sim.get_physics_step_count()-clock,physics_dt=raw.sim.get_physics_dt(),active_ticks=15600,initial_settling_excluded=600,indexed_warmup=0,complete20s_episodes=0,optimizer_updates=0,fresh_holdout=False,capture_times=[x['t'] for x in captures],down_depth_control_use=False),indent=2))
                env.close()
        except BaseException:
            (OUT/'execution_failure.txt').write_text(traceback.format_exc());raise
    assert sha(a.checkpoint)==protocol['parent_sha256']
    assert all(sha(p)==h for p,h in protocol['sources'].items())
    shutil.copy2(__file__,OUT/'source'/Path(__file__).name)
    print('DUAL_VIEW_DIAGNOSTIC_COMPLETE',flush=True)
if __name__=='__main__':main()
