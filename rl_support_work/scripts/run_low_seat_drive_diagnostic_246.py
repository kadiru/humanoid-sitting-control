"""One recording-only13s prefix of234; unchanged controller and physics.

Copy backend into PRIVATE destination descriptors at5ms joint states from8 through13s after scene.update.
Assert public cache bytes/timestamps untouched on every read; no state-property refresh.
Also records current link velocities/transforms, backend position/velocity targets, precise implicit-kernel inputs/outputs and live stage settings. No recorder value enters control.15600 active20ms env-ticks +600 initial
settling;0 full20s episodes, optimizer or holdout. Exact prefix replay is audited.
"""
from pathlib import Path
import datetime,hashlib,json,shutil,sys,traceback
import numpy as np
import torch
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');R=Path('/home/kadir/humanoid-sitting');O=D/'drive_diagnostic_246'
read=lambda p:json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
argv=read(D/'centered_probe_234/candidate.command.json')[3:]
config={}
for key in ['--landing-config','--fixed-config']:
    i=argv.index(key);config[key]=read(argv[i+1]);del argv[i:i+2]
argv[argv.index('--output')+1]=str(O);sys.argv=[__file__]+argv
import run_low_positive_feedback_v2 as harness
from low_seat_bias_runtime_v2_208 import install
from low_seat_drive_recorder_244 import DriveRecorder
install(harness,config['--landing-config'],config['--fixed-config'])

def main():
    O.mkdir(exist_ok=False);(O/'source').mkdir()
    a=harness.parent_harness.arguments();cfg,goals,resets=harness.parent_harness.build(a,O)
    frozen=read(D/'drive_revision_246.json');assert all(sha(p)==h for p,h in frozen['sources'].items())
    sources=dict(frozen['sources']);sources[str(Path(__file__).resolve())]=sha(__file__)
    protocol=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope=__doc__,sources=sources,parent_sha256=sha(a.checkpoint),seconds=13.,environments=24,active_ticks=15600,initial_settling_excluded=600,indexed_warmup=0,full20s_episodes=0,optimizer_updates=0,fresh_holdout=False,recording_control_use=False)
    (O/'diagnostic_protocol.json').write_text(json.dumps(protocol,indent=2))
    captures=[];frames=[];parent_actions=[]
    with harness.base.launch_simulation(cfg,a):
        raw=harness.base.gym.make('Humanoid-Sit-Learned-G1',cfg=cfg).unwrapped
        assert not raw._physics_handles_decimation and raw.cfg.decimation==4 and raw.physics_dt==.005
        env=harness.FeedbackEnv(raw,a,O,goals,resets);clock=raw.sim.get_physics_step_count()
        data=env.robot.data
        metadata=dict(joint_names=env.robot.joint_names,body_names=env.robot.body_names,physics_dt=raw.physics_dt,decimation=raw.cfg.decimation,
            actuator_types={k:type(v).__name__ for k,v in env.robot.actuators.items()},
            stiffness=data.joint_stiffness.torch.cpu().numpy().tolist(),damping=data.joint_damping.torch.cpu().numpy().tolist(),
            armature=data.joint_armature.torch.cpu().numpy().tolist(),effort_limits=data.joint_effort_limits.torch.cpu().numpy().tolist(),
            effort_semantics='Implicit actuator applied_torque is approximate PD effort telemetry, not an independent solver joint-effort measurement. Original reward/criteria remain unchanged.')
        (O/'actuator_metadata.json').write_text(json.dumps(metadata,indent=2))
        parent=harness.PosturePPO(np.zeros(414),np.ones(414),0,env.device);parent.load(a.checkpoint)
        for module in [parent.actor,parent.critic]:
            for w in module.parameters():w.requires_grad_(False)
        parent.save(O/'loaded_parent.pt');parent_target=torch.zeros(24,3,device=env.device)
        scene=raw.scene;original_update=scene.update;callbacks=[]
        recorder=DriveRecorder(data,env.robot.actuators)
        # Authored live-stage values, not an assertion about internal island
        # iteration counts. No set/create/write calls occur.
        import omni.usd
        stage=omni.usd.get_context().get_stage();settings=[]
        for prim in stage.Traverse():
            path=str(prim.GetPath())
            if str(prim.GetTypeName())=='PhysicsScene' or path.startswith('/World/envs/env_0/Robot'):
                attrs={x.GetName():str(x.Get()) for x in prim.GetAttributes() if any(key in x.GetName().lower() for key in ['solver','iteration','stabilization'])}
                if attrs:settings.append(dict(path=path,type=str(prim.GetTypeName()),attributes=attrs))
        (O/'live_stage_settings.json').write_text(json.dumps(dict(settings=settings,interpretation='Read-only instantiated USD attributes. Effective solver-island iterations are not independently queried.'),indent=2))
        executor=recorder.executor;original_launch=executor.launch
        def recorded_launch(collection):
            sub=raw.sim.get_physics_step_count()-clock;time_s=sub*raw.physics_dt
            if 8.-1e-9<=time_s<13.-1e-9:return recorder.launch(original_launch,collection,sub,time_s)
            return original_launch(collection)
        executor.launch=recorded_launch
        def recording_update(*args,**kw):
            result=original_update(*args,**kw)
            current=raw.sim.get_physics_step_count();elapsed=(current-clock)*raw.physics_dt
            callbacks.append(current)
            if 8.-1e-9<=elapsed<=13.+1e-9:
                recorder.capture(current-clock,elapsed,env.previous)
            return result
        scene.update=recording_update
        try:
            with torch.inference_mode():
                for step in range(130):
                    sensor=env.get_observations()['policy'].detach().clone();assert sensor.shape==(24,414);frames.append(sensor.cpu().numpy())
                    if step==110:
                        handover=harness.parent_decision_steps(env);pa=parent.choose(sensor,False)[0];due=handover==110;parent_target[due]=.008*torch.tanh(pa[due])
                        parent_actions.append(dict(step=step,observation=sensor.cpu().numpy().tolist(),action=pa.cpu().numpy().tolist(),due=due.cpu().numpy().tolist()))
                    env.step(harness.old.mapped(env,parent_target))
                    (O/'step_status.json').write_text(json.dumps(dict(completed_control_steps=step+1,active_transitions=env.active_transitions,physics_substeps=raw.sim.get_physics_step_count()-clock)))
                assert env.active_transitions==15600 and not env.episodes and env.warmup_transitions==0
                assert len(callbacks)==2600 and np.array_equal(np.array(callbacks)-clock,np.arange(1,2601))
                captures=recorder.records
                (O/'recorder_metadata.json').write_text(json.dumps(recorder.metadata(),indent=2))
                assert len(captures)==1001 and captures[0]['substep']==1600 and captures[-1]['substep']==2600
                np.savez_compressed(O/'substep_states.npz',**{k:np.stack([x[k] for x in captures]) for k in captures[0]})
                np.savez_compressed(O/'implicit_launches.npz',**{k:np.stack([x[k] for x in recorder.launches]) for k in recorder.launches[0]})
                np.savez_compressed(O/'actual_sensor_frames.npz',sensors=np.array(frames))
                (O/'parent_actions.json').write_text(json.dumps(parent_actions))
                (O/'diagnostic_accounting.json').write_text(json.dumps(dict(elapsed_s=13.,physics_substeps=2600,physics_dt=.005,active_ticks=15600,initial_settling_excluded=600,indexed_warmup=0,complete20s_episodes=0,optimizer_updates=0,fresh_holdout=False,recording_control_use=False,substep_records=1001,scene_update_callbacks=2600),indent=2))
                scene.update=original_update;executor.launch=original_launch
                assert all(sha(p)==h for p,h in sources.items())
                (O/'completion_before_close.json').write_text(json.dumps(dict(completed=True,source_hashes_unchanged=True,active_ticks=15600,initial_settling=600,full20s_episodes=0,optimizer_updates=0)))
                env.close()
        except BaseException:
            captures=recorder.records
            (O/'execution_failure.txt').write_text(traceback.format_exc())
            if captures:np.savez_compressed(O/'partial_substep_states.npz',**{k:np.stack([x[k] for x in captures]) for k in captures[0]})
            raise
        finally:scene.update=original_update;executor.launch=original_launch
    assert all(sha(p)==h for p,h in sources.items()) and sha(a.checkpoint)==protocol['parent_sha256']
    shutil.copy2(__file__,O/'source'/Path(__file__).name)
    (O/'completion.json').write_text(json.dumps(dict(completed=True,source_hashes_unchanged=True,full20s_episodes=0,optimizer_updates=0)))
    print('PASSIVE_DIAGNOSTIC_COMPLETE',flush=True)
if __name__=='__main__':main()
