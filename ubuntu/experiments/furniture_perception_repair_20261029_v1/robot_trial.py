"""Full20s furniture trials using frozen policies and unchanged physical gates."""
import argparse,sys,time
from pathlib import Path
import numpy as np
import torch
from paths import ROOT,HERE,ASSETS,CHECK,PRIOR,OUT,read,write,sha,intent

def main():
    p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--checkpoint',required=True)
    from isaaclab.app import add_launcher_args
    add_launcher_args(p);a=p.parse_args()
    case=next(c for c in read(ASSETS/'cases.json') if c['id']==a.case)
    source=PRIOR if case['task']=='sitting' else ROOT/'experiments/full_body_contact_trainer_4090_step_v5'
    sys.path[:0]=[str(source),str(ROOT/'scripts'),str(ASSETS),str(CHECK)]
    import run_resting_posture as base
    import run_supported_postures as support
    import run_supported_transition as transition
    from environment import FullBodyEnv
    from policy import Learner
    from common import OBS
    from scene_adapter import make_cfg
    from contact_adapter import MeshMetrics
    # Local module is loaded by file identity, avoiding the historical same name.
    import importlib.util
    spec=importlib.util.spec_from_file_location('aligned_capture',HERE/'capture.py');capture=importlib.util.module_from_spec(spec);spec.loader.exec_module(capture)
    observer=capture.Observer(intent(case));transition.observed_support=observer
    adapter=MeshMetrics(case,support.metrics,support);support.metrics=adapter
    a.device='cuda:0';a.num_envs=1;a.seed=64701;a.mode='evaluate';a.depth_check=True;a.enable_cameras=True
    out=a.output;out.mkdir(exist_ok=False)
    bankroot=ROOT/'outputs/full_body_contact_imitation_v4/dataset'/case['task']
    metadata=read(bankroot/'metadata.json');metadata['cases']=1;metadata['ids']=[case['id']]
    with np.load(bankroot/'bank.npz') as z:
        bank={k:z[k][:1].copy() for k in ['reset_q','reset_qd','reset_command','reset_root_pose','reset_root_velocity','phase_times','admitted']}
        bank['q_post']=z['q_post'][:,:1].copy();bank['reference']=np.zeros(1,int)
    assert sha(a.checkpoint)==read(OUT/'policy_bindings.json')[str(a.checkpoint)]
    learner=Learner(np.zeros(OBS),np.ones(OBS),a.seed,device='cuda:0');learner.load(a.checkpoint)
    cfg=make_cfg(a,case);cfg.scene.rear_camera.data_types=['rgb','distance_to_image_plane']
    start=time.monotonic()
    with base.launch_simulation(cfg,a):
        raw=base.gym.make('Humanoid-Sit-Learned-G1',cfg=cfg).unwrapped
        env=FullBodyEnv(raw,a,out,bank,metadata)
        write(out/'startup.json',env.startup)
        write(out/'manifest.json',dict(case=case,checkpoint=a.checkpoint,checkpoint_sha256=sha(a.checkpoint),source=str(source),
            contact_bodies=[x[1] for x in support.CONTACTS],task_intent=intent(case),actor_truth_input=False,action_limits_unchanged=True,success_thresholds_unchanged=True,
            scope='Novel furniture, separate from protected72. Existing leaning policy is a separate baseline.'))
        try:
            capture.save(env,observer,out,0)
            with torch.inference_mode():
                for tick in range(1000):
                    assert time.monotonic()-start<3600
                    env.step(learner.choose(env.observation(),False)[0],record=True)
                    if tick+1 in (10,20):capture.save(env,observer,out,1 if tick+1==10 else 2)
                    if tick%100==99:print(dict(tick=tick+1,alive=bool(env.alive[0])),flush=True)
            env.save_records(out/'trajectory.npz')
            write(out/'contact_mapping.json',dict(validated=adapter.validated,frames=adapter.frames,
                thresholds=dict(surface_distance_m=.015,perimeter_inset_m=.01),multiple_normal_policy='unvalidated'))
        finally:env.close()
    # This audit is from the policy's own unchanged evaluator, independent of its
    # online counters. New perception never writes physical measurements.
    from audit import physical
    with np.load(out/'trajectory.npz') as z:d={k:z[k] for k in z.files}
    result=physical(d);warm=bool(env.startup['warm_failure'][0])
    for k in ['failure','legal','no_slip']:np.testing.assert_array_equal(result[k],d[k])
    for k in ['alive','good']:np.testing.assert_array_equal(result[k]&~np.array(warm),d[k])
    row=read(out/'results.json')['episodes'][0]
    assert row['steps']==1000 and row['duration_s']==20 and row['full_standing_start']
    assert row['clean']==bool(result['clean'][0] and not warm)
    write(out/'independent_audit.json',dict(passed=True,complete_trials=1,duration_s=20,raw_clean=row['clean'],
        contact_mapping_validated=adapter.validated,clean=row['clean'] if adapter.validated else None,
        physical_failures=row['physical_failures'],final_hold_s=row['final_hold_s'],slip_ticks=row['slip_ticks'],
        illegal_ticks=row['illegal_ticks'],action_limits_changed=False,success_thresholds_changed=False,trajectory_sha256=sha(out/'trajectory.npz')))

if __name__=='__main__':main()
