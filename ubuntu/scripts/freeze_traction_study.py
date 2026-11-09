"""Freeze only after reset, exact-reference and both optimizer smokes pass."""
import json,time
from pathlib import Path
from run_traction_study import ROOT,DATA,STUDY,SEEDS,read,sha
admission=read(DATA/'preflight_with_smokes_01.json');assert admission['passed'] and len(admission['smokes'])==2
assert not STUDY.exists()
old=read(DATA.parent/'crouch_rest_bridge_01/ppo_safe_surfaces_01/protocol.json');paths=set(old['hashes'])
new_sources=['run_traction_transfer_ppo.py','foot_wrench_observer.py','traction_clearance_planner_v2.py','traction_feedback.py']
names=new_sources+['run_traction_reset_audit.py','run_traction_smoke.py','run_traction_study.py','validate_traction_preflight.py','verify_traction_study.py','freeze_traction_study.py','finalize_traction_study.py','render_traction_replay.py','report_traction_study.py','launch_traction_after_preflight.py']
paths.update(str(ROOT/'scripts'/n) for n in names)
paths.update(str(DATA/f) for f in ['selected_controller.json','preflight_admission_definition.json','preflight_with_smokes_01.json','preflight_environment_01.json','baseline_comparison_validation_01.json','prior_preservation_validation_01.json','holdout_reservation_01.json','env_reset_01/reset_probe.json'])
paths.add(str(ROOT/'docs/TRACTION_TRANSFER_PROTOCOL.md'));paths.update(str(p) for p in (DATA/'reserved_bank_01').iterdir() if p.is_file())
prior=read(DATA/'prior_hashes.json')
for folder in DATA.iterdir():
    if folder.is_dir() and folder!=STUDY:
        for p in folder.rglob('*'):
            if p.is_file():prior[str(p)]=sha(p)
for p,h in prior.items():assert sha(p)==h,p
for folder in [DATA/'env_reference_01']+[DATA/f'{name}_{s}' for s in SEEDS for name in ['smoke','smoke_eval']]:
    for name in new_sources:assert sha(folder/'source'/name)==sha(ROOT/'scripts'/name),(folder,name)
    for name,h in read(folder/'manifest.json')['sources'].items():assert sha(name)==h,(folder,name)
baselines={k:v for k,v in old['development_baselines'].items()};baselines['bridge_reference']=old['bridge_reference'];checkpoints={}
for s in [56211,56217]:checkpoints[f'current_{s}']=str(DATA.parent/f'contact_effort_repair_01/sitting_ppo_01/train_{s}/final.pt')
for s in [58111,58117]:
    f=read(DATA.parent/f'crouch_rest_bridge_01/ppo_safe_surfaces_01/training_finished_{s}.json');baselines[f'bridge_{s}']=f['final_development_evaluation'];checkpoints[f'bridge_{s}']=f['checkpoint']
baselines['new_reference']=str(DATA/'env_reference_01')
jobs={'0':[],'1':[]}
for i,(name,path) in enumerate(baselines.items()):
    kind='new' if name=='new_reference' else 'bridge' if name.startswith('bridge') else 'current_reference' if name=='current_reference' else 'current_policy'
    jobs[str(i%2)].append(dict(name=name,kind=kind,mode='evaluate' if name in checkpoints else 'reference',**(dict(checkpoint=checkpoints[name]) if name in checkpoints else {})))
protocol=dict(timestamp=time.time(),admitted=True,admission=admission,seeds=SEEDS,updates_per_seed=64,critic_only_scheduled_rollouts=8,actor_enabled_scheduled_rollouts=56,steps_per_update=32,physics_ticks_per_decision=5,num_envs=18,physics_transitions_per_seed=184320,total_physics_transitions=368640,separate_smoke_transitions=46080,global_initial_settle_excluded_per_seed=450,all_standing_resets=True,learning_start_seconds=11,controller=read(DATA/'selected_controller.json'),new_sources=new_sources,development_baselines=baselines,baseline_checkpoints=checkpoints,holdout_jobs=jobs,development_bank=str(DATA.parent/'seat_load_transfer_01/banks/development'),holdout_bank=str(DATA/'reserved_bank_01'),selection='Full-budget final checkpoints only; no promotion or holdout tuning',hashes={p:sha(p) for p in sorted(paths)},prior_hashes=prior,caveats=['All starts are standing, but the first11seconds remain reference controlled. This is residual PPO, not unassisted motion discovery.','Two seeds and nearby rigid surfaces repeated twice are limited generalization evidence.','Reference strict success16/18 remains reported: two20ms samples exceed0.90 friction utilization without slipping; physical admission requires18/18stable, zero slip/unwanted contact and utilization<=1.0.','Torque over all37 joints is an effort proxy, not energy or a global optimum.','Reward, physical labels and strict metric remain unchanged from the prior bridge; only new-study admission is separated from conservative transient grip margin.','Candidate search falls back to its bounded0.66rad forward lean if model clearance target cannot be achieved; physics validation remains mandatory.'])
STUDY.mkdir(exist_ok=False)
(STUDY/'protocol.json').write_text(json.dumps(protocol,indent=2));(STUDY/'protocol.sha256').write_text(sha(STUDY/'protocol.json')+'\n');print(dict(frozen=len(paths),protected=len(prior),total_transitions=368640))
