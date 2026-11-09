"""Freeze the human-authorized reward comparison, data only; no simulation."""
import json,hashlib,shutil,ast,os,time
from pathlib import Path
ROOT=Path('/home/kadir/humanoid-sitting');D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')
OLD=D/'contact_learning_ppo_20261026_v2';OUT=D/'contact_reward_comparison_128_v3'
def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not OUT.exists();OUT.mkdir()
shutil.copytree(OLD/'training_bank',OUT/'training_bank')
shutil.copy2(OLD/'base_command.json',OUT/'base_command.json')
audit=read(D/'contact_reward_saved_audit_v3.json');assert audit['passed'] and audit['rows']==528
shutil.copy2(D/'contact_reward_saved_audit_v3.json',OUT/'saved_reward_audit.json')
authorization='Fix the rewards and validate them against the saved trajectories, then run the 128-update PPO comparison with both seeds. Keep the current action limits and check all 72 cases for repairs and regressions.'
protocol=dict(version='contact_reward_comparison_128_v3',authorization=authorization,
 seeds=[63111,63117],arms=['original','revised'],updates_per_arm_seed=128,milestones=[16,32,64,128],chunks=[16,16,32,64],
 order=[['revised',63111],['original',63111],['original',63117],['revised',63117]],
 initialization='Identical per-seed original pilot INITIAL checkpoint, optimizer state empty, zero final actor/critic layer, original fixed normalization. Not continued from the failed 8-update learned policies.',
 continuation='Each arm/seed uses identical fixed process boundaries at16/32/64. Resume actor, critic, Adam and CPU/CUDA/NumPy RNG state. A fresh physical reset at each chunk boundary is part of both arms. Other episodes retain the existing indexed reset. No claim of an uninterrupted process.',
 observations=read(OLD/'protocol.json')['observations'],action=read(OLD/'protocol.json')['action'],parents=read(OLD/'protocol.json')['parents'],
 optimizer='Unchanged: clip0.2, Adam0.001, Normalstd0.6, max8actor and8critic epochs, KL0.02 rollback. Complete undiscounted returns; no bootstrap/GAE. Reward normalization is part of the revised package.',
 original_reward='contact_reward_v1.py unchanged, including original terminal costs and baseline-centered effort.',
 revised_reward={'equations_source':'contact_reward_v3.py','clean_completion_raw':50,'scaling_divisor':50,
 'clean_definition':'Existing success (continuous final5s valid supported rest), no physical flags, no slip ticks, no illegal patch ticks, all1000ticks retained.',
 'events':'First slip -5; first illegal patch -5; first physical failure -20, at occurrence rather than repeated as a terminal cost to later decisions.',
 'terminal':'Clean +50; unstable -10; additional traction-margin ticks beyond matching fixed baseline -2.',
 'bounded_shaping':'From6s: all37 effort, loaded tangent velocity, wrong force, illegal patch and six leg-housing collision-vertex clearance. From8s: legal stationary support (saturates0.2BW) and good rest. From6–13s: pelvis collision-vertex distance to inset finite seat patch. Coefficients frozen in source; geometry is a shaping proxy, not a physical acceptance certificate.',
 'proof':'Full scaled clean return >=0.77625; any nonclean return <=0.14875. Physical classification never relaxed.',
 'source_of_geometry':'Saved static robot collision meshes and actual body poses; reward labels only, no actor/critic truth inputs.'},
 validation=dict(saved_episodes=528,source=str(OUT/'saved_reward_audit.json'),identity='New zero-residual source must match old calibrate two-episode fresh/indexed sequence exactly, including all trace, startup, commands and sensor arrays.',
     checkpoints='Every update saves checkpoint and batch; actual actor/critic change and Adam counters checked after first update of each chunk. Independent CPU replay anchored to each preceding GPU checkpoint plus independent reward and geometry checks after every chunk.',
     evaluations='Every milestone, both arms/seeds: all3x24 development cases, full20s1000ticks, deterministic mean policy. Report per-case repaired failures, lost protected cases, stable/slip/illegal/physical, effort and hold duration. No adaptive tuning or selection of a lucky seed.'),
 acceptance='A full repair requires8/8 original failures clean and64/64 baseline clean cases retained under unchanged physical criteria. Report each arm/seed/milestone independently. No automatic promotion. Fresh24 holdout CLOSED. Sitting experiment only; no leaning training or deployment.',
 budget=dict(training_updates=512,training_episodes=12288,evaluation_episodes=1152,identity_episodes=48,
     active_physics_transitions=13488000,indexed_warmup_transitions=298200,initial_settling_transitions=39000,total_physics_transitions=13825200,
     training_policy_decisions=344064,training_actor_epoch_attempts_max=4096,training_critic_epochs=4096,cpu_replay_updates=512,
     wall_seconds=52*3600,training_child_seconds_per_update=420,training_child_overhead_seconds=900,evaluation_child_seconds=900,identity_child_seconds=1500,audit_child_seconds=1800,
     max_output_bytes=12*1024**3,min_free_bytes=5*1024**3,max_child_rss_bytes=48*1024**3,min_system_available_bytes=2*1024**3,
     cpu_threads_per_process=8,gpu_index=0,max_gpu_reserved_bytes=23*1024**3,concurrent_scientific_children=1,
     expected_wall_hours='Approximately40–46 from previous measured runtime;52h hard cap includes audits and setup; no automatic extension.',
     expected_storage_gib='Approximately10–11 including full traces, every checkpoint and batch;12GiB cap; preserve prior evidence.'),
 stops='Stop on source/input hash mismatch, reset identity, nonfinite, action-bound, independent reward/replay failure, worker conflict or resource cap. No blind retries. Exploration failures are outcomes, not grounds for stopping the learning comparison.',
 automation='Deleted by user. This is one finite explicitly requested experiment queue, not a recurring task.',
 prior_failures='All earlier solver-cycle closures and rejected vectors remain terminal; none are inputs or retried here.',
 preparation_corrections=['Initial reward audit stopped because ankle-pitch has no mesh; use the six existing leg-housing meshes.','Independent quaternion oracle initially normalized float32; corrected to float64, retaining1e-10 geometry tolerance. No simulation occurred.'])
write(OUT/'protocol.json',protocol)
write(OUT/'authorization.json',dict(human_instruction=authorization,utc_recorded=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),scope='4 matched128-update PPO runs, saved-data reward validation and full72 evaluations; current action/physical limits retained'))
for seed in protocol['seeds']:
 import torch
 cp=torch.load(OLD/f'train_{seed}/initial.pt',map_location='cpu',weights_only=False)
 assert cp['updates']==0 and not cp['actor_optimizer']['state'] and not cp['critic_optimizer']['state']
for p in (ROOT/'scripts').glob('*v3.py'):ast.parse(p.read_text(),filename=str(p))
# Rebind existing source/input manifest and all new code, calibration/initials,
# physical baseline, saved audit, full geometry model and this exact protocol.
paths=[Path(p) for p in read(OLD/'freeze.json')]
paths+=list((ROOT/'scripts').glob('*v3.py'))
paths+=[ROOT/'runs/g1_kinematics_geometry_04/model.json',OLD/'calibrate/feedback_calibration.json',OUT/'protocol.json',OUT/'authorization.json',OUT/'base_command.json',OUT/'saved_reward_audit.json']
for seed in protocol['seeds']:paths.append(OLD/f'train_{seed}/initial.pt')
for b in ['known_old','known_transfer','known_recent']:paths.append(D/'current_retention_09'/b/'candidate/results.json')
paths+=list((OUT/'training_bank').iterdir())
paths += [Path(p) for p in audit['bindings']]
write(OUT/'freeze.json',{str(p):sha(p) for p in sorted(set(paths)) if p.is_file()})
free=shutil.disk_usage(OUT).free;assert free>protocol['budget']['max_output_bytes']+protocol['budget']['min_free_bytes'],free
write(OUT/'preparation.json',dict(passed=True,source_bindings=len(read(OUT/'freeze.json')),free_bytes=free,saved_validation_passed=True,science_launched=False))
print(json.dumps(dict(prepared=str(OUT),source_bindings=len(read(OUT/'freeze.json')),free_bytes=free)),flush=True)
