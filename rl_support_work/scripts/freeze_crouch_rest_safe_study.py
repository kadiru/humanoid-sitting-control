"""Freeze the admitted small pilot; does not start a process or train a model."""
import json,hashlib,time
from pathlib import Path
from run_crouch_rest_safe_study import ROOT,DATA,STUDY,SEEDS,read,sha

assert read(DATA/'safe_ppo_admission_01.json')['passed']
assert read(DATA/'safe_env_reset_memory_02/reset_probe.json')['passed']
STUDY.mkdir(exist_ok=False)
old=read(DATA.parent/'near_seat_curriculum_01/ppo_01/protocol.json')
paths=set(old['hashes'])
# New experiment entry points and explicitly used controller/model parameters.
names=['run_crouch_rest_safe_study.py','run_crouch_rest_safe_ppo.py','run_crouch_rest_audited_ppo.py','crouch_rest_validated_controller.py','crouch_rest_motion.py','sagittal_static_bias.py','run_crouch_rest_preflight.py','reserve_crouch_rest_holdout.py','freeze_crouch_rest_safe_study.py']
paths.update(str(ROOT/'scripts'/n) for n in names)
paths.add(str(ROOT/'docs/CROUCH_REST_SAFE_PPO_PROTOCOL.md'))
paths.add(str(DATA/'balanced_standing10_01/static_inertial_model.json'))
paths.add(str(DATA/'safe_ppo_admission_01.json'));paths.add(str(DATA/'safe_env_regression_verification_01.json'));paths.add(str(DATA/'safe_env_reset_memory_02/reset_probe.json'));paths.add(str(DATA/'holdout_reservation_01.json'))
paths.update(str(p) for p in (DATA/'reserved_bank_01').iterdir() if p.is_file())
prior=read(DATA/'prior_hashes.json')
# Preserve completed development evidence. Root status/log files remain mutable.
for folder in DATA.iterdir():
    if folder.is_dir() and folder!=STUDY:
        for p in folder.rglob('*'):
            if p.is_file():prior[str(p)]=sha(p)
for p,h in prior.items():assert sha(p)==h,p
baseline=read(DATA.parent/'seat_load_transfer_01/ppo_01/protocol.json')['development_baselines']
protocol=dict(timestamp=time.time(),seeds=SEEDS,block_updates=[16]*4,updates_per_seed=64,actor_enabled_scheduled_updates_per_seed=56,initial_critic_only_scheduled_updates=8,steps_per_update=32,physics_ticks_per_decision=5,num_envs=18,physics_transitions_per_seed=184320,total_physics_transitions=368640,separate_optimizer_smoke_transitions=17280,eligible_geometry_indices=[0,1,2,4,5,8],excluded_geometry_indices=[3,6,7],control_windows_seconds=[15,13,11,10],standing_resets_only=True,pre10seconds_actions_always_zero=True,earlier_control_requires_its_own_full_horizon_validation=True,evaluation_all_conditions=True,selection='full-budget final checkpoint; development gates choose control timing only',development_bank=str(DATA.parent/'seat_load_transfer_01/banks/development'),holdout_bank=str(DATA/'reserved_bank_01'),holdout_geometries=8,holdout_trials=16,development_baselines={k:v for k,v in baseline.items() if k!='new_reference'},bridge_reference=str(DATA/'safe_env_startup_reference_01'),hashes={p:sha(p) for p in sorted(paths)},prior_hashes=prior,caveat='Restricted six-surface training domain; three known failed geometries remain stress tests. All episodes physically start standing but the first10s approach stays reference-controlled. Torque proxy, not energy. Two seeds and repeated rigid surfaces are limited evidence.')
f=STUDY/'protocol.json';f.write_text(json.dumps(protocol,indent=2));(STUDY/'protocol.sha256').write_text(sha(f)+'\n')
print(json.dumps(dict(protocol=str(f),frozen_inputs=len(protocol['hashes']),protected_prior_files=len(prior),main_physics_transitions=368640)))
