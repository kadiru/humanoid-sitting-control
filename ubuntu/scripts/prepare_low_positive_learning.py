"""Reserve the finite feedback-learning pilot after unsuccessful manual probes."""
import copy,json,hashlib,shutil,time
from pathlib import Path
import numpy as np
from low_positive_common import DATA,OLD,read

bank=DATA/'banks/training';bank.mkdir(exist_ok=False)
old=DATA/'banks/development_old';transfer=DATA/'banks/development_transfer'
geometries=copy.deepcopy(read(old/'geometry.json'));extra=[i for i,g in enumerate(read(transfer/'geometry.json')) if abs(g['height']-.37)<1e-6 and g['pitch_deg']>10]
assert len(geometries)==12 and len(extra)==2
for i in extra:
    g=copy.deepcopy(read(transfer/'geometry.json')[i]);g.update(case=len(geometries),surface_id=len(geometries),split='development_training',source_bank='previous study holdout, now development');geometries.append(g)
(bank/'geometry.json').write_text(json.dumps(geometries,indent=2))
for name in ['candidates.npz','reset_candidates.npz']:
    a=np.load(old/name);b=np.load(transfer/name)
    np.savez_compressed(bank/name,**{k:np.concatenate([a[k],b[k][extra]],axis=0) for k in a.files})
(bank/'selection.json').write_text(json.dumps(dict(source_banks=[str(old),str(transfer)],added_indices=extra,standing_only=True),indent=2))
shutil.copy2(OLD/'selected_reference_v2.json',DATA/'learning_reference.json')
proposal=dict(version='feedback_pilot_v2',timestamp=time.time(),seeds=[62111,62117],blocks=4,updates_per_block=4,total_updates_per_seed=16,
    environments=28,episode_seconds=20,active_ticks_per_episode=1000,indexed_warmup_per_episode=25,
    active_per_seed=448000,indexed_warmup_per_seed=8400,total_per_seed=456400,total_main=912800,initial_settling_excluded_total=5600,
    control_stages_seconds=[10.,8.],decision_period_s=.5,new_residual_rad=.004,slew_rad_s=.004,inputs=417,actions=3,
    reference_kind='original',reference=str(DATA/'learning_reference.json'),parent='59117 frozen conditional11/15s',
    reason='All six manual variants failed to remove slips and are rejected. Learn the transfer directly on14 known geometries, including the10 initially slipping trials and2 housing-contact trials. Do not claim the controller repair is complete.',
    preflight=dict(calibration_episodes=1,exploration_episodes=1,smoke_updates=2,smoke_evaluation_episodes=1,require_zero_global_indexed_exact=True,require_finite_bounded_actions=True,require_CPU_replay=True,policy_regressions_reported_not_hidden=True),
    gate='Preserve every matched reference-parent clean case; no additional physical-failure case; common-clean whole20 effort <=parent+1%,final5<=parent+2%. Prospective8s must independently pass. No all-case clean requirement silently substituted; known failures are explicit learning targets.',
    selection='Among gate-passing development checkpoints: most clean cases, fewer physical failures, then common-clean effort; earliest control start breaks remaining ties. Otherwise explicit unchanged-parent fallback. Both final and selected reported.',
    holdout='Fresh24 conditions stay closed until both main finals and selections freeze; previous holdout is development.',
    promotion='None',frozen=False)
(DATA/'learning_proposal.json').write_text(json.dumps(proposal,indent=2))
files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for v in ['phase_only','phase_deeper','timing_mild','timing_strong','load_delay','rolling_mild'] for p in (DATA/v).rglob('*') if p.is_file()}
(DATA/'rejected_repairs_preservation.json').write_text(json.dumps(files,indent=2))
print('Prepared14 training geometries/28 trials; no main training or holdout; preserved',len(files),'repair files')
