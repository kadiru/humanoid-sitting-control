"""Freeze a single bounded pilot. Dataset assembly only; no physics or training."""
import json,hashlib,shutil,ast
from pathlib import Path
import numpy as np
ROOT=Path('/home/kadir/humanoid-sitting'); D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')
OUT=D/'contact_learning_ppo_20261026_v1'
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert not OUT.exists();OUT.mkdir();bank=OUT/'training_bank';bank.mkdir()
all_cases=[];evidence={}
for b in ['known_old','known_transfer','known_recent']:
 p=D/'banks'/b;g=json.loads((p/'geometry.json').read_text())
 rp=D/'current_retention_09'/b/'candidate/results.json';r=json.loads(rp.read_text())['episodes']
 evidence[str(rp)]=sha(rp)
 for i,x in enumerate(g):
  rows=[v for v in r if v['surface_id']==i];assert len(rows)==2
  clean=all(v['success'] and not any(v['failures'].values()) and v['slip_ticks']==0 and v['illegal_ticks']==0 for v in rows)
  all_cases.append((b,i,x,clean))
failed=[x for x in all_cases if not x[3]];assert len(failed)==4
clean=sorted([x for x in all_cases if x[3]],key=lambda x:(x[2]['height'],x[2]['pitch_deg'],x[0],x[1]))
selected=failed+[clean[i] for i in np.linspace(0,len(clean)-1,8,dtype=int)]
geoms=[]
for n,(b,i,g,c) in enumerate(selected):geoms.append(dict(g,case=n,surface_id=n,split='development',pilot_source_bank=b,pilot_source_index=i,baseline_geometry_clean=c))
write(bank/'geometry.json',geoms)
for name in ['candidates.npz','reset_candidates.npz']:
 arrays=[dict(np.load(D/'banks'/b/name)) for b,i,g,c in selected]
 np.savez_compressed(bank/name,**{k:np.stack([a[k][selected[j][1]] for j,a in enumerate(arrays)]) for k in arrays[0]})
write(bank/'selection.json',dict(rule='All four failing development geometries, each in both existing profiles; eight evenly spaced clean geometries sorted by height/slope/bank/index, each in both profiles.',cases=[dict(bank=b,index=i,clean=c) for b,i,g,c in selected],baseline_evidence=evidence,all_72_development=True,fresh_holdout_used=False))
cmd=json.loads((D.parent/'low_positive_transfer_01/ppo_01/seed_62111/development_old_current_61111.command.json').read_text())
write(OUT/'base_command.json',cmd)
protocol=dict(version='contact_learning_ppo_20261026_v1',authorization='Human: Can we start training a ppo now. New bounded contact-learning pilot; supersedes repair-before-training and all-exploration-clean prerequisites only. Does not authorize any old solver cycle or physical deployment.',
 seeds=[63111,63117],num_envs=24,seconds_per_episode=20,physics_dt=.02,updates_per_seed=8,smoke_updates_per_seed=2,
 observations='414 sensor/reference channels plus three previous normalized residuals. No simulator truth enters actor or critic.',
 action='Three symmetric hip/knee/ankle pitch corrections, +/-0.004rad, slew0.004rad/s, chosen every0.5s from6s, commands0.1s. Existing final command bounds, filter and slew remain.',
 parents='Frozen best09: early61111 compact6–7s pulse + resting59117 at measured11/15s + normal_landing3s_guarded_09. No parent weight or gain changes.',
 reward='From8s: dt*(2*min(legal_rest_normal_load_BW/0.2,1)*exp(-(speed/0.1)^2-(gyro/0.25)^2)+2*good_rest), gated by legal/no failure/alive; minus inherited dense loaded-slip and wrong-contact costs. All37 torque-squared integral penalty coefficient0.25/1350, baseline-centered. Inherited terminal penalties unchanged. Undiscounted full reward-to-go, no bootstrap.',
 optimizer='Existing clipped PPO0.2, Adam0.001, fixed Normalstd0.6, max8actor and8critic epochs per full-episode update, KL0.02 rollback. Main starts from matching smoke INITIAL, not trained smoke.',
 phases=['old_best09_zero','calibrate_and_zero_identity','indexed_zero_2episodes','smoke63111_2updates_and_CPU_replay','smoke63117_2updates_and_CPU_replay','train63111_8updates','train63117_8updates','each_final_policy_3development_banks'],
 budget=dict(active_physics_transitions=720000,indexed_warmup_transitions=10200,initial_settling_transitions=7800,total_physics_transitions=738000,main_active_per_seed=192000,main_indexed_warmup_per_seed=4200,wall_seconds=10800,child_wall_seconds=2400,max_output_bytes=8*1024**3,min_free_bytes=6*1024**3,concurrent_scientific_children=1),
 acceptance='No automatic promotion. Both policies evaluated on all72 full20s development trials, preserving all64 baseline clean trials; report all8 unresolved separately, slip and housing contacts, final continuous5s valid rest and all37 effort. Reward improvement or completed optimizer update is not physical repair. Fresh24 holdout closed until selection.',
 stops='Stop on infrastructure/source/identity/bounds/nonfinite/CPUreplay failure, wall or storage cap. Physical exploration failures are logged and do not stop this simulation-only learning pilot. No automatic retry or budget extension.',
 limitations='Sitting only, reference-assisted residual policy, small budget; not whole-body from-scratch RL, not a causal reward ablation, not a leaning policy or hardware deployment.',automation='Deleted by user; do not recreate')
write(OUT/'protocol.json',protocol)
# Bind source, immutable controller inputs, and all three development banks.
paths=list((ROOT/'scripts').rglob('*.py'))+list((ROOT/'sitting_task/src').rglob('*.py'))
paths += [ROOT/'scripts/run_supported_gpu0.sh',D/'normal_landing3s_guarded_09.json',OUT/'protocol.json',OUT/'base_command.json']
for value in cmd:
 p=Path(value)
 if p.is_absolute() and p.is_file():paths.append(p)
for b in ['known_old','known_transfer','known_recent']:
 paths+=list((D/'banks'/b).iterdir())
paths+=list(bank.iterdir())
m=json.loads((D/'current_retention_09/known_old/candidate/manifest.json').read_text())
paths += [Path(p) for p in m.get('sources',{}) if Path(p).is_file()]
for p in paths:
 if p.name in ['contact_reward_v1.py','run_contact_learning_ppo_v1.py','prepare_contact_ppo_v1.py','audit_contact_ppo_v1.py','launch_contact_ppo_v1.py']:ast.parse(p.read_text(),filename=str(p))
write(OUT/'freeze.json',{str(p):sha(p) for p in sorted(set(paths)) if p.is_file()})
print(json.dumps(dict(output=str(OUT),bindings=len(json.loads((OUT/'freeze.json').read_text())),selected=[(b,i,c) for b,i,g,c in selected],protocol=protocol['budget'])))
