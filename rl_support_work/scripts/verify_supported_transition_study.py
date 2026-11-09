"""Independent final verification; refuses partial studies and writes new output."""
import argparse,hashlib,json,html
from pathlib import Path
import numpy as np
import torch
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--study',required=True);p.add_argument('--output',required=True);a=p.parse_args();root=Path(a.study);out=Path(a.output)
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
protocol=read(root/'protocol.json');checks={};summary={};pairs={};training={}
checks['frozen_hashes']={p:sha(Path(p))==h for p,h in protocol['hashes'].items()};assert all(checks['frozen_hashes'].values())
old=read(Path('/data/humanoid-sitting-perception/resting_bounded_ppo_02/frozen/manifest.json'))
checks['prior_study_hashes']={p:sha(Path(p))==h for p,h in old.items()};assert all(checks['prior_study_hashes'].values())
for seed in protocol['seeds']:
 directory=root/f'train_{seed}';v=read(directory/'verification.json');assert v['updates']==224 and v['transitions']==372736 and v['active']+v['warmup']==372736
 initial=torch.load(directory/'initial.pt',map_location='cpu',weights_only=False);final=torch.load(directory/'final.pt',map_location='cpu',weights_only=False)
 changed=[]
 for k,value in final['actor_state_dict'].items():
  assert torch.isfinite(value).all();delta=float((value-initial['actor_state_dict'][k]).abs().max())
  if delta>0:changed.append(dict(tensor=k,max_delta=delta))
 assert any(x['tensor'].startswith('mlp') for x in changed)
 rows=read(directory/'results.json')['episodes'];start=[r for r in rows if r['reset_index']==0];supported=[r for r in rows if r['reset_index']==5]
 training[str(seed)]=dict(**v,changed=changed,checkpoint_sha256=sha(directory/'final.pt'),curriculum_episodes=len(rows),standing_episodes=len(start),standing_successes=sum(r['success'] for r in start),supported_reset_episodes=len(supported),supported_reset_successes=sum(r['success'] for r in supported))
 for bank,n in [('train',13),('holdout',16)]:
  label=f'eval_{seed}_{bank}';path=root/label;fixed=root/f'fixed_{bank}'
  loaded=torch.load(path/'initial.pt',map_location='cpu',weights_only=False)
  assert all(torch.equal(x,loaded['actor_state_dict'][k]) for k,x in final['actor_state_dict'].items())
  m=read(path/'manifest.json');assert Path(m['args']['checkpoint'])==directory/'final.pt'
  assert read(path/'geometry.json')==read(fixed/'geometry.json')
  s=np.load(path/'startup.npz');f=np.load(fixed/'startup.npz');differences={k:float(np.max(abs(s[k]-f[k]))) for k in s.files}
  # Same simulator/device/seed/batch should reproduce startup exactly. Any
  # mismatch is surfaced for investigation instead of silently raising tolerance.
  assert all(x==0 for x in differences.values()),(label,differences)
  pairs[label]=dict(exact_startup=True,max_errors=differences,final_checkpoint_loaded=True)
  r=read(path/'results.json')['episodes'];assert len(r)==n
  geometry=read(path/'geometry.json');trace=np.load(path/'trace.npz');assert abs(trace['t'][-1]-20)<1e-8
  assert np.max(abs(np.diff(trace['command'],axis=0)))<=.075001 #5 logged control ticks
  for row in r:
   assert row['steps']==1000 and row['duration_s']==20 and row['reset_index']==0
   assert row['success']==(not any(row['failures'].values()) and row['final_hold_s']>=5)
   if row['success']:assert trace['good'][-50:,row['case']].all()
  grouped={}
  for kind in ['sit','lean']:
   subset=[x for x in r if geometry[x['case']]['kind']==kind]
   grouped[kind]=dict(success=sum(x['success'] for x in subset),total=len(subset),failures={k:sum(x['failures'][k] for x in subset) for k in ['fall','forbidden','impact','torque','joint_limit']})
  sensor=read(path/'sensor_audit.json')['truth_validation'];live=[]
  for row in sensor:
   j=max(0,min(len(trace['t'])-1,int(round(row['t']/.1))-1))
   if row['t']==0 or trace['alive'][j,row['case']]:live.append(row)
  good=[x for x in live if x['valid']]
  sensor_summary=dict(before_failure_count=len(live),valid_count=len(good),p95_normal_deg=float(np.percentile([x['normal_error_deg'] for x in good],95)) if good else None,p95_abs_offset_m=float(np.percentile([abs(x['offset_error_m']) for x in good],95)) if good else None)
  summary[label]=dict(success=sum(x['success'] for x in r),total=n,by_kind=grouped,sensor=sensor_summary,rows=[dict(x,geometry=geometry[x['case']]) for x in r])
for bank,n in [('train',13),('holdout',16)]:
 fixed=root/f'fixed_{bank}';r=read(fixed/'results.json')['episodes'];trace=np.load(fixed/'trace.npz');assert len(r)==n and all(x['duration_s']==20 for x in r)
 constant=float(np.max(abs(trace['command']-trace['command'][0])));assert constant<2e-6
 summary[f'fixed_{bank}']=dict(success=sum(x['success'] for x in r),total=n,physical_failures=sum(any(x['failures'].values()) for x in r),constant_command_error=constant)
checks.update(training=training,startup_pairs=pairs,passed=True)
out.mkdir(exist_ok=False);(out/'validation.json').write_text(json.dumps(checks,indent=2));(out/'results.json').write_text(json.dumps(summary,indent=2))
fig,axes=plt.subplots(1,2,figsize=(11,4.8))
for ax,bank,title in zip(axes,['train','holdout'],['Training surfaces','Unseen surfaces']):
 labels=['Fixed stance','PPO54211','PPO54217'];values=[summary[f'fixed_{bank}']['success']]+[summary[f'eval_{s}_{bank}']['success'] for s in protocol['seeds']]
 n=summary[f'fixed_{bank}']['total'];ax.bar(labels,values,color=['#888b96','#297e9d','#cf9847']);ax.set_ylim(0,n+1);ax.set_ylabel('Successful full standing-to-support trials');ax.set_title(title)
 for i,v in enumerate(values):ax.text(i,v+.2,f'{v}/{n}',ha='center')
 ax.spines[['top','right']].set_visible(False)
fig.suptitle('Reached support and stayed for the final5seconds • full20-second evaluation',fontsize=11);fig.tight_layout();fig.savefig(out/'standing_transition_results.png',dpi=180)
rows=''
for label,d in summary.items():
 if 'by_kind' not in d:continue
 rows+=f'<tr><td>{html.escape(label)}</td><td>{d["success"]}/{d["total"]}</td><td>{d["by_kind"]["sit"]["success"]}/{d["by_kind"]["sit"]["total"]}</td><td>{d["by_kind"]["lean"]["success"]}/{d["by_kind"]["lean"]["total"]}</td></tr>'
page='''<!doctype html><meta charset="utf-8"><title>Supported transition PPO comparison</title><style>body{max-width:1050px;margin:40px auto;font:17px/1.5 system-ui;color:#243040}img{width:100%}table{border-collapse:collapse;width:100%}td,th{padding:12px;border-bottom:1px solid #ddd;text-align:left}h1{font-size:30px}a{color:#16769a}</style><h1>Standing to supported sitting and leaning</h1><p>Two224-update PPO seeds. Effort was secondary to reaching and keeping valid support. All scores below begin from standing; training successes beginning already supported are excluded.</p><img src="standing_transition_results.png"><table><tr><th>Policy/bank</th><th>Total</th><th>Sitting</th><th>Leaning</th></tr>'''+rows+'''</table><p><a href="validation.json">Independent verification</a> · <a href="results.json">Per-surface results and physical failures</a></p><p>All frozen source/bank hashes, prior-study hashes, finite changed actor weights, budgets, final checkpoints, full horizons and exact matched startups passed. Inspect detailed physical failures and sensor errors before interpreting the overall score.</p><p>Two seeds reuse the same13 training and16 test surfaces. These are prealigned finite rigid surfaces, not arbitrary furniture or cushions. One changed-outline goal was borderline in endpoint validation. Vertical90° support is not validated. Low seats lack admissible intermediate reset states, limiting curriculum coverage. Normalized torque is an effort proxy, not energy; fixed supported endpoints do not constitute matched trajectory effort baselines.</p>'''
(out/'review.html').write_text(page)
print(json.dumps({k:{'success':v['success'],'total':v['total']} for k,v in summary.items()},indent=2))
