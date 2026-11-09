"""Read-only analysis of the completed frozen pilot, with explicit replay labels."""
import argparse,json,html
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from humanoid_sitting.deployment import BODY_JOINT_NAMES
p=argparse.ArgumentParser();p.add_argument('--study',required=True);p.add_argument('--verified',required=True);a=p.parse_args();r=Path(a.study);v=Path(a.verified)
assert json.loads((v/'validation.json').read_text())['passed']
out=v/'diagnosis';out.mkdir(exist_ok=False)
ROOT=Path(__file__).resolve().parents[1];m=json.loads((ROOT/'runs/g1_kinematics_geometry_04/model.json').read_text());ids=[m['joint_names'].index(n) for n in BODY_JOINT_NAMES]
nom=torch.tensor(m['default_q'])[ids];scale=torch.tensor([1.6 if ('hip_pitch' in n or 'knee' in n) else .65 if 'ankle_pitch' in n else 1.5 if ('shoulder' in n or 'elbow' in n) else .5 for n in BODY_JOINT_NAMES])
start=np.load(r/'fixed_train/startup.npz');obs=torch.tensor(start['observation']);previous=torch.tensor(start['command'])[:,ids]
report={'seeds':{},'replay_scope':'CPU forward pass on the same13 recorded standing observations. Requested target changes before physical joint/rate clamps, not simulated trajectories. Observed command drift is separately measured from saved rollouts.'}
for seed in [54211,54217]:
 directory=r/f'train_{seed}';meta=json.loads((directory/'verification.json').read_text());rows=json.loads((directory/'results.json').read_text())['episodes'];stages={}
 for stage in range(6):
  ss=[x for x in rows if x['reset_index']==stage]
  stages[str(stage)]=dict(count=len(ss),success=sum(x['success'] for x in ss),mean_active_duration_s=float(np.mean([x['duration_s'] for x in ss])) if ss else None,maximum_qualifying_hold_s=max([x['peak_hold_s'] for x in ss],default=0))
 replay=[]
 files=[directory/'initial.pt']+sorted(directory.glob('model_*.pt'),key=lambda x:int(x.stem.split('_')[-1]))+[directory/'final.pt']
 for file in files:
  checkpoint=torch.load(file,map_location='cpu',weights_only=False);c=checkpoint['actor_state_dict'];x=obs.clone()
  for l in [0,2]:x=F.elu(F.linear(x,c[f'mlp.{l}.weight'],c[f'mlp.{l}.bias']))
  action=F.linear(x,c['mlp.4.weight'],c['mlp.4.bias']);delta=nom+scale*action.clamp(-2,2)-previous
  # Rate-bound diagnostic reproduces physical position clamp before lowpass.
  lim=torch.tensor([next(j['limits'] for j in m['joints'] if j['name']==n) for n in BODY_JOINT_NAMES])*np.pi/180
  desired=(nom+scale*action.clamp(-2,2)).clamp(min=lim[:,0]+.03,max=lim[:,1]-.03)
  rate_limited=(.2*(desired-previous)).abs()>.015
  replay.append(dict(checkpoint=file.name,updates=0 if file.name=='initial.pt' else 224 if file.name=='final.pt' else int(file.stem.split('_')[-1])+1,maximum_requested_change_deg=float(torch.rad2deg(delta.abs().max())),rate_limited_joint_count_per_case=rate_limited.sum(1).tolist(),mean_requested_change_deg={n:float(torch.rad2deg(delta[:,j].mean())) for j,n in enumerate(BODY_JOINT_NAMES)}))
 failures=[];cases=[]
 for bank in ['train','holdout']:
  path=r/f'eval_{seed}_{bank}';geometry=json.loads((path/'geometry.json').read_text());results=json.loads((path/'results.json').read_text())['episodes'];first=json.loads((path/'first_failures.json').read_text());mapping={x['case']:x for x in first};trace=np.load(path/'trace.npz');s=np.load(path/'startup.npz')
  for row in results:
   case=row['case'];g=geometry[case];f=mapping.get(case);first_s=f['t']+.02 if f else None
   failures.append(first_s)
   cases.append(dict(seed=seed,bank=bank,case=case,kind=g['kind'],slope_deg=g['pitch_deg'],surface_center_height_m=g['height'],shape=g.get('shape','rect'),success=row['success'],first_failure_s=first_s,first_failure_flags=f['flags'] if f else {},all_failures=row['failures'],depth_availability=row['depth_availability'],final_hold_s=row['final_hold_s'],max_command_shift_before_failure_deg=float(np.rad2deg(abs(trace['command'][trace['t']<first_s,case]-s['command'][case])).max()) if first_s and np.any(trace['t']<first_s) else 0))
 report['seeds'][str(seed)]=dict(budget=meta,warmup_fraction=meta['warmup']/meta['transitions'],reset_stages=stages,replay=replay,cases=cases,first_failure_range_s=[min(x for x in failures if x is not None),max(x for x in failures if x is not None)])
(out/'diagnosis.json').write_text(json.dumps(report,indent=2))
fig,axes=plt.subplots(1,3,figsize=(15,4.5))
colors=['#267b91','#d18836']
for seed,color in zip([54211,54217],colors):
 d=report['seeds'][str(seed)];rep=d['replay'];axes[0].plot([x['updates'] for x in rep],[x['maximum_requested_change_deg'] for x in rep],label=f'Seed{seed}',color=color)
 path=r/f'eval_{seed}_train';trace=np.load(path/'trace.npz');s=np.load(path/'startup.npz');body=json.loads((path/'results.json').read_text())['body_names'].index('pelvis')
 pitch=Rotation.from_quat(trace['body'][:,0,body,3:]).as_euler('xyz',degrees=True)[:,1]
 axes[1].plot(trace['t'],pitch,label=f'Seed{seed}',color=color)
 shift=np.rad2deg(abs(trace['command'][:,0]-s['command'][0])).max(1);axes[2].plot(trace['t'],shift,label=f'Seed{seed}',color=color)
trace=np.load(r/'fixed_train/trace.npz');body=json.loads((r/'fixed_train/results.json').read_text())['body_names'].index('pelvis');pitch=Rotation.from_quat(trace['body'][:,0,body,3:]).as_euler('xyz',degrees=True)[:,1];axes[1].plot(trace['t'],pitch,color='#718079',ls='--',label='Fixed stance');axes[2].axhline(0,color='#718079',ls='--')
axes[0].set(xlabel='Completed PPO updates',ylabel='Largest requested target change (degrees)',title='Same standing observations, changing policy')
axes[1].set(xlabel='Seconds after standing start',ylabel='Pelvis pitch (degrees)',title='Recorded25cm-seat trial',xlim=(0,1.2));axes[1].axhline(55,color='#b34545',ls=':',label='Fall tilt threshold')
axes[2].set(xlabel='Seconds after standing start',ylabel='Largest actual command shift (degrees)',title='Recorded motor targets',xlim=(0,1.2))
for ax in axes:ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.15);ax.legend(fontsize=8)
fig.tight_layout();fig.savefig(v/'control_failure_diagnosis.png',dpi=180)
summary=json.loads((v/'results.json').read_text());fig,axes=plt.subplots(1,2,figsize=(10,4.5))
for ax,bank in zip(axes,['train','holdout']):
 n=summary[f'fixed_{bank}']['total'];safe=[n-summary[f'fixed_{bank}']['physical_failures']]+[sum(not any(x['failures'].values()) for x in summary[f'eval_{s}_{bank}']['rows']) for s in [54211,54217]]
 ax.bar(['Fixed stance','PPO54211','PPO54217'],safe,color=['#718079',*colors]);ax.set_ylim(0,n+1);ax.set_ylabel('Trials with no physical failure');ax.set_title('Training surfaces' if bank=='train' else 'Unseen surfaces')
 for i,x in enumerate(safe):ax.text(i,x+.2,f'{x}/{n}',ha='center')
 ax.spines[['top','right']].set_visible(False)
fig.suptitle('Fixed commands retained standing; learned policies did not',fontsize=13);fig.tight_layout();fig.savefig(v/'physical_stability_comparison.png',dpi=180)
# Preserve the verifier's initial report and build a more informative delivery page.
(v/'verification_review.html').write_text((v/'review.html').read_text())
bodyrows=''
for d in report['seeds'].values():
 for row in d['cases']:
  flag=', '.join(k for k,val in row['all_failures'].items() if val)
  bodyrows+=f'<tr><td>{row["seed"]}</td><td>{row["bank"]}</td><td>{row["case"]}</td><td>{row["kind"]}</td><td>{100*row["surface_center_height_m"]:.1f}</td><td>{row["slope_deg"]:g}</td><td>{row["shape"]}</td><td>{row["first_failure_s"]:.2f}</td><td>{html.escape(flag)}</td></tr>'
budgetrows=''.join(f'<tr><td>{s}</td><td>{d["budget"]["active"]:,}</td><td>{d["budget"]["warmup"]:,}</td><td>{100*d["warmup_fraction"]:.1f}%</td><td>{d["reset_stages"]["0"]["count"]}</td><td>{d["reset_stages"]["0"]["mean_active_duration_s"]:.2f}s</td></tr>' for s,d in report['seeds'].items())
htmltext='''<!doctype html><meta charset="utf-8"><title>Standing-to-support PPO: verified comparison</title><style>body{max-width:1150px;margin:36px auto;font:17px/1.5 system-ui;color:#253443;padding:0 20px}h1{font-size:34px;line-height:1.15}h2{margin-top:34px;font-size:23px}img{width:100%}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:8px;border-bottom:1px solid #dde1e5;text-align:left}.result{padding:20px;background:#fff0eb;border-left:5px solid #bb4a2b}a{color:#1e7398}small{color:#596979}</style><h1>The robot did not learn to reach sitting or leaning support</h1><p class="result">Both PPO seeds scored<strong>0/13 training surfaces and0/16 unseen surfaces</strong>. Every learned-policy evaluation had a physical failure. Fixed commands retained standing in29/29 conditions, but did not attempt a sitting transition. No policy is promoted.</p><img src="physical_stability_comparison.png"><h2>What failed</h2><p>The initialized policy reproduced the stable command. After training, replaying the same standing observations produced immediate target changes of about31° and38°, affecting17–18 rate-limited joints at once. Recorded motor targets then moved by roughly30° within0.7seconds, and the robot lost balance before reaching support. These numbers distinguish requested targets from the rate-limited commands actually sent.</p><img src="control_failure_diagnosis.png"><p>Training never established sustained support, even when episodes began already supported. Low-seat curricula also lacked valid intermediate starts. Approximately half the collected transitions were reset settling ticks. These observations point to failed movement learning and loss of stance retention; they do not establish a unique causal explanation. Raising the effort reward is not supported by this result.</p><h2>Budget and reset accounting</h2><p>Each seed completed224 updates and372,736 transitions. All weights were finite and changed. The initial0.5-second global settling and1664-transition development smoke test are separate from these budgets. No training episode met full20-second success.</p><table><tr><th>Seed</th><th>Active transitions</th><th>Warmup transitions</th><th>Warmup share</th><th>Standing-start episodes</th><th>Mean active duration</th></tr>'''+budgetrows+'''</table><h2>Per-surface physical results</h2><p>All listed rollouts still ran the full20seconds after failure, with failure flags latched. First-failure times below correct the stored pre-step timestamp by one20ms control tick. Height is the support-center height; for tilted panels it is not a seat-top height across the full surface.</p><table><tr><th>Seed</th><th>Bank</th><th>Case</th><th>Task</th><th>Height cm</th><th>Slope°</th><th>Outline</th><th>First failure s</th><th>Latched physical failures</th></tr>'''+bodyrows+'''</table><h2>What remains validated</h2><p>Supported endpoints remain useful:13/13 training goals pass20-second holds; separate75° back-supported poses are valid. Standing starts pass29/29. One unseen changed-outline endpoint was borderline, so it remains in the reported test instead of being removed. Vertical90° support is not validated.</p><h2>Interpretation and next step</h2><p>This pilot provides no learned generalization or effort-saving evidence. There are no successful learned trials on which to make an effort comparison. Torque is a proxy, not measured energy. The next implementation should first prove that the learning interface retains stance and supported poses under exploration, then train short physically feasible motions connecting those states. That is a proposed follow-up; no additional training was launched.</p><p>Two seeds reuse a finite bank of prealigned rigid surfaces. Arbitrary furniture, compliant cushions, approach/alignment and hardware performance remain untested.</p><p><a href="validation.json">Independent verification</a> · <a href="results.json">Results and before-failure sensor accuracy</a> · <a href="diagnosis/diagnosis.json">Checkpoint replay and failure diagnosis</a> · <a href="verification_review.html">Original verification report</a></p>'''
(v/'review.html').write_text(htmltext)
print(json.dumps({s:dict(failure_range=d['first_failure_range_s'],warmup_fraction=d['warmup_fraction']) for s,d in report['seeds'].items()},indent=2))
