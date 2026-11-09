"""Independent, read-only validation of split-surface controls and PPO results."""
import argparse,hashlib,json,shutil
from pathlib import Path
import numpy as np
import torch

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False))
def load(p):return torch.load(p,map_location='cpu',weights_only=False)
def equal(a,b):
 if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
 return a==b

def weights(initial,final):
 changes={}
 for role in ['actor_state_dict','critic_state_dict']:
  assert all(torch.isfinite(v).all() for v in final[role].values())
  changes[role]=[k for k,v in final[role].items() if not torch.equal(v,initial[role][k])]
  assert any(k.startswith('mlp') for k in changes[role])
 return changes

def check_manifest(folder):
 m=read(folder/'manifest.json');assert (m['observations'],m['actions'])==(405,23)
 for i,(p,h) in enumerate(m['sources'].items()):
  assert sha(p)==h,p
  assert sha(folder/'source'/f'{i:02d}_{Path(p).name}')==h
 for name,h in m['bank_hashes'].items():assert sha(Path(m['args']['bank'])/name)==h
 return m

def starts(a,b):
 x=np.load(a/'startup.npz');y=np.load(b/'startup.npz')
 errors={k:float(abs(x[k]-y[k]).max()) for k in ['root','q','command','geometry']}
 # The appended reference may differ by controller; the sensor history must match.
 errors['sensor_history']=float(abs(x['observation'].reshape(-1,3,135)[:,:,:111]-y['observation'].reshape(-1,3,135)[:,:,:111]).max())
 assert max(errors.values())==0,errors
 assert [(r['surface_id'],r['start_stage'],r['top_center'],r['normal']) for r in read(a/'geometry.json')]==[(r['surface_id'],r['start_stage'],r['top_center'],r['normal']) for r in read(b/'geometry.json')]
 return errors

def evaluation(folder,count,checkpoint=None):
 m=check_manifest(folder);r=read(folder/'results.json');rows=r['episodes'];assert len(rows)==r['count']==count
 assert r['successes']==sum(x['success'] for x in rows)
 for e in rows:
  assert (e['steps'],e['duration_s'],e['late_ticks'],e['last5_ticks'])==(1000,20,500,250)
  assert e['success']==(not any(e['failures'].values()) and e['final_hold_s']>=5)
  for window in ['metrics_whole_trial','metrics_last10s','metrics_last5s']:
   v=e[window];assert all(np.isfinite(x) for x in v.values())
   assert abs(v['total_squared_torque_Nm2']-v['leg_squared_torque_Nm2']-v['upper_squared_torque_Nm2'])<max(.015,v['total_squared_torque_Nm2']*1e-5)
 if checkpoint:
  assert m['args']['checkpoint']==str(checkpoint)
  expected=load(checkpoint);actual=load(folder/'initial.pt')
  assert all(equal(expected[k],actual[k]) for k in ['actor_state_dict','critic_state_dict','optimizer_state_dict','iter'])
 t=np.load(folder/'trace.npz');start=np.load(folder/'startup.npz');assert t['command'].shape==(200,count,37)
 assert all(np.isfinite(t[k]).all() for k in t.files)
 assert np.allclose(t['t'],np.arange(1,201)/10,rtol=0,atol=1e-12)
 names=r['joint_names'];ids=[i for i,n in enumerate(names) if not any('_'+x+'_' in n for x in ['zero','one','two','three','four','five','six'])];other=[i for i in range(37) if i not in ids];assert len(ids)==23
 assert np.array_equal(t['command'][:,:,other],np.broadcast_to(start['command'][None,:,other],t['command'][:,:,other].shape))
 change=float(abs(np.diff(np.concatenate([start['command'][None],t['command']]),axis=0)).max());assert change<=.010002,change
 fixed=m['args']['mode']=='fixed'
 if fixed:assert np.array_equal(t['command'],np.broadcast_to(start['command'][None],t['command'].shape))
 bank=np.load(Path(m['args']['bank'])/'reset_candidates.npz')['command'];geoms=read(folder/'geometry.json');span=np.array(m['spans']);maxerr=0.;referr=0.
 for e in rows:
  c=e['case'];surf=e['surface_id'];p=geoms[e['geometry_index']]['control_profile']
  paths=bank[surf][:,ids];lo=paths.min(0)-span;hi=paths.max(0)+span
  for j,idx in enumerate(ids):
   n=names[idx];k=p.get('knee',0.)
   offset=k if 'knee' in n else -.5*k if 'hip_pitch' in n or 'ankle_pitch' in n else 0
   lo[j]+=min(0,offset);hi[j]+=max(0,offset)
   if 'ankle_pitch' in n and p.get('kp',0):lo[j]-=p.get('cap',.15);hi[j]+=p.get('cap',.15)
  cmd=t['command'][:,c][:,ids];err=max(float((lo-cmd).max()),float((cmd-hi).max()),0);assert err<3e-6,(c,err);maxerr=max(maxerr,err)
  for tick,ts in enumerate(t['t']):
   phase=e['start_stage'] if fixed else min(6.,e['start_stage']+(ts-.02)/2)
   l=int(phase);h=min(l+1,6);alpha=phase-l;alpha=alpha*alpha*(3-2*alpha);ref=bank[surf,l]*(1-alpha)+bank[surf,h]*alpha
   a=0 if fixed else np.clip((phase-2)/4,0,1);a=a*a*(3-2*a);k=p.get('knee',0)*a*(e['start_stage']<=p.get('knee_start_max',6))
   for idx,n in enumerate(names):
    if 'knee' in n:ref[idx]+=k
    elif 'hip_pitch' in n or 'ankle_pitch' in n:ref[idx]-=.5*k
   diff=t['reference'][tick,c]-ref
   ankle=[names.index('left_ankle_pitch_joint'),names.index('right_ankle_pitch_joint')]
   rest=[i for i in range(37) if i not in ankle]
   assert abs(diff[rest]).max()<3e-6
   cap=p.get('cap',.15) if p.get('kp',0) and not fixed else 0.
   assert abs(diff[ankle]).max()<=cap+3e-6
   assert abs(diff[ankle[0]]-diff[ankle[1]])<3e-6
   referr=max(referr,float(abs(diff[rest]).max()))
 assert np.allclose((t['torque'].astype(float)**2).sum(-1),t['effort'],rtol=2e-6,atol=.02)
 first=read(folder/'first_failures.json');assert {x['case'] for x in first}=={x['case'] for x in rows if any(x['failures'].values())}
 audit=read(folder/'sensor_audit.json');assert audit['channels']==405 and audit['physics_advances_during_camera']==0
 cut={x['case']:x['t'] for x in first};geom=[]
 for e in rows:
  c=e['case'];z=[x for x in audit['truth_validation'] if x['case']==c and x['t']<cut.get(c,20.01)];valid=[x for x in z if x['valid']]
  moving=(t['t']<cut.get(c,20.01))&((t['root_speed'][:,c]>.02)|(t['gyro'][:,c]>.05))
  geom.append(dict(case=c,stage=e['start_stage'],samples=len(z),valid=len(valid),max_offset_mm=max([abs(x['offset_error_m'])*1000 for x in valid],default=None),max_normal_deg=max([x['normal_error_deg'] for x in valid],default=None),moving_samples=int(moving.sum()),moving_missing=int((t['geometry'][moving,c,11]==0).sum())))
 standing=[x for x in rows if x['start_stage']==0]
 return dict(count=count,successes=r['successes'],physical_failures=sum(any(x['failures'].values()) for x in rows),standing_count=len(standing),standing_successes=sum(x['success'] for x in standing),standing_physical_failures=sum(any(x['failures'].values()) for x in standing),failures_by_flag={k:sum(x['failures'][k] for x in rows) for k in rows[0]['failures']},successes_by_stage={str(s):dict(count=sum(x['start_stage']==s for x in rows),success=sum(x['start_stage']==s and x['success'] for x in rows)) for s in range(7)},max_command_change_100ms=change,command_envelope_error=maxerr,reference_other_joint_error=referr,constant_uncontrolled_commands=True,constant_fixed_commands=fixed,geometry=geom),r

def compare(policy,baseline):
 pairs=[]
 for p,b in zip(policy['episodes'],baseline['episodes']):
  assert (p['case'],p['surface_id'],p['start_stage'])==(b['case'],b['surface_id'],b['start_stage'])
  if not (p['success'] and b['success']):continue
  for window in ['metrics_whole_trial','metrics_last10s','metrics_last5s']:
   for metric in p[window]:
    pv=p[window][metric];bv=b[window][metric]
    pairs.append(dict(case=p['case'],stage=p['start_stage'],window=window,metric=metric,policy=pv,baseline=bv,change_percent=100*(pv/bv-1) if bv else None))
 return pairs

def preflight(data,surface):
 assert read(data/f'{surface}_preflight_status_02.json')['phase']=='complete_pending_verification'
 report=dict(passed=False,surface=surface,evaluations={},startups={})
 fixed=data/f'{surface}_fixed_01'
 for label in ['fixed_01','old_reference_01','reference_validation_02','smoke_evaluation_02']:
  checkpoint=data/f'{surface}_optimizer_smoke_02/final.pt' if label=='smoke_evaluation_02' else None
  folder=data/f'{surface}_{label}';check,rows=evaluation(folder,14,checkpoint);report['evaluations'][label]=check
  report['startups'][label]=starts(fixed,folder)
  if label in ['reference_validation_02','smoke_evaluation_02']:assert check['successes']==14 and check['physical_failures']==0
  if label=='fixed_01':assert check['physical_failures']==0
 assert read(data/f'{surface}_reset_isolation_02/reset_probe.json')['passed'];report['reset_isolation']=True
 smoke=data/f'{surface}_optimizer_smoke_02';check_manifest(smoke);v=read(smoke/'verification.json');assert v==dict(updates=2,transitions=2048,active=2048,warmup=0)
 initial=load(smoke/'initial.pt');assert torch.count_nonzero(initial['actor_state_dict']['mlp.4.weight'])==0 and torch.count_nonzero(initial['actor_state_dict']['mlp.4.bias'])==0
 report['changed_weights']=weights(initial,load(smoke/'final.pt'));report['smoke_budget']=v;report['passed']=True
 return report

def study(folder):
 protocol=read(folder/'protocol.json');report=dict(passed=False,surface=protocol['surface'],training={},evaluations={},startup_matching={},comparisons={},limitations=protocol['limitations'])
 for i,(p,h) in enumerate(protocol['hashes'].items()):assert sha(p)==h and sha(folder/'frozen'/f'{i:03d}_{Path(p).name}')==h,p
 for p,h in protocol['prior_hashes'].items():assert sha(p)==h,p
 report['frozen_hashes']=len(protocol['hashes']);report['prior_hashes']=len(protocol['prior_hashes'])
 for seed in protocol['seeds']:
  state=read(folder/f'seed_{seed}.json');assert state['phase']=='complete';previous=None;updates=0;transitions=0;active=0;warm=0;blocks=[]
  for number,stage in enumerate(protocol['stages']):
   label=f'train_{seed}_block{number}_stage{stage}';d=folder/label;m=check_manifest(d);v=read(d/'verification.json');assert v['updates']==32 and v['transitions']==32768 and v['active']+v['warmup']==32768
   assert m['args']['seed']==seed and m['args']['start_stage']==str(stage)
   initial=load(d/'initial.pt');final=load(d/'final.pt');changes=weights(initial,final)
   if previous:
    prev=load(previous);assert all(equal(initial[k],prev[k]) for k in ['actor_state_dict','critic_state_dict','optimizer_state_dict']);assert initial['iter']==prev['iter']+1;assert m['args']['checkpoint']==str(previous)
   else:
    assert initial['iter']==0 and m['args']['checkpoint'] is None
    assert torch.count_nonzero(initial['actor_state_dict']['mlp.4.weight'])==0 and torch.count_nonzero(initial['actor_state_dict']['mlp.4.bias'])==0
   updates+=32;transitions+=v['transitions'];active+=v['active'];warm+=v['warmup'];assert final['iter']==updates-1
   episodes=read(d/'results.json')['episodes'];pending=len(episodes)*25-v['warmup'];unfinished=v['active']-sum(x['steps'] for x in episodes);assert 0<=pending<=400 and 0<=unfinished<=16000
   previous=d/'final.pt';blocks.append(dict(stage=stage,budget=v,weights=changes,pending_warmup=pending,unfinished_active=unfinished,checkpoint_sha256=sha(previous)))
  assert (updates,transitions)==(128,131072);assert state['last_checkpoint']==str(previous)
  assert (state['updates'],state['transitions'],state['active_transitions'],state['warmup_transitions'])==(updates,transitions,active,warm)
  report['training'][str(seed)]=dict(updates=updates,transitions=transitions,active=active,warmup=warm,blocks=blocks)
 datasets={}
 for label,checkpoint in [('fixed',None),('old_reference',None),('reference',None)]+[(f'policy_{seed}',Path(read(folder/f'seed_{seed}.json')['last_checkpoint'])) for seed in protocol['seeds']]:
  check,data=evaluation(folder/label,28,checkpoint);report['evaluations'][label]=check;datasets[label]=data;report['startup_matching'][label]=starts(folder/'fixed',folder/label)
 for label in ['reference']+[f'policy_{s}' for s in protocol['seeds']]:
  report['comparisons'][label]={b:compare(datasets[label],datasets[b]) for b in ['fixed','old_reference','reference'] if b!=label}
 report['passed']=True
 return report

def main():
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--surface',choices=['sitting','leaning']);p.add_argument('--preflight',action='store_true');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 a.output.mkdir(exist_ok=False);shutil.copy2(__file__,a.output/'verification_source.py')
 report=preflight(a.data,a.surface) if a.preflight else study(a.data)
 save(a.output/'verification.json',report);print(json.dumps({k:v for k,v in report.items() if k not in ['evaluations','comparisons','training']},indent=2))
if __name__=='__main__':main()
