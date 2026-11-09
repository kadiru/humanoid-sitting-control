"""Read-only saved-trajectory arithmetic. No simulation, optimization, or CUDA.

Fixed scope: all completed 16/32-update runs plus revised63111 at64.
Compare original failed identities, applied/target saturation, slew engagement,
failure timing, stochastic clean witnesses, and in-batch critic fit. This does
not establish reachability or causality. Only completed audited folders read.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import json,hashlib,time,resource,datetime
from pathlib import Path
import numpy as np
import torch
from torch import nn
torch.set_num_threads(1)
resource.setrlimit(resource.RLIMIT_CPU,(120,125))
START=time.monotonic()
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')
P=D/'contact_reward_comparison_128_v4'
OUT={'classification':'Saved-data arithmetic; no simulator, optimizer, CUDA, new trajectories or causal intervention', 'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'sources':{},'baseline':{},'runs':{}}

def bind(p):
    p=Path(p)
    if str(p) not in OUT['sources']:
        h=hashlib.sha256()
        with p.open('rb') as f:
            for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
        OUT['sources'][str(p)]=h.hexdigest()
    return p
def read(p):return json.loads(bind(p).read_text())
def load(p):return torch.load(bind(p),map_location='cpu',weights_only=False)
def clean(r):return bool(r['success'] and not any(r['failures'].values()) and not r['slip_ticks'] and not r['illegal_ticks'])
def first(x):
    ix=np.flatnonzero(x)
    return round((int(ix[0])+1)*.02,3) if len(ix) else None
def summary(x):
    x=np.asarray(x)
    return {'min':float(x.min()),'mean':float(x.mean()),'max':float(x.max())}
def critic(cp,obs,reward):
    model=nn.Sequential(nn.Linear(417,32),nn.Tanh(),nn.Linear(32,32),nn.Tanh(),nn.Linear(32,1))
    model.load_state_dict(cp['critic_state_dict'])
    norm=(obs.float()-cp['input_mean'])/cp['input_scale']
    with torch.no_grad():value=model(norm.clamp(-5,5)).squeeze(-1).numpy()
    return {'explained_variance':float(1-np.var(reward-value)/np.var(reward)), 'mse':float(np.mean((reward-value)**2)), 'observation_clip_fraction':float((norm.abs()>5).float().mean())}

tickkeys=['t','no_slip','legal','failure','wrong','traction','slip','force','weight','unexpected_body_normal']
def ticks(p):
    with np.load(bind(p/'every_tick.npz')) as z:return {k:z[k] for k in tickkeys}

def trial(row,r,j,names):
    assert int((~r['no_slip'][:,j]).sum())==row['slip_ticks']
    assert int((~r['legal'][:,j]).sum())==row['illegal_ticks']
    assert bool(r['failure'][:,j].any())==any(row['failures'].values())
    loaded=np.linalg.norm(r['force'][:,j],axis=-1)>.01*r['weight'][:,j,None]
    speed=np.where(loaded,r['slip'][:,j],0).max(-1)
    normals=r['unexpected_body_normal'][:,j]
    peak=np.unravel_index(normals.argmax(),normals.shape)
    return {'case':row['case'],'clean':clean(row),'stable':row['success'],'strict_transfer':row['strict_transfer'],
            'slip_ticks':row['slip_ticks'],'illegal_ticks':row['illegal_ticks'],'traction_bad_ticks':row['traction_bad_ticks'],
            'failures':row['failures'],'final_hold_s':row['final_hold_s'],
            'first_slip_s':first(~r['no_slip'][:,j]),'first_physical_failure_s':first(r['failure'][:,j]),
            'first_wrong_contact_gt10N_s':first(r['wrong'][:,j]>10),'wrong_force_peak_N':float(r['wrong'][:,j].max()),
            'loaded_speed_peak_m_s':float(speed.max()),'loaded_excess_over_002_integral_m':float(np.maximum(speed-.02,0).sum()*.02),
            'unexpected_body_peak':{'body':names[peak[1]],'normal_N':float(normals[peak]),'time_s':round((peak[0]+1)*.02,3)}}

baseline={}
for bank in ['known_old','known_transfer','known_recent']:
    p=D/'current_retention_09'/bank/'candidate'
    result=read(p/'results.json');r=ticks(p)
    assert len(result['episodes'])==24
    for j,row in enumerate(result['episodes']):
        key=bank+'/'+str(row['case']);baseline[key]=row
        if not clean(row):OUT['baseline'][key]=trial(row,{k:v[:1000] for k,v in r.items()},j,result['body_names'])
assert len(OUT['baseline'])==8
selection=read(P/'training_bank/selection.json')['cases']
mapping=[s['bank']+'/'+str(2*s['index']+profile) for s in selection for profile in range(2)]
assert all(not clean(baseline[k]) for k in mapping[:8])
OUT['training_case_identity']=mapping

stages=[(a,s,n) for n in [16,32] for a,s in [('original',63111),('original',63117),('revised',63111),('revised',63117)]]+[('revised',63111,64)]
folders=[]
for a,s,n in stages:
    folders.append((f'train_{a}_{s}_{n:03d}',None))
    folders.extend((f'eval_{a}_{s}_{n:03d}_{bank}',bank) for bank in ['known_old','known_transfer','known_recent'])
for name,bank in folders:
    p=P/name
    assert read(p/'independent_audit.json')['passed']
    result=read(p/'results.json');rows=result['episodes'];rall=ticks(p)
    batches=load(p/'decision_batches.pt');initial=load(p/'initial.pt')
    offsets=np.load(bind(p/'applied_offsets.npz'))['offsets']
    with np.load(bind(p/'reward_geometry.npz')) as z:geom=dict(z)
    la=read(p/'learning_audit.json')
    assert len(rows)==24*len(batches)
    assert len(rall['t'])==1000*len(batches)+25*(len(batches)-1)
    entries=[]
    for it,b in enumerate(batches):
        r={k:v[it*1025:it*1025+1000] for k,v in rall.items()}
        assert np.max(np.abs(r['t']-np.arange(1,1001)*.02))<1e-9
        erows=rows[it*24:(it+1)*24];ds=np.array(b['decision_steps'])
        act=b['actions'].numpy();mu=b['old_mean'].numpy()
        target=.004*np.tanh(act);mean_target=.004*np.tanh(mu)
        applied=offsets[it*200:(it+1)*200,0]
        prev=np.concatenate([np.zeros_like(applied[:1]),applied[:-1]])
        step_target=np.zeros_like(applied)
        for z,d in enumerate(ds):step_target[d:d+5]=target[z]
        requested=step_target-prev
        expected=prev+np.clip(requested,-.0004,.0004)
        maxerr=float(np.max(np.abs(expected-applied)))
        assert maxerr<2e-8,maxerr
        assert np.abs(applied).max()<=.00400001
        assert np.abs(offsets[it*200:(it+1)*200,2]).max()<=.01200001
        ts=[]
        for j,row in enumerate(erows):
            identity=mapping[j] if bank is None else bank+'/'+str(row['case'])
            assert row['profile']==baseline[identity]['profile']
            t=trial(row,r,j,result['body_names']);t['identity']=identity;t['baseline_clean']=clean(baseline[identity])
            for label,mask in [('active',np.arange(200)>=60),('landing', (np.arange(200)>=90)&(np.arange(200)<130))]:
                v=applied[mask,j];wanted=step_target[mask,j];req=requested[mask,j]
                t[label+'_action']={'applied_max_abs_rad':float(np.abs(v).max()),'applied_max_abs_per_joint_rad':np.abs(v).max(0).tolist(),
                    'applied_fraction_ge95pct_bound':float((np.abs(v)>=.0038).mean()),
                    'target_max_abs_rad':float(np.abs(wanted).max()),'target_fraction_ge95pct_bound':float((np.abs(wanted)>=.0038).mean()),
                    'slew_limited_fraction':float((np.abs(req)>.00040001).mean()),'mean_abs_target_lag_rad':float(np.abs(wanted-v).mean())}
            t['mean_policy_target_max_rad']=float(np.abs(mean_target[:,j]).max())
            t['pelvis_gap_min_9to13_m']=float(geom['pelvis_target_distance_m'][it,89:130,j].min())
            t['housing_clearance_min_9to13_m']=float(geom['housing_vertex_clearance_m'][it,89:130,j].min())
            t['return6s']=float(b['reward'][0,j]);t['revised_return6s']=float(b['revised_reward'][0,j])
            ts.append(t)
        entry={'global_updates':initial['updates']+it+1 if bank is None else initial['updates'],
               'trials':ts,'slew_reconstruction_max_abs_error':maxerr,'optimizer':la[it]['optimizer']}
        if bank is None:
            before=p/('initial.pt' if it==0 else f'model_{initial["updates"]+it:03d}.pt')
            cp=load(before)
            entry['critic_before']=critic(cp,b['obs'],b['reward'].numpy())
            after=load(p/f'model_{initial["updates"]+it+1:03d}.pt')
            entry['critic_after']=critic(after,b['obs'],b['reward'].numpy())
        entries.append(entry)
    OUT['runs'][name]=entries
    del rall,batches,offsets,geom
OUT['resource']={'elapsed_seconds':time.monotonic()-START,'cpu_threads':1,'cuda_used':False,'maxrss_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
print(json.dumps(OUT,allow_nan=False))
