"""Independent final comparison. Reads frozen runs and creates a new report."""
import argparse,hashlib,html,json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from humanoid_sitting.deployment import BODY_JOINT_NAMES

p=argparse.ArgumentParser();p.add_argument('--study',required=True);p.add_argument('--output',required=True);a=p.parse_args()
r,out=Path(a.study),Path(a.output);read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
protocol=read(r/'protocol.json');check={'training':{},'evaluations':{},'fixed':{}}
for label,hashes in [('frozen',protocol['hashes']),('prior',protocol['prior_study_hashes'])]:
    check[label]={p:sha(Path(p))==h for p,h in hashes.items()}
    assert all(check[label].values()),label
summary={};details=[]
for bank,n in [('train',13),('holdout',16)]:
    fixed=Path(protocol['fixed_baselines'][bank]);data=read(fixed/'results.json');tr=np.load(fixed/'trace.npz');s=np.load(fixed/'startup.npz')
    assert len(data['episodes'])==n and data['successes']==n
    assert all(x['steps']==1000 and x['duration_s']==20 and x['last10s_ticks']==500 and x['success'] for x in data['episodes'])
    assert float(abs(tr['command']-s['command'][None]).max())==0
    check['fixed'][bank]=dict(success=n,total=n,constant_command=True)

for seed in protocol['seeds']:
    train=r/f'train_{seed}';v=read(train/'verification.json');progress=read(train/'progress.json')
    assert v['updates']==256 and v['transitions']==425984 and v['active']+v['warmup']==425984
    assert all(v[k]==progress[k] for k in ['updates','transitions','active','warmup'])
    initial=torch.load(train/'initial.pt',map_location='cpu',weights_only=False);final=torch.load(train/'final.pt',map_location='cpu',weights_only=False)
    assert final['iter']==255
    changed={role:[k for k,x in final[role].items() if not torch.equal(x,initial[role][k])] for role in ['actor_state_dict','critic_state_dict']}
    assert any(k.startswith('mlp') for k in changed['actor_state_dict'])
    assert all(torch.isfinite(x).all() for role in changed for x in final[role].values())
    assert read(train/'initial_policy_check.json')['zero_action']
    rows=read(train/'results.json')['episodes'];pending=len(rows)*25-v['warmup'];unfinished=v['active']-sum(x['steps'] for x in rows)
    assert 0<=pending<=26*25 and 0<=unfinished<26*1000
    assert rows==[json.loads(x) for x in (train/'episodes.jsonl').read_text().splitlines()]
    check['training'][str(seed)]=dict(**v,changed=changed,completed_episodes=len(rows),pending_warmup=pending,unfinished_active=unfinished,final_sha256=sha(train/'final.pt'))
    for bank,n in [('train',13),('holdout',16)]:
        label=f'eval_{seed}_{bank}';d=r/label;fixed=Path(protocol['fixed_baselines'][bank])
        loaded=torch.load(d/'initial.pt',map_location='cpu',weights_only=False)
        assert all(torch.equal(x,loaded['actor_state_dict'][k]) for k,x in final['actor_state_dict'].items())
        assert Path(read(d/'manifest.json')['args']['checkpoint'])==train/'final.pt'
        assert read(d/'geometry.json')==read(fixed/'geometry.json')
        for k,h in read(d/'manifest.json')['source_hashes'].items():
            assert protocol['hashes'][str(Path(k).resolve())]==h
        s=np.load(d/'startup.npz');f=np.load(fixed/'startup.npz')
        errors={k:float(abs(s[k]-f[k]).max()) for k in s.files};assert all(x==0 for x in errors.values()),(label,errors)
        data=read(d/'results.json');rows=data['episodes'];base=read(fixed/'results.json')['episodes'];geometry=read(d/'geometry.json');tr=np.load(d/'trace.npz')
        assert len(rows)==n and np.allclose(tr['t'],np.arange(1,201)/10,atol=1e-12,rtol=0)
        ids=[data['joint_names'].index(k) for k in BODY_JOINT_NAMES];other=[k for k in range(len(data['joint_names'])) if k not in ids]
        span=np.array(read(d/'manifest.json')['spans'])
        assert np.all(abs(tr['command'][:,:,ids]-s['command'][None,:,ids])<=span+1e-6)
        assert float(abs(np.diff(tr['command'],axis=0)).max())<=.010001
        assert np.array_equal(tr['command'][:,:,other],np.broadcast_to(s['command'][None,:,other],tr['command'][:,:,other].shape))
        for x in rows:
            assert x['steps']==1000 and x['duration_s']==20 and x['last10s_ticks']==500 and x['reset_index']==5
            assert x['success']==(not any(x['failures'].values()) and x['good_fraction']>=.95 and x['final_hold_s']>=2)
            if x['success']:assert tr['good'][-20:,x['case']].all()
        common=[i for i in range(n) if rows[i]['success'] and base[i]['success']]
        ratios={}
        for window in ['metrics_whole_trial','metrics_last10s']:
            ratios[window]={}
            for metric in ['total_squared_torque_Nm2','total_absolute_torque_Nm','all_joint_rms_Nm','normalized_torque_squared','leg_squared_torque_Nm2','upper_squared_torque_Nm2']:
                before=np.array([base[i][window][metric] for i in common]);after=np.array([rows[i][window][metric] for i in common])
                ratios[window][metric]=dict(fixed_mean=float(before.mean()) if common else None,ppo_mean=float(after.mean()) if common else None,
                    pooled_change_percent=float(100*(after.mean()/before.mean()-1)) if common and before.mean()>0 else None,
                    mean_paired_change_percent=float(np.mean(100*(after/before-1))) if common and (before>0).all() else None)
        by_kind={kind:dict(total=sum(g['kind']==kind for g in geometry),success=sum(x['success'] for x,g in zip(rows,geometry) if g['kind']==kind)) for kind in ['sit','lean']}
        first={x['case']:x['t'] for x in read(d/'first_failures.json')}
        samples=[];body=data['body_names'].index('pelvis')
        for i,g in enumerate(geometry):
            before_failure=tr['t']<first.get(i,float('inf'))-1.e-10
            local=tr['body'][:,i,body,:3]-np.array(data['origins'][i]);rot=Rotation.from_quat(tr['body'][:,i,body,3:]).as_matrix()
            normal=np.einsum('nji,j->ni',rot,np.array(g['normal']));offset=(local-np.array(g['top_center']))@np.array(g['normal'])
            for j in np.flatnonzero(before_failure):
                z=tr['geometry'][j,i]
                samples.append(dict(valid=bool(z[11]),offset_mm=float(1000*abs(z[5]-offset[j])),normal_deg=float(np.rad2deg(np.arccos(np.clip(z[2:5]@normal[j],-1,1))))))
            paired=i in common
            details.append(dict(seed=seed,bank=bank,case=i,geometry=g,success=rows[i]['success'],failures=rows[i]['failures'],
                common_success=paired,fixed=base[i],ppo=rows[i],
                late_effort_change_percent=100*(rows[i]['metrics_last10s']['total_squared_torque_Nm2']/base[i]['metrics_last10s']['total_squared_torque_Nm2']-1) if paired else None))
        valid=[s for s in samples if s['valid']]
        sensor=dict(recorded_samples_before_failure=len(samples),valid=len(valid),lost=len(samples)-len(valid),
            p95_abs_offset_mm=float(np.percentile([x['offset_mm'] for x in valid],95)) if valid else None,
            max_abs_offset_mm=max([x['offset_mm'] for x in valid],default=None),
            p95_normal_deg=float(np.percentile([x['normal_deg'] for x in valid],95)) if valid else None,
            scope='10Hz saved motion samples before first failure; not every50Hz tick')
        check['evaluations'][label]=dict(exact_startup_errors=errors,final_actor_loaded=True,full_duration=True,bounded_commands=True,unchanged_other_commands=True)
        summary[label]=dict(success=sum(x['success'] for x in rows),total=n,common_successes=len(common),by_kind=by_kind,
            physical_failures={k:sum(x['failures'][k] for x in rows) for k in ['fall','forbidden','impact','torque','joint_limit']},paired_effort=ratios,sensor=sensor)

out.mkdir(exist_ok=False);check['passed']=True
(out/'validation.json').write_text(json.dumps(check,indent=2));(out/'results.json').write_text(json.dumps(dict(summary=summary,details=details),indent=2))
fig,axes=plt.subplots(1,2,figsize=(11,4.5))
for ax,bank,n in zip(axes,['train','holdout'],[13,16]):
    values=[n]+[summary[f'eval_{s}_{bank}']['success'] for s in protocol['seeds']]
    ax.bar(['Fixed posture','PPO 54511','PPO 54517'],values,color=['#829188','#247e98','#d59140']);ax.set_ylim(0,n+1)
    ax.set(title='Training surfaces' if bank=='train' else 'Held-out surfaces',ylabel='Successful full 20-second rests')
    for i,x in enumerate(values):ax.text(i,x+.15,f'{x}/{n}',ha='center')
    ax.spines[['top','right']].set_visible(False)
fig.tight_layout();fig.savefig(out/'stability.png',dpi=180)
fig,axes=plt.subplots(1,2,figsize=(13,4.8))
for ax,bank in zip(axes,['train','holdout']):
    for seed,color in zip(protocol['seeds'],['#247e98','#d59140']):
        points=[x for x in details if x['seed']==seed and x['bank']==bank]
        ax.plot([x['case'] for x in points],[x['late_effort_change_percent'] if x['common_success'] else np.nan for x in points],marker='o',label=str(seed),color=color)
    ax.axhline(0,color='#777',lw=1);ax.axhline(-10,color='#4b8258',ls=':',label='10% saving')
    ax.set(title=bank,xlabel='Surface case (see table)',ylabel='Final-10-second total squared torque change (%)');ax.legend()
    ax.spines[['top','right']].set_visible(False)
fig.suptitle('Negative values save effort; failed trials are omitted, never scored as savings')
fig.tight_layout();fig.savefig(out/'effort_by_surface.png',dpi=180)
table=''
for key,x in summary.items():
    delta=x['paired_effort']['metrics_last10s']['total_squared_torque_Nm2']['pooled_change_percent']
    table+=f'<tr><td>{key}</td><td>{x["success"]}/{x["total"]}</td><td>{x["common_successes"]}</td><td>{"not available" if delta is None else f"{delta:+.2f}%"}</td></tr>'
percase=''
for x in details:
    g=x['geometry'];change=x['late_effort_change_percent'];fail=', '.join(k for k,v in x['failures'].items() if v) or 'none'
    percase+=f'<tr><td>{x["seed"]}</td><td>{x["bank"]}/{x["case"]}</td><td>{g["kind"]}</td><td>{100*g["height"]:.1f}</td><td>{g["pitch_deg"]:g}</td><td>{g.get("shape","rect")}</td><td>{x["success"]}</td><td>{"—" if change is None else f"{change:+.2f}%"}</td><td>{html.escape(fail)}</td></tr>'
page='''<!doctype html><meta charset="utf-8"><title>Direct effort PPO comparison</title><style>body{max-width:1150px;margin:35px auto;padding:0 20px;font:17px/1.5 system-ui;color:#253443}img{width:100%}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:8px;text-align:left;border-bottom:1px solid #ddd}h1{font-size:32px}h2{margin-top:32px}a{color:#167395}</style><h1>Learning lower-effort sitting and leaning</h1><p>Direct total squared joint torque objective, starting from supported postures. Stable support and physical safety are required. These results do not test reaching support from standing.</p><img src="stability.png"><table><tr><th>Policy / bank</th><th>Stable successes</th><th>Common successes</th><th>Total squared torque change</th></tr>'''+table+'''</table><p>Effort deltas use final-ten-second means on common full-duration successes only. Whole-trial and additional torque metrics are in the results file. A saving on surviving cases cannot compensate for losing stability elsewhere.</p><img src="effort_by_surface.png"><h2>Per-surface comparison</h2><table><tr><th>Seed</th><th>Bank/case</th><th>Task</th><th>Center cm</th><th>Slope °</th><th>Outline</th><th>Success</th><th>Effort change</th><th>Physical failures</th></tr>'''+percase+'''</table><h2>Verification and limits</h2><p>Both full budgets, finite changed weights, correct final checkpoint loading, bitwise matched startups, full evaluation durations, bounded learned commands, constant fixed commands and frozen source/bank hashes passed. Active and reset-warmup accounting is separate in validation.json. The actor and critic use sensor-derived features and remembered motor commands. Simulator truth is confined to resets, rewards and validation labels.</p><p>Two seeds reuse 29 prealigned rigid surfaces. The 16 held-out surfaces were withheld from this policy’s training but appeared in prior project evaluations. One outline endpoint was historically borderline and remains included. Torque is an effort proxy, not measured energy or proof of a global minimum. A supported start does not demonstrate autonomous approach or sitting from standing. No policy is promoted automatically.</p><p><a href="validation.json">Verification</a> · <a href="results.json">Full metrics, physical failures and tracking accuracy</a></p>'''
(out/'review.html').write_text(page)
print(json.dumps(summary,indent=2))
