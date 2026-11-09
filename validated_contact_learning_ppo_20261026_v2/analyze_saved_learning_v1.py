"""Read-only saved-rollout analysis. No simulator, optimizer, or new trajectory."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn

ROOT = Path('/data/humanoid-sitting-perception/low_seat_contact_01/contact_learning_ppo_20261026_v2')
torch.set_num_threads(1)
OUT = {'classification': 'Saved-data arithmetic only; not a causal ablation or physical repair', 'sources': {}, 'runs': {}}

def bind(p):
    OUT['sources'][str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    return p

def first(mask):
    ix = np.flatnonzero(mask)
    return round((int(ix[0])+1)*.02, 3) if len(ix) else None

def stats(x):
    x = np.asarray(x)
    return dict(mean=float(x.mean()), std=float(x.std()), min=float(x.min()), max=float(x.max()))

def net(checkpoint, key, obs):
    model = nn.Sequential(nn.Linear(417,32),nn.Tanh(),nn.Linear(32,32),nn.Tanh(),nn.Linear(32,3 if key=='actor_state_dict' else 1))
    model.load_state_dict(checkpoint[key])
    x=((obs.float()-checkpoint['input_mean'])/checkpoint['input_scale']).clamp(-5,5)
    with torch.no_grad(): return model(x)

for name in ['calibrate','train_63111','train_63117'] + [f'evaluate_{seed}_{bank}' for seed in [63111,63117] for bank in ['known_old','known_transfer','known_recent']]:
    p=ROOT/name
    z=np.load(bind(p/'every_tick.npz'))
    rows=json.loads(bind(p/'results.json').read_text())['episodes']
    batches=torch.load(bind(p/'decision_batches.pt'), map_location='cpu', weights_only=False)
    offsets=np.load(bind(p/'applied_offsets.npz'))['offsets']
    n=len(batches)
    assert len(rows)==24*n and len(z['t'])==1000*n+25*(n-1)
    rr=[]
    for it,b in enumerate(batches):
        start=it*1025
        r={k:z[k][start:start+1000] for k in z.files}
        assert np.max(np.abs(r['t']-np.arange(1,1001)*.02)) < 1e-9
        elapsed=np.arange(1,1001)*.02
        allowed=r['permitted'] & r['inside'] & r['rest_patch']
        load=(np.maximum((r['force']*r['normal']).sum(-1),0)*allowed).sum(-1)/r['weight']
        gate=r['legal'] & (r['wrong']<=10) & ~r['failure'] & r['alive'] & (elapsed[:,None]>=8)
        still=np.exp(-(r['root_speed']/.1)**2-(r['gyro']/.25)**2)
        loaded=np.linalg.norm(r['force'],axis=-1)>.01*r['weight'][...,None]
        excess=np.where(loaded,np.maximum(r['slip']-.02,0),0).max(-1)
        components={
            'stationary_support_bonus': .04*np.clip(load/.2,0,1)*still*gate,
            'good_rest_bonus': .04*r['good']*gate,
            'slip_cost': -.02*500*excess,
            'wrong_contact_cost': -.02*20*np.maximum(r['wrong']-10,0)/r['weight'],
            'illegal_patch_cost': -.1*(~r['legal']),
        }
        dense=sum(components.values()).reshape(200,5,24).sum(1)
        assert np.max(np.abs(dense-b['physical_cost'].numpy())) < 2e-5
        ds=np.array(b['decision_steps'])
        tail=lambda a: np.cumsum(a[::-1],axis=0)[::-1][ds]
        terms={k:tail(v.reshape(200,5,24).sum(1)) for k,v in components.items()}
        terms['effort_delta_cost']=tail(-.25*(b['torque_integrals']-b['baseline_integrals']).numpy()/1350)
        terms['terminal_penalty']=-np.broadcast_to(b['terminal_penalty'].numpy(),(len(ds),24))
        reward=b['reward'].numpy()
        err=float(np.max(np.abs(sum(terms.values())-reward)))
        assert err<1e-4
        erows=rows[it*24:(it+1)*24]
        trials=[]
        for j,row in enumerate(erows):
            assert int((~r['no_slip'][:,j]).sum())==row['slip_ticks']
            assert int((~r['legal'][:,j]).sum())==row['illegal_ticks']
            fs=first(~r['no_slip'][:,j])
            trials.append(dict(case=j,geometry_index=row['geometry_index'],success=row['success'],failures=row['failures'],
                slip_ticks=row['slip_ticks'],first_slip_s=fs,last_slip_s=(round((np.flatnonzero(~r['no_slip'][:,j])[-1]+1)*.02,3) if fs else None),
                first_physical_failure_s=first(r['failure'][:,j]),first_unexpected_contact_s=first(r['wrong'][:,j]>10),
                first_legal_support_005bw_s=first(load[:,j]>=.05),first_good_s=first(r['good'][:,j]),
                decisions_at_or_after_first_slip=int((ds*.1>=fs).sum()) if fs else 0,
                slip_excess_integral_m=float(excess[:,j].sum()*.02),
                post8_bonus_gate_fraction=float(gate[399:,j].mean()),
                return_at_first_decision=float(reward[0,j]),mean_reward_to_go=float(reward[:,j].mean()),
                reward_terms_mean_to_go={k:float(v[:,j].mean()) for k,v in terms.items()}))
        of=offsets[it*200:(it+1)*200,0][60:]
        act=b['actions'].numpy();mu=b['old_mean'].numpy()
        entry=dict(iteration=it+1,arithmetic_max_abs_error=err,trials=trials,
            reward_terms_mean_to_go={k:float(v.mean()) for k,v in terms.items()},
            reward_terms_failed8={k:float(v[:,:8].mean()) for k,v in terms.items()} if not name.startswith('evaluate') else None,
            reward_terms_training_clean16={k:float(v[:,8:].mean()) for k,v in terms.items()} if not name.startswith('evaluate') else None,
            action=dict(applied_residual_rms_rad=float(np.sqrt(np.mean(of**2))),applied_max_abs_rad=float(np.abs(of).max()),
                applied_fraction_over_95pct_bound=float((np.abs(of)>=.0038).mean()),
                raw_target_rms_rad=float(np.sqrt(np.mean((.004*np.tanh(act))**2))),
                policy_mean_target_rms_rad=float(np.sqrt(np.mean((.004*np.tanh(mu))**2))),
                policy_mean_target_max_rad=float(np.abs(.004*np.tanh(mu)).max())),
            return_stats=stats(reward))
        if name.startswith('train'):
            prev=p/('initial.pt' if it==0 else f'model_{it:02d}.pt')
            curr=p/f'model_{it+1:02d}.pt'
            ev={}
            for label,file in [('before',prev),('after',curr)]:
                cp=torch.load(bind(file),map_location='cpu',weights_only=False)
                val=net(cp,'critic_state_dict',b['obs']).squeeze(-1).numpy()
                ev[label]=dict(explained_variance=float(1-np.var(reward-val)/np.var(reward)),mse=float(np.mean((reward-val)**2)),value_range=[float(val.min()),float(val.max())])
            entry['critic']=ev
            norm=((b['obs']-cp['input_mean'])/cp['input_scale']).numpy()
            entry['observation_clip_fraction']=float((np.abs(norm)>5).mean())
        rr.append(entry)
    OUT['runs'][name]=rr
print(json.dumps(OUT,allow_nan=False))
