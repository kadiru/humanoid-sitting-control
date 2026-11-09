"""Independent NumPy reward/physical audit and checkpoint-change verification.

Does not import environment.py, rewards.py, or their physical evaluator.
"""
import argparse
from pathlib import Path
import numpy as np
from common import read, write, sha


def reward_numpy(d,weight=1.):
    assert weight==1.
    # Independent reference uses float64 so cancellation in potential differences
    # does not masquerade as a failure of the exact discounted-return identity.
    d={k:v.astype(np.float64) if np.issubdtype(v.dtype,np.floating) else v for k,v in d.items()}
    safe=d['alive'].astype(bool)&d['legal'].astype(bool);good=safe&d['good'].astype(bool);phase=d['phase']
    phi=2*np.exp(-np.maximum(d['q_error'],0))*safe
    phi+=2*(phase>=1)*safe*np.clip(d['support_bw']/.2,0,1)*np.exp(-(d['speed']/.1)**2-(d['gyro']/.25)**2)
    phi+=4*good*np.clip(d['held_ticks']/250,0,1)
    phi-=np.clip((d['loaded_slip']/.02)**2,0,1)+np.clip((d['wrong_force']/10)**2,0,1)+np.clip(d['peak_bw']-1,0,1)**2
    phi-=.1*np.clip((d['normalized_command_delta']**2).mean(-1),0,1)
    horizon=np.zeros_like(d['alive'],bool);horizon[-1]=True
    phi=np.where(d['first_failure']|horizon|~d['alive'],0,phi)
    previous=np.concatenate((np.zeros_like(phi[:1]),phi[:-1]),axis=0)
    if 'previous_potential' in d:np.testing.assert_allclose(d['previous_potential'],previous,rtol=3e-6,atol=3e-6)
    base=.02-20*d['first_failure']+400*d['clean_completion']+20*d['stable_completion']+40*d['clean_completion']/(1+10*d['mean_qualifying_effort'])
    return base+.999*phi-previous


def physical(d):
    m={k[9:]:v for k,v in d.items() if k.startswith('physical_')}
    force=m['force_w']; normal=m['normal_w']; fn=(force*normal).sum(-1)
    ft=np.linalg.norm(force-fn[...,None]*normal,axis=-1)
    active=np.linalg.norm(force,axis=-1)>.01*m['bodyweight_N'][...,None]
    util=np.where(ft<=1e-7,0,ft/np.maximum(m['friction']*np.maximum(fn,0),1e-7))
    legal=(~active|(m['permitted']&m['inside_patch']&(fn>=0))).all(-1)
    noslip=(~active|(m['slip_m_s']<=.02)).all(-1)
    traction=(~active|(util<=.90)).all(-1)
    load=(np.maximum(fn,0)*m['rest_patch']*m['permitted']).sum(-1)/m['bodyweight_N']
    flags=np.stack([(d['root_height']<.28)|(m['trunk_tilt_rad']>np.deg2rad(55)),
                    (d['wrong_force']>10)|(active&~m['inside_patch']).any(-1),
                    (abs(m['torque'])>m['torque_scale']+.001).any(-1),d['peak_bw']>2,~m['joint_margin_ok']],-1)
    failure=flags.any(-1); alive=~np.maximum.accumulate(failure,axis=0)
    # Initial warmup validity is checked separately, never granted here.
    good=(alive&m['joint_margin_ok']&legal&traction&noslip&(load>=.05)&
          (m['speed_m_s']<=.1)&(m['angular_speed_rad_s']<=.25)&(abs(m['trunk_tilt_rad'])<=np.deg2rad(40)))
    stable=alive[-1]&good[-250:].all(0); clean=stable&noslip.all(0)&legal.all(0)
    return dict(failure=failure,alive=alive,good=good,legal=legal,no_slip=noslip,stable=stable,clean=clean)


def evaluate(root):
    root=Path(root); rows=read(root/'results.json')['episodes']; assert 1<=len(rows)<=72
    assert all(r['full_standing_start'] and r['steps']==1000 for r in rows)
    with np.load(root/'trajectory.npz') as z:d={k:z[k] for k in z.files}
    assert d['reward'].shape==(1000,len(rows))
    for k,v in d.items():assert np.isfinite(v).all(),k
    result=physical(d)
    for k in ['failure','alive','good','legal','no_slip']:
        np.testing.assert_array_equal(result[k],d[k],err_msg=k)
    np.testing.assert_array_equal(result['clean'],[r['clean'] for r in rows])
    np.testing.assert_array_equal(result['stable'],[r['stable'] for r in rows])
    expected=np.where(d['learn'],reward_numpy(d),0)
    np.testing.assert_allclose(d['reward'],expected,rtol=3e-6,atol=3e-6)
    data=dict(passed=True,clean=result['clean'].tolist(),stable=result['stable'].tolist(),
              clean_count=int(result['clean'].sum()),rows=len(rows),independent_reward_equations=True,
              strict_predicates_exact=True,physical_tolerances_changed=False,
              numerical_reward_replay_tolerance=dict(rtol=3e-6,atol=3e-6),
              files={p.name:sha(p) for p in [root/'trajectory.npz',root/'results.json']})
    startup=read(root/'startup.json')
    assert not any(startup['warm_failure']) and all(startup['sensor_valid'])
    command=d['command'][:,:,:23]
    previous=d['previous'][:,:,:23]
    assert np.max(abs(command-previous))<=.0030002
    meta=read(root/'manifest.json')['metadata']
    expected=previous+np.clip(.1*(np.clip(d['desired'],np.array(meta['lower'])+.03,np.array(meta['upper'])-.03)-previous),-.003,.003)
    expected=np.where(d['learn'][...,None],expected,previous)
    assert np.max(abs(expected-command))<=2e-7
    assert np.array_equal(d['command'][:,:,23:],d['previous'][:,:,23:])
    mapping=root/'contact_mapping.json'
    if mapping.exists():
        mapped=read(mapping);assert mapped['calls']==1025
        data['mapped']=mapped['validated']
        data['validated_clean']=(result['clean']&np.array(mapped['validated'])).tolist()
    effort=np.mean((d['physical_torque']/d['physical_torque_scale'])**2,axis=-1)
    data['final_normalized_torque_squared']=effort[-250:].mean(0).tolist()
    data['final_mean_knee_angle_rad']=d['q'][-250:,:,[11,12]].mean(axis=(0,2)).tolist()
    data['final_absolute_mechanical_power_W']=np.abs(d['physical_torque']*d['physical_joint_velocity']).sum(-1)[-250:].mean(0).tolist()
    write(root/'independent_audit.json',data); return data


def weights(root,checkpoint='first_update.pt'):
    import torch
    root=Path(root); before=torch.load(root/'initial.pt',map_location='cpu',weights_only=False)
    after=torch.load(root/checkpoint,map_location='cpu',weights_only=False)
    changes={k:float(sum((after[k][v]-before[k][v]).double().square().sum() for v in before[k]).sqrt()) for k in ['actor','critic']}
    assert all(v>0 and np.isfinite(v) for v in changes.values()),changes
    assert after['updates']>before['updates'] and after['actor_steps']>before['actor_steps'] and after['critic_steps']>before['critic_steps']
    for key in ['actor_opt','critic_opt']:
        state=after[key]['state']; assert state and all(float(v['step'])>0 for v in state.values())
    for key in ['sitting','library']:
        for k,v in before[key].items():assert torch.equal(v,after[key][k]),('PRIOR_CHANGED',key,k)
    result=dict(passed=True,frozen_priors_unchanged=True,checkpoint=checkpoint,changes_l2=changes,updates=after['updates'],actor_steps=after['actor_steps'],critic_steps=after['critic_steps'],checkpoint_sha256=sha(root/checkpoint))
    write(root/(checkpoint.replace('.pt','')+'_audit.json'),result); return result


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('kind',choices=['evaluate','weights']); p.add_argument('root'); p.add_argument('--checkpoint',default='first_update.pt'); a=p.parse_args()
    print(evaluate(a.root) if a.kind=='evaluate' else weights(a.root,a.checkpoint))
