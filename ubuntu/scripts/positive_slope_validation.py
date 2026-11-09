"""Independent physical, checkpoint, action and CPU-optimizer checks."""
import numpy as np
import torch
from positive_slope_common import *
from posture_episode_ppo import PosturePPO
from verify_seat_load_study import equal,load,starts,evaluation,compare

def physical(folder,episodes=1,require_clean=True):
    rows=read(folder/'results.json')['episodes'];n=np.load(folder/'startup.npz')['q'].shape[0]
    assert len(rows)==episodes*n and all(x['steps']==1000 and x['duration_s']==20 for x in rows)
    clean=[bool(x['success'] and not any(x['failures'].values()) and not x['slip_ticks'] and not x['illegal_ticks']) for x in rows]
    z=np.load(folder/'every_tick.npz');f=z['force'];normal=z['normal'];fn=(f*normal).sum(-1)
    ft=np.linalg.norm(f-fn[...,None]*normal,axis=-1);loaded=np.linalg.norm(f,axis=-1)>.01*z['weight'][...,None]
    util=np.where(loaded,ft/(.6*np.maximum(fn,1e-7)),0).max(-1)
    result=dict(trials=len(rows),stable=sum(x['success'] for x in rows),clean=sum(clean),
        slip=sum(x['slip_ticks']>0 for x in rows),illegal=sum(x['illegal_ticks']>0 for x in rows),
        physical=sum(any(x['failures'].values()) for x in rows),strict=sum(x['strict_transfer'] for x in rows),
        max_friction_utilization=float(util.max()),max_conservative_margin_ticks=max(x['traction_bad_ticks'] for x in rows))
    if episodes==1:
        assert len(z['t'])==1000
        assert (~z['no_slip']).sum(0).tolist()==[x['slip_ticks'] for x in rows]
        assert (~z['traction']).sum(0).tolist()==[x['traction_bad_ticks'] for x in rows]
        if 'legal' in z:assert (~z['legal']).sum(0).tolist()==[x['illegal_ticks'] for x in rows]
    if require_clean:assert all(clean) and result['max_friction_utilization']<1.,result
    return result

def reward(rows,baseline):
    return [-(x['metrics_whole_trial']['total_squared_torque_Nm2']-baseline['whole20_Nm2'][i])*20/(9*150)
        -20*any(x['failures'].values())-10*(not x['success'])-5*(x['slip_ticks']>0)-5*(x['illegal_ticks']>0)
        -2*(x['traction_bad_ticks']>baseline['margin_ticks'][i]) for i,x in enumerate(rows)]

def replay(folder,updates):
    cal=read(DATA/'preflight/calibrate/early_calibration.json');initial=load(folder/'initial.pt')
    batches=load(folder/'decision_batches.pt');audit=read(folder/'learning_audit.json');rows=read(folder/'results.json')['episodes']
    n=len(cal['baseline']['whole20_Nm2']);assert len(batches)==len(audit)==updates and len(rows)==updates*n
    model=PosturePPO(np.zeros(414),np.ones(414),0);model.load(folder/'initial.pt')
    initial_steps={role:int(next(iter(initial[role]['state'].values()))['step']) if initial[role]['state'] else 0 for role in ['actor_optimizer','critic_optimizer']}
    for i,(b,a) in enumerate(zip(batches,audit)):
        assert b['obs'].shape==(n,414) and b['actions'].shape==(n,3)
        assert np.allclose(reward(rows[n*i:n*(i+1)],cal['baseline']),b['reward'].numpy(),atol=2e-6,rtol=1e-6)
        with torch.no_grad():
            dist=model.distribution(b['obs']);assert torch.allclose(dist.mean,b['old_mean'],atol=2e-5,rtol=2e-5)
            assert torch.allclose(dist.log_prob(b['actions']).sum(-1),b['logp'],atol=1e-4,rtol=2e-5)
        stats=model.update(b['obs'],b['actions'],b['logp'],b['old_mean'],b['reward'])
        assert stats['actor_steps']==a['optimizer']['actor_steps'] and stats['critic_steps']==8 and stats['max_accepted_kl']<=.020001
        saved=load(folder/f'model_{i+1:02d}.pt')
        for role,network in [('actor_state_dict',model.actor),('critic_state_dict',model.critic)]:
            for k,x in network.state_dict().items():assert torch.allclose(x,saved[role][k],atol=5e-5,rtol=5e-4),(folder,i,role,k,float(abs(x-saved[role][k]).max()))
        for role,field in [('actor_optimizer','actor_steps'),('critic_optimizer','critic_steps')]:
            expected=initial_steps[role]+sum(x['optimizer'][field] for x in audit[:i+1])
            assert all(int(x['step'])==expected for x in saved[role]['state'].values())
        assert saved['updates']==initial['updates']+i+1
        assert np.allclose(a['offsets_rad'],SPAN*np.tanh(b['actions'].numpy()),atol=1e-8)
    final=load(folder/'final.pt');assert equal(final,load(folder/f'model_{updates:02d}.pt'))
    for role in ['actor_state_dict','critic_state_dict']:
        assert all(torch.isfinite(x).all() for x in final[role].values())
        assert any(not torch.equal(x,final[role][k]) for k,x in initial[role].items())
    return dict(passed=True,CPU_replay=True,weight_atol=5e-5,weight_rtol=5e-4,actor_steps=sum(x['optimizer']['actor_steps'] for x in audit),critic_steps=8*updates)

def matched(reference,folder,start=9.):
    result=starts(reference,folder);assert result['exact']
    a=np.load(reference/'traction_startup.npz');b=np.load(folder/'traction_startup.npz')
    for k in a.files:assert np.array_equal(a[k],b[k]),(folder,k)
    a=np.load(reference/'trace.npz');b=np.load(folder/'trace.npz');mask=a['t']<=start
    for k in ['q','body','command','reference','torque','geometry']:assert np.array_equal(a[k][mask],b[k][:200][mask]),(folder,k,'prefix')
    return dict(**result,actual414_startup_exact=True,prefix_exact_until=start)

def actors(folder,cp=None):
    c=read(folder/'early_contract.json');m=read(folder/'manifest.json');initial=load(folder/'initial.pt')
    cal=read(DATA/'preflight/calibrate/early_calibration.json');norm=cal['stages'][str(c['start_s'])]
    assert m['parent_checkpoint_sha256']==sha(checkpoint(59117)) and equal(load(checkpoint(59117)),load(folder/'loaded_parent.pt'))
    if cp:
        assert sha(folder/'loaded_early.pt')==sha(cp) and m['early_checkpoint_sha256']==sha(cp)
        saved=load(cp)
        for k in ['actor_state_dict','critic_state_dict','actor_optimizer','critic_optimizer']:assert equal(saved[k],initial[k])
        assert saved['updates']==initial['updates']
    assert torch.equal(initial['input_mean'],torch.tensor(norm['mean'],dtype=torch.float32))
    assert torch.equal(initial['input_scale'],torch.tensor(norm['scale'],dtype=torch.float32))
    d=np.load(folder/'episode_decisions.npz');applied=np.load(folder/'applied_offsets.npz')['offset'];n=d['early_observation'].shape[1]
    assert d['early_observation'].shape[-1]==d['parent_observation'].shape[-1]==414
    assert c['early_span']==SPAN and c['ramp_seconds']==RAMP and c['parent_span']==.008 and c['handover_s']==11.
    model=PosturePPO(np.zeros(414),np.ones(414),0);model.load(folder/'initial.pt')
    parent=PosturePPO(np.zeros(414),np.ones(414),0);parent.load(checkpoint(59117))
    for i,(obs,action,po,pa) in enumerate(zip(d['early_observation'],d['early_action'],d['parent_observation'],d['parent_action'])):
        with torch.no_grad():
            assert np.allclose(parent.choose(torch.tensor(po),False)[0].numpy(),pa,atol=1e-5,rtol=1e-5)
            if c['mode']=='evaluate':assert np.allclose(model.choose(torch.tensor(obs),False)[0].numpy(),action,atol=1e-5,rtol=1e-5)
        expected=np.zeros((200,n,3))
        for step in range(int(c['start_s']*10),110):
            u=np.clip((step/10-c['start_s'])/RAMP,0,1);expected[step]=SPAN*np.tanh(action)*u*u*(3-2*u)
        expected[110:]=.008*np.tanh(pa)
        assert np.allclose(applied[i*200:(i+1)*200],expected,atol=3e-9,rtol=1e-6)
    return dict(parent_exact=True,checkpoint_exact=True,smooth_bounded_offsets_verified=True,actual414_actions_verified=True)

def gate(folder):
    outcome=physical(folder,require_clean=False);rows=read(folder/'results.json')['episodes']
    base=read(DATA/'assessment/59117/results.json')['episodes'];ref=read(DATA/'assessment/reference/results.json')['episodes']
    effort={};ok=outcome['clean']==len(rows) and outcome['max_friction_utilization']<1.
    for window,tol in [('metrics_whole_trial',1.001),('metrics_last5s',1.01)]:
        v=float(np.mean([x[window]['total_squared_torque_Nm2'] for x in rows]));b=float(np.mean([x[window]['total_squared_torque_Nm2'] for x in base]));r=float(np.mean([x[window]['total_squared_torque_Nm2'] for x in ref]))
        effort[window]=dict(policy=v,parent=b,reference=r,within_parent_tolerance=v<=b*tol,below_reference=v<r)
        ok=ok and v<=b*tol and v<r
    return dict(passed=bool(ok),physical=outcome,effort=effort,
        conservative_warnings='Unchanged .90 flags reported and penalized; advancement requires no slipping/contact and utilization below1, not zero .90 warnings.')
