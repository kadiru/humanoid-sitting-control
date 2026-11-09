"""Independent action, reward, physics-budget and optimizer audit."""
import argparse
from pathlib import Path
import numpy as np
import torch
from feedback_posture_ppo import FeedbackPPO
from posture_episode_ppo import PosturePPO
from positive_slope_common_v5 import read,write,sha
from positive_slope_validation_v4 import physical
from verify_seat_load_study import equal,load

def validate(folder,replay=True):
    folder=Path(folder);c=read(folder/'feedback_contract.json');m=read(folder/'manifest.json');rows=read(folder/'results.json')['episodes']
    a=np.load(folder/'applied_offsets.npz')['offsets'];s=np.load(folder/'sensor_inputs.npz');batches=load(folder/'decision_batches.pt');audit=read(folder/'learning_audit.json')
    count,n=s['sensors'].shape[:1][0],s['sensors'].shape[2]
    assert s['sensors'].shape==(count,200,n,414) and a.shape==(count*200,3,n,3)
    assert len(rows)==count*n and c['inputs']==417 and c['actions']==3 and c['decision_period_s']==.5
    assert c['active']==count*n*1000 and c['warmup']==(count-1)*n*25
    assert c['total']==c['active']+c['warmup'] and c['initial_settling_excluded']==25*n
    assert c['feedback_window_ticks']+c['teacher_prefix_ticks']==c['active']
    assert len(batches)==len(audit)==count
    parent=PosturePPO(np.zeros(414),np.ones(414),0);parent.load(folder/'loaded_parent.pt')
    parent_path=m['args']['checkpoint'];assert sha(parent_path)==c['parent_sha256']==m['parent_checkpoint_sha256']
    assert equal(load(parent_path),load(folder/'loaded_parent.pt'))
    model=FeedbackPPO(np.zeros(417),np.ones(417),0);model.load(folder/'initial.pt')
    initial=load(folder/'initial.pt');ticks=np.load(folder/'every_tick.npz')
    assert len(ticks['t'])==count*1000+(count-1)*25
    if m['feedback_checkpoint_sha256']:
        assert sha(folder/'loaded_feedback.pt')==m['feedback_checkpoint_sha256']
        assert equal(initial,load(folder/'loaded_feedback.pt'))
    tolerance=dict(action_atol=1e-5,action_rtol=1e-5,weight_atol=5e-5,weight_rtol=5e-4)
    actor_steps=critic_steps=0
    for episode,b in enumerate(batches):
        applied=a[episode*200:(episode+1)*200];steps=b['decision_steps'];previous=np.zeros((n,3),np.float32);target=previous.copy();parent_offset=previous.copy();idx=0
        assert len(steps)*n==audit[episode]['decisions']
        for step in range(200):
            sensor=s['sensors'][episode,step]
            if step in steps:
                expected=np.concatenate([sensor,previous/c['span_rad']],-1)
                assert np.allclose(expected,b['obs'][idx].numpy(),atol=1e-7,rtol=1e-6),(episode,step,'actual417')
                action=b['actions'][idx].numpy()
                if c['mode']=='evaluate':
                    with torch.no_grad():assert np.allclose(model.choose(b['obs'][idx],False)[0].numpy(),action,atol=1e-5,rtol=1e-5)
                if c['mode'] in ['zero','reference','calibrate']:assert not action.any()
                target=c['span_rad']*np.tanh(action);idx+=1
            if step in [110,150]:
                due=s['handover'][episode]==step
                assert np.array_equal(sensor[due],s['parent_observation'][episode,due])
                with torch.no_grad():pred=parent.choose(torch.tensor(sensor),False)[0].numpy()
                assert np.allclose(pred[due],s['parent_action'][episode,due],atol=1e-5,rtol=1e-5)
                if c['parent_enabled']:parent_offset[due]=.008*np.tanh(s['parent_action'][episode,due])
            if step>=round(10*c['start_s']):previous=np.clip(previous+np.clip(target-previous,-c['slew_rad_s']*.1,c['slew_rad_s']*.1),-c['span_rad'],c['span_rad'])
            assert np.allclose(applied[step,0],previous,atol=3e-9,rtol=1e-6)
            assert np.allclose(applied[step,1],parent_offset,atol=3e-9,rtol=1e-6)
            assert np.allclose(applied[step,2],previous+parent_offset,atol=3e-9,rtol=1e-6)
        offset=episode*1025;cost=ticks['total_squared_torque_Nm2'][offset:offset+1000].reshape(200,5,n).sum(1)*.02
        assert np.array_equal(cost,b['torque_integrals'].numpy()),'Every physical tick in reward'
        # Independent suffix sums rather than calling the training return function.
        expected=np.array([-(cost[step:]-b['baseline_integrals'].numpy()[step:]).sum(0)/1350-b['terminal_penalty'].numpy() for step in steps])
        assert np.allclose(expected,b['reward'].numpy(),atol=2e-5,rtol=1e-6),'Complete reward-to-go'
        for i,row in enumerate(rows[episode*n:(episode+1)*n]):
            assert np.isclose(cost[:,i].sum()/20,row['metrics_whole_trial']['total_squared_torque_Nm2'],rtol=3e-6,atol=.001)
            assert int((~ticks['no_slip'][offset:offset+1000,i]).sum())==row['slip_ticks']
            assert int((~ticks['legal'][offset:offset+1000,i]).sum())==row['illegal_ticks']
        if c['mode']=='train' and replay:
            with torch.no_grad():
                dist=model.distribution(b['obs'].flatten(0,1))
                assert torch.allclose(dist.mean,b['old_mean'].flatten(0,1),atol=2e-5,rtol=2e-5)
                assert torch.allclose(dist.log_prob(b['actions'].flatten(0,1)).sum(-1),b['logp'].flatten(),atol=1e-4,rtol=2e-5)
            stats=model.update(b['obs'].flatten(0,1),b['actions'].flatten(0,1),b['logp'].flatten(),b['old_mean'].flatten(0,1),b['reward'].flatten())
            assert stats['actor_steps']==audit[episode]['optimizer']['actor_steps'] and stats['critic_steps']==8
            actor_steps+=stats['actor_steps'];critic_steps+=8;saved=load(folder/f'model_{episode+1:02d}.pt')
            for role,network in [('actor_state_dict',model.actor),('critic_state_dict',model.critic)]:
                for key,value in network.state_dict().items():assert torch.allclose(value,saved[role][key],atol=5e-5,rtol=5e-4),(episode,role,key)
            for role,steps_taken in [('actor_optimizer',actor_steps),('critic_optimizer',critic_steps)]:
                old=initial[role]['state'];old_steps=int(next(iter(old.values()))['step']) if old else 0
                assert all(int(v['step'])==old_steps+steps_taken for v in saved[role]['state'].values())
            assert saved['updates']==initial['updates']+episode+1
    final=load(folder/'final.pt')
    if c['mode']=='train':
        assert equal(final,load(folder/f'model_{count:02d}.pt'))
        for role in ['actor_state_dict','critic_state_dict']:
            assert all(torch.isfinite(v).all() for v in final[role].values())
            assert any(not torch.equal(v,final[role][k]) for k,v in initial[role].items())
    else:assert equal(initial,final)
    return dict(passed=True,physical=physical(folder,episodes=count,require_clean=False),actual417_inputs=True,full_reward_audited=True,bounds_and_slew=True,parent_exact=True,CPU_optimizer_replay=c['mode']=='train' and replay,actor_steps=actor_steps,critic_steps=critic_steps,numerical_tolerances=tolerance,active=c['active'],warmup=c['warmup'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--folder',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output);assert not out.exists();write(out,validate(a.folder));print(out.read_text())
