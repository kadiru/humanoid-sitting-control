"""No simulator or scientific launch. Deterministic algorithm/contract tests."""
import tempfile
from pathlib import Path
import numpy as np
import torch
from policy import command, advantages, Learner
from rewards import staged_reward, phase_from_clock
from audit import reward_numpy, weights, physical
from common import STEP, write


def run():
    torch.set_num_threads(1); torch.manual_seed(87131); tests=[]
    prev=torch.zeros(5,23); lo=torch.full_like(prev,-1.); hi=-lo
    for j in range(23):
        action=torch.zeros_like(prev); action[:,j]=1
        delta=command(prev,action,lo,hi)-prev
        assert (delta[:,j]>0).all() and torch.count_nonzero(delta)==5
    pos=prev.clone()
    for _ in range(1000):
        new=command(pos,torch.full_like(pos,10),lo,hi)
        assert (abs(new-pos)<=STEP+1e-7).all() and (new<=.97).all()
        pos=new
    assert pos.min()>.9; tests.append('23 independent channels; integrated motion exceeds old residual box; rate and position bounds')
    reward=torch.tensor([[1.],[100.],[9.]])
    adv,_=advantages(reward,torch.zeros_like(reward),torch.ones_like(reward)*7,torch.tensor([[True],[True],[True]]))
    assert torch.equal(adv,reward); tests.append('physical/horizon termination cuts both bootstrap and GAE')
    n=8; one=torch.ones(n); zero=torch.zeros(n)
    x=dict(alive=one.bool(),legal=one.bool(),no_slip=one.bool(),good=one.bool(),phase=one.long()*3,
           q_error=zero,speed=zero,gyro=zero,loaded_slip=zero,wrong_force=zero,peak_bw=one,
           normalized_command_delta=torch.zeros(n,23),support_bw=one*.2,held_ticks=one*250,
           torque_squared=one*100,clean_completion=zero.bool(),stable_completion=zero.bool(),first_failure=zero.bool())
    actual,terms=staged_reward(x); expected=reward_numpy({k:v.numpy() for k,v in x.items()})
    np.testing.assert_allclose(actual.numpy(),expected,rtol=1e-6)
    for key in ['legal','good','alive']:
        bad=dict(x);bad[key]=zero.bool();_,t=staged_reward(bad)
        assert t['hold'].sum()==t['effort'].sum()==0
    bad=dict(x);bad['loaded_slip']=one*.04; worse=dict(bad);worse['loaded_slip']=one*.08
    assert (staged_reward(worse)[0]<staged_reward(bad)[0]).all()
    bad=dict(x);bad['support_bw']=zero;bad['good']=zero.bool();_,t=staged_reward(bad)
    assert t['contact'].sum()==t['hold'].sum()==t['effort'].sum()==0
    bad=dict(x);bad['held_ticks']=zero;assert staged_reward(bad)[1]['effort'].sum()==0
    assert torch.equal(phase_from_clock(torch.tensor([0.,5.,10.,15.]),torch.ones(4)*5,torch.ones(4)*10),torch.arange(4))
    # Conservative episode-return ordering, under unchanged clean constraints.
    clean_lower=420-20*(2+2+2+.02); failed_upper=20*(2+1+3+2+.2)+20
    assert clean_lower>failed_upper
    tests.append('independent reward replay; slip severity; legal hold/effort gates; phase schedule; conservative clean-return dominance')
    agent=Learner(np.zeros(336),np.ones(336),87131)
    obs=torch.randn(128,336); action,logp,val,mu=agent.choose(obs)
    with tempfile.TemporaryDirectory() as path:
        path=Path(path);agent.save(path/'initial.pt')
        metrics=agent.update(obs,action,logp,mu,action[:,0]+obs[:,1],torch.randn(128))
        agent.save(path/'first_update.pt'); result=weights(path)
        assert result['passed'] and metrics['actor_steps']>0
    # Force KL rejection and ensure both actor and Adam return to the snapshot.
    before={k:v.clone() for k,v in agent.actor.state_dict().items()}
    import copy
    opt=copy.deepcopy(agent.actor_opt.state_dict()); agent.actor_opt.param_groups[0]['lr']=1e5
    action,logp,val,mu=agent.choose(obs)
    metrics=agent.update(obs,action,logp,mu,action[:,1],torch.randn(128))
    assert metrics['actor_steps']==0 and metrics['rejected_kl_steps']==1
    assert all(torch.equal(v,agent.actor.state_dict()[k]) for k,v in before.items())
    assert all(torch.equal(v['exp_avg'],agent.actor_opt.state_dict()['state'][k]['exp_avg']) for k,v in opt['state'].items())
    tests.append('PPO actor/critic and Adam update; rejected KL restores optimizer and weights')
    # A single early slip must fail clean completion even if the final hold
    # recovers. A forbidden contact must remain latched through recovery.
    shape=(1000,2); force=np.zeros((*shape,7,3),np.float32);force[:,:,0,2]=20;force[:,:,5,2]=80
    normal=np.zeros_like(force);normal[:,:,:,2]=1
    m=dict(force_w=force,normal_w=normal,friction=np.full((*shape,7),.6,np.float32),
           permitted=np.ones((*shape,7),bool),inside_patch=np.ones((*shape,7),bool),
           rest_patch=np.broadcast_to(np.array([1,1,1,1,1,0,0],bool),(*shape,7)),
           bodyweight_N=np.full(shape,100,np.float32),slip_m_s=np.zeros((*shape,7),np.float32),
           torque=np.zeros((*shape,37),np.float32),torque_scale=np.ones((*shape,37),np.float32),
           joint_margin_ok=np.ones(shape,bool),trunk_tilt_rad=np.zeros(shape,np.float32),
           speed_m_s=np.zeros(shape,np.float32),angular_speed_rad_s=np.zeros(shape,np.float32))
    d={'physical_'+k:v for k,v in m.items()};d.update(root_height=np.full(shape,.7),wrong_force=np.zeros(shape),peak_bw=np.zeros(shape))
    assert physical(d)['clean'].all()
    m['slip_m_s'][10,1,0]=.0201;r=physical(d)
    assert r['stable'].all() and r['clean'].tolist()==[True,False]
    d['wrong_force'][5,0]=10.1;r=physical(d);assert not r['alive'][5:,0].any() and not r['stable'][0]
    tests.append('independent full-trial criteria: early slip fails clean; forbidden contact stays latched')
    from launch_after_comparison import dependency_ready
    final={'results':[dict(arm=arm,seed=seed,updates=128) for arm in ['original','revised'] for seed in [63111,63117]]}
    assert not dependency_ready(dict(state='running',complete=False),final)
    assert dependency_ready(dict(state='comparison_complete',complete=True),final)
    for status,report in [(dict(state='stopped'),final),(dict(state='comparison_complete',complete=True),dict(results=final['results'][:3]))]:
        try:dependency_ready(status,report)
        except (RuntimeError,AssertionError):pass
        else:raise AssertionError('Unsatisfied comparison dependency admitted')
    tests.append('handoff refuses incomplete or stopped comparison')
    return dict(passed=True,tests=tests,simulator_steps=0,training_dataset_used=False,
                clean_episode_reward_lower_bound=clean_lower,failed_episode_reward_upper_bound=failed_upper)


if __name__=='__main__':
    import sys
    result=run();print(result)
    if len(sys.argv)>1:write(sys.argv[1],result)
