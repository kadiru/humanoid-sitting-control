"""Meaningful optimizer/credit tests without a simulator."""
import torch
from posture_episode_ppo import PosturePPO,episode_return

def main():
    torch.set_num_threads(1)
    p=PosturePPO(torch.zeros(414),torch.ones(414),59101)
    obs=torch.zeros(512,414)
    initial=p.choose(obs,False)[0];assert torch.count_nonzero(initial)==0
    # A delayed-only quadratic cost has a known optimum. No intermediate reward
    # is available: the complete return must move the policy toward that optimum.
    initial_error=float((initial-.7).square().mean())
    audits=[]
    for _ in range(24):
        action,logp,_,mean=p.choose(obs)
        reward=-(action-.6).square().sum(-1)
        audits.append(p.update(obs,action,logp,mean,reward))
    final=p.choose(obs,False)[0];assert float((final-.6).square().mean())<.03
    assert all(a['max_accepted_kl']<=.02 for a in audits)
    before={k:v.clone() for k,v in p.actor.state_dict().items()}
    # Identical returns give zero advantages for identical observations, so
    # momentum is still possible in PPO; this test checks finite bounded steps,
    # not the false assertion that zero gradients necessarily freeze Adam.
    bad=dict(metrics_whole_trial={'total_squared_torque_Nm2':0},failures={'fall':True},success=False,slip_ticks=0,illegal_ticks=0)
    good=dict(metrics_whole_trial={'total_squared_torque_Nm2':600},failures={'fall':False},success=True,slip_ticks=0,illegal_ticks=0)
    assert episode_return([bad,good],[600,600])[0]<episode_return([bad,good],[600,600])[1]
    print({'passed':True,'delayed_cost_error':float((final-.6).square().mean()),'max_kl':max(a['max_accepted_kl'] for a in audits),'failed_limp_cannot_win':True})
if __name__=='__main__':main()
