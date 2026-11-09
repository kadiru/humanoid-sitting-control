"""Check credit assignment, bounded dynamics and a reproducible PPO update."""
import tempfile
from pathlib import Path
import numpy as np
import torch
from feedback_posture_ppo import FeedbackPPO,slew,complete_returns

def main():
    x=torch.zeros(2,3)
    assert torch.equal(slew(x,x),x)
    for _ in range(60):
        y=slew(x,torch.tensor([[99.,-99.,0.],[99.,-99.,0.]]))
        assert abs(y-x).max()<=.000400001 and abs(y).max()<=.004000001
        x=y
    assert torch.allclose(x[:,:2],torch.tensor([[.004,-.004],[.004,-.004]]))
    t=torch.zeros(200,2);b=torch.zeros_like(t);t[150,0]=1350.;t[199,1]=2700.
    r=complete_returns(t,b,torch.tensor([5.,0.]),[100,150,155,195])
    assert torch.equal(r,torch.tensor([[-6.,-2.],[-6.,-2.],[-5.,-2.],[-5.,-2.]]))
    assert torch.equal(complete_returns(t,t,torch.zeros(2),[100,195]),torch.zeros(2,2))
    m=FeedbackPPO(np.zeros(417),np.ones(417),62111)
    obs=torch.randn(12,417);a,lp,v,mu=m.choose(obs);reward=-a.square().sum(-1)
    with tempfile.TemporaryDirectory() as d:
        p=Path(d)/'initial.pt';m.save(p)
        twin=FeedbackPPO(np.zeros(417),np.ones(417),0);twin.load(p)
        stats=m.update(obs,a,lp,mu,reward);again=twin.update(obs,a,lp,mu,reward)
        assert stats==again and stats['actor_steps']>0 and stats['critic_steps']==8
        assert all(torch.equal(x,twin.actor.state_dict()[k]) for k,x in m.actor.state_dict().items())
        assert m.updates==1 and all(torch.isfinite(x).all() for x in m.actor.parameters())
    print('PASS: complete delayed credit, exact zero, bounded slew, finite PPO and exact CPU replay')

if __name__=='__main__':main()
