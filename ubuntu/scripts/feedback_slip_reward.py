"""Dense evaluation-label costs for transfer learning; never network inputs."""
import numpy as np
import torch
from feedback_posture_ppo import complete_returns

def per_tick_cost(record):
    loaded=np.linalg.norm(record['force'],axis=-1)>.01*record['weight'][:,None]
    excess=np.where(loaded,np.maximum(record['slip']-.02,0.),0.).max(-1)
    wrong=np.maximum(record['wrong']-10.,0.)/record['weight']
    # Distances and forces remain continuous learning signals even when an
    # episode already has a binary slip/contact failure label. No clipping.
    return .02*(500.*excess+20.*wrong+5.*(~record['legal']))

def returns(torque_integrals,baseline_integrals,terminal_penalty,decision_steps,physical_cost):
    assert physical_cost.shape==torque_integrals.shape
    return complete_returns(torque_integrals,baseline_integrals,terminal_penalty,decision_steps)-physical_cost.flip(0).cumsum(0).flip(0)[decision_steps]

def test():
    record=dict(force=np.zeros((2,7,3)),weight=np.array([100.,100.]),slip=np.full((2,7),.1),wrong=np.zeros(2),legal=np.ones(2,dtype=bool))
    assert not per_tick_cost(record).any(),'Unloaded motion is not slip'
    record['force'][0,0,2]=5.;record['slip'][0,0]=.03
    assert np.allclose(per_tick_cost(record),[.1,0.])
    record['wrong'][1]=20.;record['legal'][1]=False
    assert np.allclose(per_tick_cost(record),[.1,.14])
    t=torch.zeros(200,2);d=t.clone();d[199,0]=1.;p=torch.zeros(2)
    assert torch.equal(returns(t,t,p,[100,190],d),torch.tensor([[-1.,0.],[-1.,0.]]))
    print('PASS dense loaded-slip/forbidden-contact costs and complete delayed credit')

if __name__=='__main__':test()
