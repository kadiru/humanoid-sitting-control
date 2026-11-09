"""Behavioral checks for control preservation and the actual executed objective."""
import torch
from humanoid_sitting.effort_learning import coordinated_actions,resting_reward,mask_learning_advantages
from humanoid_sitting.two_surface_path import residual_span
from humanoid_sitting.deployment import BODY_JOINT_NAMES

torch.manual_seed(56101)
names=BODY_JOINT_NAMES;span=residual_span(names)
parent=torch.randn(12,23)*.4
child=torch.randn(12,2)
assert torch.equal(coordinated_actions(parent,torch.zeros_like(child),torch.ones(12),span,names),parent)
assert torch.equal(coordinated_actions(parent,child,torch.zeros(12),span,names),parent)
value=coordinated_actions(parent,child,torch.ones(12),span,names)
assert torch.isfinite(value).all()
unchanged=[i for i,n in enumerate(names) if not any(x in n for x in ('knee','hip_pitch','ankle_pitch'))]
# atan/tanh roundoff is bounded for unchanged channels on active rows.
assert torch.allclose(torch.tanh(value[:,unchanged]),torch.tanh(parent[:,unchanged]),atol=1e-7,rtol=0)
good=torch.tensor([True]);alive=good;active=good;no=torch.tensor([False])
def reward(e,g=good,a=alive,t=active,f=no):return resting_reward(torch.tensor([e]),torch.tensor([500.]),g,a,t,f).item()
assert reward(300.)>reward(500.)>reward(700.)
assert reward(50000.)>reward(0.,g=no) # Losing support never earns an effort saving.
assert reward(0.,a=no,g=no,f=good)<-4.9
assert reward(100.,t=no)==reward(1000.,t=no)==0
a=torch.tensor([[[1000.],[1000.]],[[2.],[4.]]]);mask=torch.tensor([[False,False],[True,True]])
out=mask_learning_advantages(a,mask)
assert torch.equal(out[0],torch.zeros_like(out[0])) and torch.equal(out[1],torch.tensor([[-1.],[1.]]))
assert torch.equal(mask_learning_advantages(a,torch.zeros_like(mask)),torch.zeros_like(a))
print('PASS: exact zero-child/approach preservation, bounded finite actions, torque preference, support/failure priority, and learning-mask isolation')
