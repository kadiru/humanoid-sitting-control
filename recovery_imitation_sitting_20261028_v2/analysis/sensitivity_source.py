import sys,json,pathlib,numpy as np,torch
torch.set_num_threads(1)
root=pathlib.Path('/home/kadir/humanoid-sitting')
src=root/'experiments/recovery_imitation_sitting_20261028_v2';sys.path.insert(0,str(src))
from policy import Learner
bank=root/'outputs/full_body_contact_imitation_v4/dataset/sitting/bank.npz'
ck=root/'outputs/recovery_imitation_sitting_20261028_v2/clone_64111_2/initial.pt'
agent=Learner(np.zeros(336),np.ones(336),64111);agent.load(ck)
with np.load(bank,allow_pickle=False) as z:
 ix=np.flatnonzero(z['admitted']);times=[0,100,250,400,500,750,999]
 obs=z['obs'][times][:,ix].reshape(-1,336).copy()
x=torch.from_numpy(obs).requires_grad_(True)
pred=agent.actor(agent.normalize(x))
grads=[]
for j in range(23):
 grad=torch.autograd.grad(pred[:,j].sum(),x,retain_graph=j<22)[0]
 grads.append(grad)
jac=torch.stack(grads,1).detach().numpy()
Jp=sum(jac[:,:,offset+75:offset+98] for offset in [0,111,222])
Jq=sum(jac[:,:,offset:offset+23] for offset in [0,111,222])
Ap=.9*np.eye(23)[None]+.1*Jp
Ac=.9*np.eye(23)[None]+.1*(Jp+Jq)
def stats(a):
 rad=abs(np.linalg.eigvals(a)).max(-1)
 return dict(spectral_radius_min=float(rad.min()),median=float(np.median(rad)),max=float(rad.max()),fraction_above_one=float((rad>1).mean()))
print(json.dumps(dict(sample_count=len(obs),ticks=times,prior_command_only=stats(Ap),synchronous_position_and_previous_command=stats(Ac),scope='Local teacher-state observation sensitivity, not a full plant stability proof or causal intervention. q and command histories perturbed coherently; other channels held fixed. Filter linearization valid only when unsaturated.',checkpoint=str(ck))))

