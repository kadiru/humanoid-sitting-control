"""Privileged-label supervision; strictly physical-equivalent input packets."""
import argparse,json
from pathlib import Path
import numpy as np,torch
from humanoid_sitting.soft_landing import ContactEstimator
p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--output',required=True);p.add_argument('--steps',type=int,default=800);a=p.parse_args()
torch.set_num_threads(8);torch.manual_seed(541)
out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
r=json.loads((Path(a.dataset)/'results.json').read_text());assert r['geometry_seed'] not in (24001,25001,26001)
z=np.load(Path(a.dataset)/'rollouts.npz');x=z['deployment_inputs'].transpose(1,0,2)
# Labels at packet acquisition time t, never future post-action forces.
load=np.concatenate((np.zeros_like(z['support'][:1]),z['support'][:-1])).T
label=np.stack((load>.05,np.clip(load,0,1)),axis=-1).astype(np.float32)
# Every fourth geometry is withheld from estimator fitting.
train=np.array([o['geometry_index']%4!=3 for o in r['outcomes']]);test=~train
X=torch.tensor(x,device='cuda');Y=torch.tensor(label,device='cuda')
m=ContactEstimator().cuda();m.mean.copy_(X[train].mean((0,1)));m.std.copy_(X[train].std((0,1)).clamp_min(.05))
opt=torch.optim.Adam(m.parameters(),lr=.001);history=[]
for step in range(a.steps):
 ids=torch.tensor(np.random.default_rng(541+step).choice(np.where(train)[0],8),device='cuda')
 pred,_=m(X[ids]);w=1+4*((Y[ids,:,1]>.02)&(Y[ids,:,1]<.7)).float()
 cls=torch.nn.functional.binary_cross_entropy(pred[...,0],Y[ids,:,0],reduction='none')
 loss=((cls+3*(pred[...,1]-Y[ids,:,1]).square())*w).mean()
 opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);opt.step()
 if step%100==0:history.append({'step':step,'loss':loss.item()});print(history[-1],flush=True)
with torch.no_grad():pred,_=m(X[test]);pred=pred.cpu().numpy();truth=label[test]
metrics={'false_positive_fraction':float(((pred[...,0]>.65)&(truth[...,0]==0)).sum()/max(1,(truth[...,0]==0).sum())), 'false_negative_fraction':float(((pred[...,0]<=.65)&(truth[...,0]==1)).sum()/max(1,(truth[...,0]==1).sum())), 'load_fraction_mae':float(abs(pred[...,1]-truth[...,1]).mean()),'heldout_trajectories':int(test.sum()),'training_trajectories':int(train.sum())}
torch.save({'state_dict':m.state_dict(),'arguments':vars(a),'label_timing':'support at t from previous post-step; threshold .05 mg','actor_inputs':109},out/'contact.pt')
(out/'training.json').write_text(json.dumps({'history':history,'metrics':metrics,'arguments':vars(a)},indent=2));print(metrics)
