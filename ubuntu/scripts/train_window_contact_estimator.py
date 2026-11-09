"""Finite-window contact estimator: no episode-length recurrent timing signal."""
import json,argparse
from pathlib import Path
import numpy as np,torch
from humanoid_sitting.soft_landing import ContactEstimator
p=argparse.ArgumentParser();p.add_argument('--dataset',nargs='+',required=True);p.add_argument('--output',required=True);p.add_argument('--steps',type=int,default=1800);p.add_argument('--negative-weight',type=float,default=1.);a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False);torch.manual_seed(892);torch.set_num_threads(8)
Xs=[];Ys=[];training_indices=[];testing_indices=[];offset=0
for folder in a.dataset:
 r=json.loads((Path(folder)/'results.json').read_text());assert r['geometry_seed'] not in (24001,25001,26001)
 z=np.load(Path(folder)/'rollouts.npz');xs=z['deployment_inputs'].transpose(1,0,2);load=np.concatenate((np.zeros_like(z['support'][:1]),z['support'][:-1])).T
 Xs.append(xs);Ys.append(np.stack((load>.05,np.clip(load,0,1)),axis=-1).astype(np.float32))
 training_indices.extend(offset+i for i,o in enumerate(r['outcomes']) if o['geometry_index']%4!=3);testing_indices.extend(offset+i for i,o in enumerate(r['outcomes']) if o['geometry_index']%4==3);offset+=len(r['outcomes'])
X=torch.tensor(np.concatenate(Xs),device='cuda');Y=torch.tensor(np.concatenate(Ys),device='cuda');train=torch.tensor(training_indices,device='cuda');test=torch.tensor(testing_indices,device='cuda')
m=ContactEstimator().cuda();m.mean.copy_(X[train].mean((0,1)));m.std.copy_(X[train].std((0,1)).clamp_min(.05));opt=torch.optim.Adam(m.parameters(),lr=.001);hist=[];window=32
for step in range(a.steps):
 ids=train[torch.randint(len(train),(64,),device='cuda')];ends=torch.randint(1,1001,(64,),device='cuda');ends[:8]=torch.randint(1,101,(8,),device='cuda')
 times=(ends[:,None]-window+torch.arange(window,device='cuda')[None]).clamp_min(0);x=X[ids[:,None],times];y=Y[ids,ends-1];pred,_=m(x);pred=pred[:,-1];w=1+4*((y[:,1]>.02)&(y[:,1]<.8)).float()
 classification=torch.nn.functional.binary_cross_entropy(pred[:,0],y[:,0],reduction='none')*torch.where(y[:,0]>0,1.,a.negative_weight)
 loss=((classification+3*(pred[:,1]-y[:,1]).square())*w).mean();opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);opt.step()
 if step%300==0:hist.append({'step':step,'loss':loss.item()});print(hist[-1],flush=True)
with torch.inference_mode():
 preds=[]
 for t in range(1000):
  times=torch.arange(t-window+1,t+1,device='cuda').clamp_min(0);p,_=m(X[test][:,times]);preds.append(p[:,-1].cpu().numpy())
pred=np.stack(preds,axis=1);truth=Y[test].cpu().numpy();metrics={'false_positive_fraction':float(((pred[...,0]>.65)&(truth[...,0]==0)).sum()/max(1,(truth[...,0]==0).sum())),'false_negative_fraction':float(((pred[...,0]<=.65)&(truth[...,0]==1)).sum()/max(1,(truth[...,0]==1).sum())),'load_fraction_mae':float(abs(pred[...,1]-truth[...,1]).mean()),'heldout_trajectories':len(test)}
torch.save({'state_dict':m.state_dict(),'window':window,'arguments':vars(a),'labels':'previous post-step seat support at packet time t, .05mg contact threshold','inputs':'109 physical-equivalent channels; no simulator contacts or clock'},out/'contact.pt');(out/'training.json').write_text(json.dumps({'metrics':metrics,'history':hist,'window':window,'arguments':vars(a)},indent=2));print(metrics)
