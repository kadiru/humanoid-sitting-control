"""Bounded behavior cloning of absolute teacher targets; geometry-disjoint validation."""
import argparse,json,time,hashlib,shutil
from pathlib import Path
import numpy as np
import torch
from humanoid_sitting.student_network import Student,StructuredStudent
p=argparse.ArgumentParser();p.add_argument('--train',nargs='+',required=True);p.add_argument('--validation',required=True);p.add_argument('--output',required=True);p.add_argument('--steps',type=int,default=6000);p.add_argument('--seed',type=int,default=101);p.add_argument('--device',default='cuda:0');p.add_argument('--checkpoint');p.add_argument('--structured',action='store_true');p.add_argument('--minimal-geometry',action='store_true');a=p.parse_args()
out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
assert shutil.disk_usage(out).free>30*2**30
np.random.seed(a.seed);torch.manual_seed(a.seed);torch.set_num_threads(8)
def load(path):
    path=Path(path);r=json.loads((path/'results.json').read_text());good=np.array([x['success'] for x in r['outcomes']]);z=np.load(path/'rollouts.npz')
    if 'expert_targets' in z.files:
        mask=z['safe_training_state'].astype(bool);x=z['student_observations'][mask];y=z['expert_targets'][mask]
    else:x=z['student_observations'][:,good].reshape(-1,99);y=z['targets'][:,good].reshape(-1,37)
    return torch.tensor(x,device=a.device),torch.tensor(y,device=a.device),r
train=[load(x) for x in a.train];x=torch.cat([v[0] for v in train]);y=torch.cat([v[1] for v in train]);vx,vy,vr=load(a.validation)
assert vr['geometry_seed'] not in [v[2]['geometry_seed'] for v in train]
assert not any(v[2]['geometry_seed'] in (9612,10514,10515,24001,25001) for v in train)
model=(StructuredStudent(minimal_geometry=a.minimal_geometry) if a.structured else Student()).to(a.device)
if a.checkpoint:model,_=Student.load(a.checkpoint,a.device);model.train()
else:
    if a.structured:
        # Fit the teacher's actual balance correction separately; no reference is
        # evaluated at student inference. Body-X approximates world-X in aligned cases.
        correction=(8*x[:,95]+1.5*x[:,0]).clamp(-.35,.35)*(x[:,96]<.25)
        base_y=y.clone();base_y[:,model.ankle_indices]-=correction[:,None]
        selected=(x[:,96]<.25)&(correction.abs()<.34)
        A=torch.stack((x[selected,95],x[selected,0]),dim=1)
        model.balance_gain.copy_(torch.linalg.lstsq(A,correction[selected,None]).solution[:,0])
        f=model.features(x);model.mean.copy_(f.mean(0));model.std.copy_(f.std(0).clamp_min(.01))
        model.target_mean.copy_(base_y.mean(0));model.target_std.copy_(base_y.std(0).clamp_min(.02))
    else:
        model.mean.copy_(x.mean(0));model.std.copy_(x.std(0).clamp_min(.01));model.target_mean.copy_(y.mean(0));model.target_std.copy_(y.std(0).clamp_min(.02))
opt=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=1e-6)
scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.steps,eta_min=2e-5)
history=[];start=time.time();best=float('inf')
for step in range(a.steps):
    ids=torch.randint(len(x),(2048,),device=a.device);prediction=model(x[ids]);loss=((prediction-y[ids])/model.target_std).square().mean()
    opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step();scheduler.step()
    if step%500==0 or step==a.steps-1:
        with torch.no_grad():
            ids=torch.arange(0,len(vx),5,device=a.device);error=model(vx[ids])-vy[ids];rmse=error.square().mean().sqrt().item();perjoint=error.square().mean(0).sqrt().cpu().tolist()
        history.append(dict(step=step,train_loss=loss.item(),validation_joint_rmse_rad=rmse,validation_per_joint_rmse_rad=perjoint,elapsed_s=time.time()-start))
        print(json.dumps(history[-1]),flush=True)
        if rmse<best:
            best=rmse;torch.save(dict(architecture=dict(inputs=99,outputs=37,width=256,**({'minimal_geometry':a.minimal_geometry} if a.structured else {})),state_dict=model.state_dict(),variant='structured' if a.structured else 'unstructured',seed=a.seed,step=step,training_geometry_seeds=[v[2]['geometry_seed'] for v in train],validation_geometry_seed=vr['geometry_seed'],absolute_targets=True,teacher_reference_in_action_path=False),out/'best.pt')
(out/'training.json').write_text(json.dumps(dict(arguments=vars(a),samples=len(x),validation_samples=len(vx),training_successful_trajectories=sum(sum(t['success'] for t in v[2]['outcomes']) for v in train),history=history,best_validation_rmse_rad=best,elapsed_s=time.time()-start),indent=2)+'\n')
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.plot([h['step'] for h in history],[h['validation_joint_rmse_rad'] for h in history]);plt.xlabel('Gradient updates');plt.ylabel('Held-out target RMSE (rad)');plt.yscale('log');plt.tight_layout();plt.savefig(out/'learning_curve.png');plt.close()
