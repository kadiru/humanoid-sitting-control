"""Small sequence BC pilot using only the deployment sensor contract."""
import argparse,json,time
from pathlib import Path
import numpy as np,torch
from humanoid_sitting.deployment import DeploymentStudent,SensorPacket,pack,seat_vector,BODY_JOINT_NAMES
p=argparse.ArgumentParser();p.add_argument('--fixtures',nargs='*',default=[]);p.add_argument('--perception',nargs='*',default=[]);p.add_argument('--dagger',nargs='*',default=[]);p.add_argument('--ankle-only',action='store_true');p.add_argument('--contact-weight',type=float,default=1.);p.add_argument('--checkpoint');p.add_argument('--variant',choices=['full','geometry_recurrent'],default='full');p.add_argument('--output',required=True);p.add_argument('--steps',type=int,default=600);p.add_argument('--device',default='cuda:0');p.add_argument('--seed',type=int,default=401);a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False);torch.manual_seed(a.seed);torch.set_num_threads(8)
if len(a.fixtures)!=len(a.perception):p.error('Each fixture needs its own measured perception source.')
if not a.fixtures and not a.dagger:p.error('At least one demonstration source is required.')
X=[];Y=[];W=[];sources=[]
for fixture,camera in zip(a.fixtures,a.perception):
 root=Path(fixture);r=json.loads((root/'results.json').read_text());z=np.load(root/'rollouts.npz');est=json.loads((Path(camera)/'estimates.json').read_text());ids=[r['joint_names'].index(n) for n in BODY_JOINT_NAMES];T,N,_=z['pre_joint_pos'].shape
 assert r['geometry_seed'] not in (9612,10514,10515,24001,25001,26001),'Protected held-out banks cannot train the student'
 good=np.array([o['success'] and e['valid'] for o,e in zip(r['outcomes'],est)])
 q=z['pre_joint_pos'][:,:,ids];dq=z['pre_joint_vel'][:,:,ids];tau=np.concatenate((np.zeros_like(z['joint_torque'][:1]),z['joint_torque'][:-1]),axis=0)[:,:,ids];targets=z['targets'][:,:,ids];previous=np.concatenate((q[:1],targets[:-1]),axis=0)
 actor=z['teacher_actor_observations']
 filtered=np.zeros_like(tau)
 for t in range(T):filtered[t]=(.8*filtered[t-1] if t else 0)+.2*tau[t]
 tau=filtered
 gyro=actor[:,:,3:6];gravity=actor[:,:,6:9]
 q=np.concatenate((q[:1],q[:-1]));dq=np.concatenate((dq[:1],dq[:-1]));tau=np.concatenate((tau[:1],tau[:-1]));gyro=np.concatenate((gyro[:1],gyro[:-1]));gravity=np.concatenate((gravity[:1],gravity[:-1]))
 geom=np.broadcast_to(np.array([seat_vector(e) for e in est],np.float32)[None],(T,N,10)).copy()
 # Static initial geometry estimate, latched at command start; no episode or sample-age timer in actor.
 age=np.zeros((T,N,1),np.float32);motor_age=age.copy();command=np.ones_like(age)
 x=pack(SensorPacket(*[torch.tensor(v) for v in (q,dq,tau,gyro,gravity,previous,geom,command)])).numpy()
 X.append(x[:,good].transpose(1,0,2));Y.append(targets[:,good].transpose(1,0,2));W.append(np.ones((good.sum(),T,1),np.float32));sources.append(dict(fixture=fixture,perception=camera,trajectories=int(good.sum()),geometry_seed=r['geometry_seed']))
for folder in a.dagger:
 root=Path(folder);r=json.loads((root/'results.json').read_text());z=np.load(root/'rollouts.npz');ids=[r['joint_names'].index(n) for n in BODY_JOINT_NAMES]
 assert r['geometry_seed'] not in (9612,10514,10515,24001,25001,26001),'Protected held-out banks cannot train the student'
 assert z['deployment_inputs'].shape[-1]==109
 X.append(z['deployment_inputs'].transpose(1,0,2));Y.append(z['expert_targets'][:,:,ids].transpose(1,0,2));W.append(z['safe_training_state'].T[...,None].astype(np.float32)*3*(1+(a.contact_weight-1)*((z['support'].T[...,None]>.02)&(z['support'].T[...,None]<.8))))
 sources.append(dict(fixture=folder,kind='measured deployment packets at mixed-controller states',teacher_action_fraction=r['deployment_teacher_action_fraction'],safe_steps=int(z['safe_training_state'].sum())))
x=torch.tensor(np.concatenate(X),device=a.device);y=torch.tensor(np.concatenate(Y),device=a.device);weight=torch.tensor(np.concatenate(W),device=a.device)
if a.checkpoint:m,_=DeploymentStudent.load(a.checkpoint,a.device);m.train()
else:
 m=DeploymentStudent(variant=a.variant).to(a.device);m.mean.copy_(x.mean((0,1)));m.std.copy_(x.std((0,1)).clamp_min(.05));m.target_mean.copy_(y.mean((0,1)));m.target_std.copy_(y.std((0,1)).clamp_min(.05))
if a.ankle_only:
 assert m.variant=='geometry_recurrent'
 for name,param in m.named_parameters():param.requires_grad_(name.startswith('ankle_head.'))
opt=torch.optim.Adam([param for param in m.parameters() if param.requires_grad],lr=.0002 if a.checkpoint else .0004);history=[];start=time.time()
for step in range(a.steps):
 ids=torch.randint(len(x),(8,),device=a.device);inp=x[ids].clone()
 # Sensor-like perturbations; measured geometry stays the source, no GT replacement.
 inp[:,:,:23]+=torch.randn_like(inp[:,:,:23])*.001;inp[:,:,23:46]+=torch.randn_like(inp[:,:,23:46])*.01;inp[:,:,46:69]+=torch.randn_like(inp[:,:,46:69])*.25
 inp[:,:,69:75]+=torch.randn_like(inp[:,:,69:75])*.002
 inp[:,:,98:102]+=torch.randn_like(inp[:,:,98:102])*.002
 inp[:,:,75:98]+=torch.randn_like(inp[:,:,75:98])*.005
 pred,_=m(inp);w=weight[ids].clone();w[:,:100]*=5
 error=((pred-y[ids])/m.target_std).square();loss=(error*w).sum()/(w.sum()*23).clamp_min(1);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
 if step%100==0 or step==a.steps-1:history.append(dict(step=step,loss=loss.item(),elapsed_s=time.time()-start));print(history[-1],flush=True)
torch.save(dict(architecture=dict(inputs=109,outputs=23,hidden=128,variant=m.variant,mask_perception_status=m.mask_perception_status),state_dict=m.state_dict(),joint_names=BODY_JOINT_NAMES,sources=sources,actor_ground_truth_geometry=False,actor_contact_forces=False,actor_explicit_timer=False,actor_scripted_balance=False),out/'student.pt');(out/'training.json').write_text(json.dumps(dict(arguments=vars(a),history=history,sources=sources,trajectories=len(x),transitions=len(x)*1000),indent=2)+'\n')
