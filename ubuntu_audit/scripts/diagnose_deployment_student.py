"""Offline teacher-forced versus autoregressive previous-command diagnostic."""
import json,argparse
from pathlib import Path
import numpy as np,torch
from humanoid_sitting.deployment import DeploymentStudent,SensorPacket,pack,seat_vector,BODY_JOINT_NAMES
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--fixture',required=True);p.add_argument('--perception',required=True);p.add_argument('--output',required=True);a=p.parse_args();torch.set_num_threads(8)
r=json.loads((Path(a.fixture)/'results.json').read_text());z=np.load(Path(a.fixture)/'rollouts.npz');est=json.loads((Path(a.perception)/'estimates.json').read_text());ids=[r['joint_names'].index(n) for n in BODY_JOINT_NAMES];q=z['pre_joint_pos'][:,:12,ids];dq=z['pre_joint_vel'][:,:12,ids];y=z['targets'][:,:12,ids];tau=np.concatenate((np.zeros_like(z['joint_torque'][:1,:12]),z['joint_torque'][:-1,:12]),0)[:,:,ids];filtered=np.zeros_like(tau)
for t in range(len(tau)):filtered[t]=(.8*filtered[t-1] if t else 0)+.2*tau[t]
previous=np.concatenate((q[:1],y[:-1]));actor=z['teacher_actor_observations'][:,:12];fields=[q,dq,filtered,actor[:,:,3:6],actor[:,:,6:9]];fields=[np.concatenate((v[:1],v[:-1]),0) for v in fields];g=np.broadcast_to(np.array([seat_vector(e) for e in est[:12]],np.float32),(len(q),12,10)).copy();x=pack(SensorPacket(*[torch.tensor(v,device='cuda') for v in fields+[previous,g,np.ones((len(q),12,1),np.float32)]]));m,_=DeploymentStudent.load(a.checkpoint,'cuda');result={}
with torch.inference_mode():
 for mode in ('teacher_forced','autoregressive_previous'):
  state=None;prev=x[0,:,75:98].clone();preds=[]
  for t in range(len(x)):
   packet=x[t].clone()
   if mode=='autoregressive_previous':packet[:,75:98]=prev
   pred,state=m(packet[:,None,:],state);prev=pred[:,0];preds.append(prev.cpu().numpy())
  pred=np.stack(preds);err=pred-y;result[mode]=dict(rmse_rad=float(np.sqrt(np.mean(err**2))),first_second_max_error_rad=float(abs(err[:50]).max()),by_second_max_error_rad=[float(abs(err[s*50:(s+1)*50]).max()) for s in range(20)])
Path(a.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
