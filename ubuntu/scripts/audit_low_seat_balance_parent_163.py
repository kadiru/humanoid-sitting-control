"""New balance/parent composition checks; synthetic command arithmetic only."""
from pathlib import Path
import datetime,hashlib,json
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from humanoid_sitting.shared_frame import BodyModel
from posture_episode_ppo import PosturePPO
from low_seat_gated_command_128 import GatedCommandPath
from low_seat_balance_feedback_160 import BalanceFeedback
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');O=D/'balance_parent_audit_163'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 O.mkdir(exist_ok=False);torch.set_num_threads(2);m=BodyModel(D/'cooked_model_50/model.json')
 base=D/'normal_landing3s_06/candidate';manifest=read(base/'manifest.json');checkpoint=Path(manifest['args']['checkpoint']);assert sha(checkpoint)==manifest['parent_checkpoint_sha256']
 fixture=dict(np.load(base/'sensor_inputs.npz'));parent=PosturePPO(np.zeros(414),np.ones(414),0,'cuda:0');parent.load(checkpoint)
 with torch.no_grad():action=parent.choose(torch.tensor(fixture['parent_observation'][0],device='cuda:0'),False)[0]
 assert np.array_equal(action.cpu().numpy(),fixture['parent_action'][0]);offset=(.008*torch.tanh(action)).cpu().numpy()
 six=[m.names.index(s+'_'+j+'_joint') for j in ['hip_pitch','knee','ankle_pitch'] for s in ['left','right']]
 limits={j['name']:np.deg2rad(j['limits']) for j in m.joints if j['type']=='PhysicsRevoluteJoint'}
 lo=torch.tensor([limits[n][0]+.03 for n in m.names[:23]],dtype=torch.float32);hi=torch.tensor([limits[n][1]-.03 for n in m.names[:23]],dtype=torch.float32);tests=[]
 for case in [0,1,6,7]:
  plan=read(D/f'balanced_force_paths_144/plan_{case}.json')['plan'];Q=np.array([r['q'] for r in plan['path']]);angles=Rotation.from_quat(np.array([r['root'][3:] for r in plan['path']])).as_euler('xyz')
  for sign in [-1,1]:
   e=GatedCommandPath(plan,m);e.cap=np.float32(.00225);f=BalanceFeedback(m,plan);targets=[];previous=f.pitch[0]
   for tick in range(1000):
    nominal=e.advance(True,.1);q=np.array([np.interp(e.position,e.arc,Q[:,j]) for j in range(37)]);abc=np.array([np.interp(e.position,e.arc,angles[:,j]) for j in range(3)])
    rate=(abc[1]-previous)/.02;previous=abc[1];abc[1]+=sign*.1
    g=-Rotation.from_euler('xyz',abc).as_matrix().T[:,2]
    target,r=f.update(q,g,np.array([0.,rate,0.]),e.position,nominal,np.array([[0.,50.,0.],[0.,50.,0.]]),[20.,20.],100.)
    targets.append(target)
   arrays=[]
   for device in ['cpu','cuda:0']:
    cmd=torch.tensor(e.Q[0],device=device,dtype=torch.float32);rows=[]
    for tick,t in enumerate(targets):
     ref=torch.tensor(t,device=device);off=torch.zeros(37,device=device)
     if tick>=700:
      for j in range(3):off[six[2*j:2*j+2]]=float(offset[case,j])
     desired=ref[:23]+off[:23];assert torch.all((desired>=lo.to(device))&(desired<=hi.to(device)))
     cmd[:23]+=(.1*(desired.clamp(min=lo.to(device),max=hi.to(device))-cmd[:23])).clamp(-.003,.003);rows.append(cmd.cpu().numpy().copy())
    arrays.append(np.array(rows))
   assert np.array_equal(*arrays);a=arrays[0];b=np.vstack([e.Q[0].astype(np.float32),a[:-1]])
   assert np.all(a<=b+np.float32(.003)) and np.all(a>=b-np.float32(.003));assert np.array_equal(a[:,23:],np.broadcast_to(e.Q[0,23:].astype(np.float32),a[:,23:].shape))
   expected=targets[-1].copy()
   for j in range(3):expected[six[2*j:2*j+2]]+=offset[case,j]
   err=float(abs(a[-1]-expected).max());assert err<2e-6
   tests.append(dict(case=case,synthetic_pitch_error_rad=sign*.1,command_ticks=1000,cpu_gpu_exact=True,parent_not_cancelled_max_error=err,max_step_rad=float(abs(a-b).max()),no_clipped_targets=True))
 out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,scope=__doc__,tests=tests,actual_parent_fixture_exact=True,parent_sha256=sha(checkpoint),physics_ticks=0,optimizer_updates=0,physical_admission=False,source_sha256={str(p.resolve()):sha(p) for p in [Path(__file__),Path(__file__).with_name('low_seat_balance_feedback_160.py'),checkpoint,base/'sensor_inputs.npz',D/'balanced_force_paths_144/SHA256.json']})
 (O/'independent_audit.json').write_text(json.dumps(out,indent=2));(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2));print(json.dumps(out),flush=True)
if __name__=='__main__':main()
