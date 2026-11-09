"""Current-measurement settling fixtures and independent Torch command replay.
Synthetic poses are offline fixtures only. No environment or physics exists.
"""
from pathlib import Path
import json,datetime,hashlib
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from humanoid_sitting.shared_frame import BodyModel
from posture_episode_ppo import PosturePPO
from low_seat_touchdown_path_175 import TouchdownPath
from low_seat_bias_feedback_v2_207 import BiasFeedback as BalanceFeedback
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');O=D/'bias_control_audit_v3_209'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 O.mkdir(exist_ok=False);torch.set_num_threads(2);assert read(D/'touchdown_schedule_audit_174/independent_audit.json')['passed']
 m=BodyModel(D/'cooked_model_50/model.json');base=D/'normal_landing3s_06/candidate';mf=read(base/'manifest.json');cp=Path(mf['args']['checkpoint']);assert sha(cp)==mf['parent_checkpoint_sha256']
 fixture=dict(np.load(base/'sensor_inputs.npz'));parent=PosturePPO(np.zeros(414),np.ones(414),0,'cuda:0');parent.load(cp)
 with torch.no_grad():action=parent.choose(torch.tensor(fixture['parent_observation'][0],device='cuda:0'),False)[0]
 assert np.array_equal(action.cpu().numpy(),fixture['parent_action'][0]);offset=(.008*torch.tanh(action)).cpu().numpy()
 limits={j['name']:np.deg2rad(j['limits']) for j in m.joints if j['type']=='PhysicsRevoluteJoint'};lo=torch.tensor([limits[n][0]+.03 for n in m.names[:23]],dtype=torch.float32);hi=torch.tensor([limits[n][1]-.03 for n in m.names[:23]],dtype=torch.float32)
 six=[m.names.index(s+'_'+j+'_joint') for j in ['hip_pitch','knee','ankle_pitch'] for s in ['left','right']];rows=[]
 for c in [0,1,6,7]:
  plan=read(D/f'balanced_force_paths_144/plan_{c}.json')['plan'];Q=np.array([r['q'] for r in plan['path']]);abc=Rotation.from_quat(np.array([r['root'][3:] for r in plan['path']])).as_euler('xyz')
  for mode in ['nominal','late_settle','no_settle','no_load','stale_hover','high_raw_speed']:
   e=TouchdownPath(plan,m);f=BalanceFeedback(m,plan);targets=[];positions=[];quiet_count=0;hover_visits=0;released_tick=None
   for tick in range(1000):
    q=np.array([np.interp(e.position,e.arc,Q[:,j]) for j in range(37)]);angles=np.array([np.interp(e.position,e.arc,abc[:,j]) for j in range(3)]);geo=np.zeros(13);geo[11]=1;gyro=np.zeros(3);qd=np.zeros(37)
    at_hover=abs(e.position-e.hover)<1e-10
    if at_hover:hover_visits+=1
    bad=(mode=='no_settle' or mode=='late_settle' and at_hover and hover_visits<=50)
    if bad:angles[1]+=.1
    if mode=='stale_hover' and at_hover and 10<=hover_visits<35:geo[12]=.6
    if mode=='high_raw_speed':qd[e.ids]=.2
    g=-Rotation.from_euler('xyz',angles).as_matrix().T[:,2]
    e.observe_settling(tick,q,g,gyro,qd,geo);saved=dict(e.settle);e.observe_settling(tick,q,g+10,gyro,qd,geo);assert saved==e.settle
    independently_quiet=bool(geo[12]<=.5 and abs(np.arctan2(g[0],-g[2])-np.interp(e.position,e.arc,abc[:,1]))<=np.deg2rad(.5) and max(abs(qd[e.ids]))<=.05)
    assert e.settle['quiet']==independently_quiet
    old_released=e.released;before=e.position
    target=e.advance(geo[12]<=.5,0. if mode=='no_load' else .1)
    if at_hover and not old_released:
     if independently_quiet:quiet_count+=1
     else:quiet_count=0
    if e.released and not old_released:assert quiet_count>=26;released_tick=tick
    if not e.released:assert e.position<=e.hover
    if e.released:
     speed=(np.interp(before,e.arc,e.gaps)-np.interp(e.position,e.arc,e.gaps))/.02;assert speed<=.006+1e-10
    if mode=='no_load':assert e.position<=e.arc[80]
    target,feedback=f.update(q,g,gyro,e.position,target,np.array([[0.,50.,0.],[0.,50.,0.]]),[20.,20.],100.,geometry=geo,at_hover=bool(abs(e.position-e.hover)<1e-10 and not e.released))
    assert np.max(abs(np.asarray(feedback['offset_after'])))<=.06+1e-7
    targets.append(target);positions.append(e.position)
   positions=np.array(positions);done=np.flatnonzero(positions==e.arc[-1]);duration=float((done[0]+1)*.02) if len(done) else None
   if mode in ['nominal','late_settle','stale_hover']:assert duration is not None and duration<=13.5
   if mode in ['no_settle','high_raw_speed']:assert not e.released and duration is None
   if mode=='no_load':assert duration is None and e.position==e.arc[80]
   arrays=[]
   for device in ['cpu','cuda:0']:
    cmd=torch.tensor(e.Q[0],device=device,dtype=torch.float32);out=[]
    for tick,tar in enumerate(targets):
     desired=torch.tensor(tar[:23],device=device)
     if tick>=700:
      for j in range(3):desired[six[2*j:2*j+2]]+=float(offset[c,j])
     assert torch.all((desired>=lo.to(device))&(desired<=hi.to(device)))
     cmd[:23]+=(.1*(desired.clamp(min=lo.to(device),max=hi.to(device))-cmd[:23])).clamp(-.003,.003);out.append(cmd.cpu().numpy().copy())
    arrays.append(np.array(out))
   assert np.array_equal(*arrays);a=arrays[0];b=np.vstack([e.Q[0].astype(np.float32),a[:-1]]);assert np.all(a<=b+np.float32(.003)) and np.all(a>=b-np.float32(.003));assert np.array_equal(a[:,23:],np.broadcast_to(e.Q[0,23:].astype(np.float32),a[:,23:].shape))
   expected=targets[-1].copy()
   for j in range(3):expected[six[2*j:2*j+2]]+=offset[c,j]
   err=float(abs(a[-1]-expected).max());assert err<2e-6
   rows.append(dict(case=c,mode=mode,command_ticks=1000,duration_s=duration,released_tick=released_tick,cpu_gpu_exact=True,no_clipped_targets=True,parent_not_cancelled_max_error=err,max_step_rad=float(abs(a-b).max())));print(rows[-1],flush=True)
 paths=[Path(__file__),Path('scripts/low_seat_touchdown_path_175.py'),Path('scripts/low_seat_bias_feedback_v2_207.py'),D/'balanced_force_paths_144/SHA256.json',D/'touchdown_schedule_audit_174/SHA256.json',cp]
 out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,scope=__doc__,rows=rows,physics_ticks=0,optimizer_updates=0,physical_admission=False,current_guard_idempotent=True,stale_or_unsettled_hold=True,no_assumed_load_before_permission=True,actual_parent_fixture_exact=True,source_sha256={str(p.resolve()):sha(p) for p in paths})
 (O/'independent_audit.json').write_text(json.dumps(out,indent=2));(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2))
if __name__=='__main__':main()
