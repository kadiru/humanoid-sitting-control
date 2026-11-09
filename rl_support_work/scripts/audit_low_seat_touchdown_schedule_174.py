"""Independent vectorized interpolation, bounded-target and Torch filter audit173.

No schedule generator or feedback controller is imported. Synthetic settling
and loading remain assumptions; this verifies command arithmetic only.
"""
from pathlib import Path
import datetime,hashlib,json
import numpy as np
import torch
from humanoid_sitting.shared_frame import BodyModel
from posture_episode_ppo import PosturePPO
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');O=D/'touchdown_schedule_audit_174'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 O.mkdir(exist_ok=False);torch.set_num_threads(2);src=D/'touchdown_schedule_173';result=read(src/'results.json')
 for p,h in result['source_sha256'].items():assert sha(p)==h,p
 for p,h in read(src/'SHA256.json').items():assert sha(src/p)==h,p
 m=BodyModel(D/'cooked_model_50/model.json');mf=read(D/'normal_landing3s_06/candidate/manifest.json');cp=Path(mf['args']['checkpoint']);assert sha(cp)==mf['parent_checkpoint_sha256']
 parent=PosturePPO(np.zeros(414),np.ones(414),0,'cuda:0');parent.load(cp);fixture=dict(np.load(D/'normal_landing3s_06/candidate/sensor_inputs.npz'))
 with torch.no_grad():act=parent.choose(torch.tensor(fixture['parent_observation'][0],device='cuda:0'),False)[0]
 assert np.array_equal(act.cpu().numpy(),fixture['parent_action'][0]);off=(.008*torch.tanh(act)).cpu().numpy()
 limits={j['name']:np.deg2rad(j['limits']) for j in m.joints if j['type']=='PhysicsRevoluteJoint'};lo=torch.tensor([limits[n][0]+.03 for n in m.names[:23]],dtype=torch.float32);hi=torch.tensor([limits[n][1]-.03 for n in m.names[:23]],dtype=torch.float32)
 ankles=[m.names.index(s+'_ankle_pitch_joint') for s in ['left','right']];six=[m.names.index(s+'_'+j+'_joint') for j in ['hip_pitch','knee','ankle_pitch'] for s in ['left','right']];rows=[]
 for c in [0,1,6,7]:
  a=dict(np.load(src/f'schedule_{c}.npz'));plan=read(D/f'balanced_force_paths_144/plan_{c}.json')['plan'];Q=np.array([e['load_checks'][0]['command'] for e in plan['path']]);arc=np.r_[0,np.cumsum(abs(np.diff(Q,axis=0)).max(1))];p=a['positions'];assert np.all(np.diff(p)>=0) and p[-1]==arc[-1]
  edges=np.clip(np.searchsorted(arc,p,side='right')-1,0,len(arc)-2);f=(p-arc[edges])/(arc[edges+1]-arc[edges]);reconstructed=(Q[edges]+f[:,None]*(Q[edges+1]-Q[edges])).astype(np.float32)
  # Arithmetic filters can leave one float32 ULP relative to a double-precision
  # interpolation; use the existing3e-7 command subtraction check, report exactness.
  err=float(abs(reconstructed-a['nominal']).max());assert err<3e-7
  assert np.array_equal(a['nominal'][-1],Q[-1].astype(np.float32));assert np.array_equal(Q[-1,23:],Q[0,23:])
  gaps=np.array([x['pelvis_gap_m'] for x in plan['path']]);gaps[80:]=0;assert np.array_equal(np.interp(p,arc,gaps),a['gaps'])
  hover=float(np.interp(.012,gaps[:81][::-1],arc[:81][::-1]));flat=np.flatnonzero(abs(p-hover)<1e-10);assert len(flat)==26 and np.all(np.diff(flat)==1)
  near=p[:-1]>=hover-1e-10;speed=float(abs(np.diff(a['gaps'])[near]).max()/.02);assert speed<=.006+1e-10
  b=np.vstack([Q[0].astype(np.float32),a['nominal'][:-1]]);cap=np.full(37,.003,np.float32);cap[ankles]=.00225;assert np.all(a['nominal']<=b+cap) and np.all(a['nominal']>=b-cap)
  replays=[]
  for device in ['cpu','cuda:0']:
   cmd=torch.tensor(Q[0],dtype=torch.float32,device=device);out=[];clipped=0
   for tick,tar in enumerate(a['targets']):
    desired=torch.tensor(tar[:23],device=device)
    if tick>=700:
     for j in range(3):desired[six[2*j:2*j+2]]+=float(off[c,j])
    clipped+=int(((desired<lo.to(device))|(desired>hi.to(device))).sum());cmd[:23]+=(.1*(desired.clamp(min=lo.to(device),max=hi.to(device))-cmd[:23])).clamp(-.003,.003);out.append(cmd.cpu().numpy().copy())
   out=np.array(out);assert not clipped and np.array_equal(out,a['commands']);replays.append(True)
  before=np.vstack([Q[0].astype(np.float32),a['commands'][:-1]]);assert np.all(a['commands']<=before+np.float32(.003)) and np.all(a['commands']>=before-np.float32(.003));assert np.array_equal(a['commands'][:,23:],np.broadcast_to(Q[0,23:].astype(np.float32),a['commands'][:,23:].shape))
  duration=(np.flatnonzero(p==arc[-1])[0]+1)*.02;assert duration<=13.5
  rows.append(dict(case=c,ticks=len(p),duration_s=duration,nominal_interpolation_max_error_rad=err,nominal_interpolation_exact=bool(np.array_equal(reconstructed,a['nominal'])),nominal_endpoint_exact=True,hold_ticks=25,max_normal_speed_m_s=speed,cpu_gpu_actual_filter_exact=all(replays),clipped_targets=0,fixed14_fingers_exact=True))
 out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,scope=__doc__,rows=rows,physics_ticks=0,optimizer_updates=0,physical_admission=False,source_sha256={str(p.resolve()):sha(p) for p in [Path(__file__),src/'SHA256.json',D/'balanced_force_paths_144/SHA256.json',D/'cooked_model_50/model.json',cp]})
 (O/'independent_audit.json').write_text(json.dumps(out,indent=2));(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2));print(json.dumps(out),flush=True)
if __name__=='__main__':main()
