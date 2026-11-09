"""Independent fixed-integer-time/new-interior row algebra and legal-filter fixtures.
Uses saved377 arrays and audited373 source evidence; no FK/projection/optimization rerun.
CPU and GPU evaluation is arithmetic, ZERO physics or real command-schedule ticks.
"""
from pathlib import Path
import json,hashlib,datetime,math
import numpy as np
import torch
from low_seat_forward_lattice_377 import target_for_goal
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');O=D/'discrete_preparation_audit_378'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 O.mkdir(exist_ok=False);r=read(D/'discrete_preparation_v2_377/preparation.json');a=dict(np.load(D/'discrete_preparation_v2_377/fixture.npz'));old=dict(np.load(D/'support_path_373/final.npz'));pre=dict(np.load(D/'support_preparation_371/fixture.npz'))
 for p,h in r['source_input_sha256'].items():assert sha(p)==h,p
 counts=[0]*114
 for group,total in [(list(range(99,104)),101),([i for i in range(114) if not 99<=i<104],549)]:
  summed=sum(float(old['x'][1026+i]) for i in group);raw={i:float(old['x'][1026+i])/summed*total for i in group};base={i:max(1,math.floor(raw[i])) for i in group};remaining=total-sum(base.values());order=sorted(group,key=lambda i:(-(raw[i]-base[i]),i))
  for i in order[:remaining]:base[i]+=1
  for i in group:counts[i]=base[i]
 assert np.array_equal(counts,a['segment_ticks']);assert sum(counts)==650 and sum(counts[99:104])==101
 assert np.array_equal(a['x'][:1026],old['x'][:1026]);assert np.array_equal(a['x'][1026:],np.array(counts)*.02)
 raw=old['ineq'].copy();Y=a['Y']
 # All time-dependent original rows are rebuilt using saved373 neighboring commands.
 result=read(D/'support_path_373/results.json');from humanoid_sitting.shared_frame import BodyModel
 m=BodyModel(D/'cooked_model_50/model.json');moving=[m.names.index(s+'_'+j+'_joint') for s in ['left','right'] for j in ['hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch','ankle_roll']]+[m.names.index(s+'_'+j+'_joint') for s in ['left','right'] for j in ['shoulder_pitch','elbow_pitch']]
 C=np.array([v['command'] for v in result['dense']['command_nodes']])[:,moving];delta=np.diff(C,axis=0);dt=np.array(counts)*.02
 raw[12212:14036]=(dt[:,None]*pre['rates']-delta).ravel();raw[14036:15860]=(dt[:,None]*pre['rates']+delta).ravel();raw[15860:17684]=((pre['command_hi']-C[1:])*dt[:,None]-.18*delta).ravel();raw[17684:19508]=((C[1:]-pre['command_lo'])*dt[:,None]+.18*delta).ravel();gap=np.array([v['pelvis_gap_m'] for v in result['dense']['path']]);raw[19754:19759]=.006*dt[99:104]-(gap[99:104]-gap[100:105]);raw[19759]=13.-dt.sum()
 error=float(abs(raw-a['raw_ineq']).max());assert error<1e-10
 keep=old['keep'].copy();keep[19759]=False;assert np.array_equal(keep,a['keep']) and np.array_equal(old['hard'],a['hard']) and np.array_equal(old['scale'],a['scale'])
 tight=np.zeros(20284);eps=2.**(-23)
 for k in range(114):
  for j in range(8):
   for off in [64,72]:
    if keep[k*80+off+j]:tight[k*80+off+j]=8*eps
 tight[12004:12212]=8*eps;tight[19760:]=8*eps
 limits={j['name']:np.deg2rad(j['limits']) for j in m.joints if j['type']=='PhysicsRevoluteJoint'};lo=np.array([limits[m.names[j]][0]+.03 for j in moving],np.float32);hi=np.array([limits[m.names[j]][1]-.03 for j in moving],np.float32)
 mag=np.maximum(abs(lo),abs(hi));ulp=np.array([2.**(math.floor(math.log2(float(t)))-23) for t in mag]);assert np.array_equal(ulp,a['ulp'])
 for start in [12212,14036]:tight[start:start+1824]=(np.array(counts)[:,None]*4*ulp).ravel()
 for start in [15860,17684]:tight[start:start+1824]=(dt[:,None]*32*ulp).ravel()
 assert np.array_equal(tight,a['tighten']);assert np.array_equal(a['ineq'],a['raw_ineq']-tight)
 dot=np.r_[np.array([math.fsum(float(x)*float(y) for x,y in zip(row,a['direction'][a['free']])) for row in a['E']]),np.array([math.fsum(float(x)*float(y) for x,y in zip(row,a['direction'][a['free']])) for row in a['I']])];doterror=float(abs(dot-a['analytic']).max());assert doterror<1e-12;assert abs(dot-a['fd']).max()<2e-6
 # Legal bound lattice is inward with respect to exact original .03 margin.
 physical_lo=np.array([limits[m.names[j]][0]+.03 for j in moving]);physical_hi=np.array([limits[m.names[j]][1]-.03 for j in moving]);lo=np.where(lo.astype(float)<physical_lo,np.nextafter(lo,np.float32(np.inf)),lo);hi=np.where(hi.astype(float)>physical_hi,np.nextafter(hi,np.float32(-np.inf)),hi)
 fixtures=[]
 for i in range(16):
  cap=float(pre['rates'][i]*.02)
  for prev,goal in [(lo[i]+np.float32(64*ulp[i]),lo[i]+np.float32(65*ulp[i])),(hi[i]-np.float32(64*ulp[i]),hi[i]-np.float32(65*ulp[i])),(np.float32((lo[i]+hi[i])*.5),np.float32((lo[i]+hi[i])*.5+cap*.75)),(np.float32((lo[i]+hi[i])*.5),np.float32((lo[i]+hi[i])*.5-cap*.75))]:
   prev=np.float32(prev);goal=np.float32(goal);res=target_for_goal(prev,goal,lo[i],hi[i]);assert res is not None;u,out=res;assert lo[i]<=u<=hi[i];assert abs(float(out)-float(goal))<=ulp[i];fixtures.append([i,prev,goal,u,out,lo[i],hi[i]])
  if lo[i]<0<hi[i]:
   prev=np.float32(-cap*.25);goal=np.float32(cap*.25);res=target_for_goal(prev,goal,lo[i],hi[i]);assert res is not None;u,out=res;assert abs(float(out)-float(goal))<=ulp[i];fixtures.append([i,prev,goal,u,out,lo[i],hi[i]])
 f=np.array(fixtures);outs=[]
 for device in ['cpu','cuda:0']:
  previous=torch.tensor(f[:,1],dtype=torch.float32,device=device);target=torch.tensor(f[:,3],dtype=torch.float32,device=device);low=torch.tensor(f[:,5],dtype=torch.float32,device=device);high=torch.tensor(f[:,6],dtype=torch.float32,device=device);out=previous+(.1*(target.clamp(min=low,max=high)-previous)).clamp(-.003,.003);outs.append(out.cpu().numpy())
 assert np.array_equal(outs[0],outs[1]) and np.array_equal(outs[0],f[:,4].astype(np.float32));np.savez_compressed(O/'filter_fixtures.npz',records=f,cpu=outs[0],gpu=outs[1],moving=moving,lo=lo,hi=hi,ulp=ulp)
 out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,scope=__doc__,all_original_rows=20284,new_interior_rows=int(np.count_nonzero(tight)),movement_ticks=650,normal_ticks=101,synthetic_quiet_ticks=25,exact_integer_time_s=13.5,raw_row_replay_error=error,derivative_scalar_dot_error=doterror,raw_directional_error=float(abs(dot-a['fd']).max()),filter_fixture_records=len(f),cpu_gpu_exact=True,filter_nearest_output_not_relabelled_exact_goal=True,maximum_fixture_goal_departure_rad=float(abs(f[:,4]-f[:,2]).max()),actual_candidate_schedule_checked=False,geometry_not_recomputed=True,physical_admission=False,physics_ticks=0,policy_optimizer_updates=0,source_sha256={str(p.resolve()):sha(p) for p in [Path(__file__),Path('scripts/low_seat_forward_lattice_377.py'),D/'discrete_preparation_v2_377/preparation.json',D/'discrete_preparation_v2_377/fixture.npz',D/'support_path_373/final.npz',D/'support_path_audit_374/independent_audit.json']})
 (O/'independent_audit.json').write_text(json.dumps(out,indent=2));(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
