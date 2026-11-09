"""Independent saved-state audit434, without constructing new candidates.
Rebuild all1467 interpolation states with SciPySlerp, independent FK geometry,
equilibrium, and16-joint central finite virtual work per state. Replay derived
clock and rounding arithmetic. Failure of compatibility is reported, not an
assertion that the candidate must pass. No root/optimizer/filter/runtime/physics.
"""
from pathlib import Path
import json,hashlib,datetime
import numpy as np
from scipy.spatial import ConvexHull,QhullError
from scipy.spatial.transform import Rotation,Slerp
from humanoid_sitting.shared_frame import BodyModel,apply,transform
from plan_seat_load_targets import clipped_vertices
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');O=D/'clock_lift_audit_v2_435'
read=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def main():
 launch=read(D/'clock_lift_launch_v2_434.json')
 for p,h in launch['source_input_sha256'].items():assert sha(p)==h,p
 for p,h in read(D/'clock_lift_v2_434/SHA256.json').items():assert sha(D/'clock_lift_v2_434'/p)==h,p
 O.mkdir(exist_ok=False);data=read(D/'clock_lift_v2_434/results.json');records=read(D/'clock_lift_v2_434/records.json');a=dict(np.load(D/'clock_lift_v2_434/lift.npz'));old=read(D/'lattice_path_430/results.json');path=old['dense']['path'];nodes=old['dense']['command_nodes'];m=BodyModel(D/'cooked_model_50/model.json');ine=read(D.parent/'crouch_rest_bridge_01/balanced_standing10_01/static_inertial_model.json');mass=np.array(ine['mass']);W=mass.sum()*9.81;moving=a['moving'];fixed=np.array([j for j in range(37) if j not in moving]);kp=np.array([200,150,150,200,160,160]*2+[40]*4);effort=np.array([300,300,300,300,20,20]*2+[300]*4);q0=np.array(path[0]['q']);c0=np.array(nodes[0]['command']);tau0=np.array(nodes[0]['planned_static_torque_Nm']);Q=np.array([p['q'] for p in path]);roots=np.array([p['root'] for p in path]);times=np.r_[0.,np.cumsum(np.r_[old['dense']['segment_durations_s'][:99],.5,old['dense']['segment_durations_s'][99:]])];ids=np.r_[np.arange(100),99,np.arange(100,115)];assert np.array_equal(times,a['times']) and np.array_equal(ids,a['vertex_ids']);sides=['left','right'];n=a['plane_normal'];d=float(a['plane_d']);limits={j['name']:np.deg2rad(j['limits']) for j in m.joints if j['type']=='PhysicsRevoluteJoint'}
 # Plane angle and seat origin recovered from immutable measured430 source plane.
 sr=Rotation.from_euler('y',np.rad2deg(np.arctan2(n[0],n[2]))).as_matrix();height=-d/n[2];center=np.array([-.4,0,height])+np.array([0,0,-.4*n[0]/n[2]])
 # The measured plane is n dot x+d=0. Its height at x=-.4 is the seat-center height.
 center[2]=-(d+n[0]*center[0])/n[2]
 stored_initial=dict(np.load(D/'lattice_path_430/initial.npz'));stored_final=dict(np.load(D/'lattice_path_430/final.npz'));Wnodes=np.vstack([nodes[0]['corner_fractions'],stored_final['local_outputs'][:,46:54]]);Pnodes=np.concatenate([np.array(nodes[0]['corner_world_points'])[None],stored_initial['points']])
 for k,v in enumerate(nodes):
  cnt=len(v['corner_fractions']);assert np.array_equal(Wnodes[k,:cnt],v['corner_fractions']) and np.array_equal(Pnodes[k,:cnt],v['corner_world_points']);assert not np.count_nonzero(Wnodes[k,cnt:]) and not np.count_nonzero(Pnodes[k,cnt:])
 errors=dict(interpolation=0.,geometry=0.,work=0.,equilibrium=0.,command=0.,derived_clock=0.,rounding=0.);finite_work=[];metrics=[]
 for ix,rec in enumerate(records):
  t=rec['time_s'];vertex=rec['vertex']
  if vertex is None:
   j=min(max(int(np.searchsorted(times,t,side='right')-1),0),114);frac=(t-times[j])/(times[j+1]-times[j]);k,l=int(ids[j]),int(ids[j+1]);frac=0. if k==l else frac
  else:j=vertex;k=l=int(ids[j]);frac=0.
  assert (j,k,l)==(rec['segment'],rec['knot0'],rec['knot1']) and frac==rec['fraction']
  q=(1-frac)*Q[k]+frac*Q[l];q[fixed]=q0[fixed];root=(1-frac)*roots[k,:3]+frac*roots[l,:3];R=Rotation.from_quat(roots[k,3:]).as_matrix() if frac==0 else Slerp([0,1],Rotation.from_quat(roots[[k,l],3:]))([frac]).as_matrix()[0]
  T=transform(R,root);fk=m.fk(q);poses={b:T@v for b,v in fk.items()};com=np.average([apply(poses[b],cc) for b,cc in zip(ine['body_names'],ine['local_com'])],axis=0,weights=mass)
  errors['interpolation']=max(errors['interpolation'],float(abs(q-a['q'][ix]).max()),float(abs(root-a['root'][ix,:3]).max()),float(abs(R-Rotation.from_quat(a['root'][ix,3:]).as_matrix()).max()))
  w=(1-frac)*Wnodes[k]+frac*Wnodes[l];P=(1-frac)*Pnodes[k]+frac*Pnodes[l];normal=(1-frac)*nodes[k]['assumed_seat_normal_fraction']+frac*nodes[l]['assumed_seat_normal_fraction'];assert abs(w-a['weights'][ix]).max()<1e-15 and abs(P-a['points'][ix]).max()<1e-15
  pv=apply(T,m.vertices['pelvis']);pd=pv@n+d;seat=pv[pd<=pd.min()+1e-7].mean(0);seat-=n*(seat@n+d);equ=np.r_[w.sum()+normal/n[2]-1,w@P[:,:2]+normal/n[2]*seat[:2]-com[:2]];errors['equilibrium']=max(errors['equilibrium'],float(abs(equ-a['equilibrium'][ix]).max()),float(abs(com-a['com'][ix]).max()))
  feet={s:apply(poses[s+'_ankle_roll_link'],m.vertices[s+'_ankle_roll_link']) for s in sides};st=np.concatenate([feet[s] for s in rec['stance']]);sole=st[st[:,2]<st[:,2].min()+.003]
  try:H=ConvexHull(sole[:,:2]).equations;cop=float(min(-(H[:,:2]@com[:2]+H[:,2])))
  except QhullError:cop=-1.
  clear=[];floor=[]
  for b,v in m.vertices.items():
   z=apply(poses[b],v)
   if b not in ['left_ankle_roll_link','right_ankle_roll_link']:floor.append(float(z[:,2].min()))
   if b in ['pelvis','left_hip_yaw_link','right_hip_yaw_link','torso_link','logo_link','left_ankle_roll_link','right_ankle_roll_link']:continue
   z=(z-center)@sr;z[:,0]-=.4;z=clipped_vertices(z)
   if len(z):clear.append(float(z[:,2].min()))
  pe=[];re=[];fe=[];fr=[]
  for side in sides:
   pa=path[k]['foot_targets'][side];pb=path[l]['foot_targets'][side];target=(1-frac)*np.array(pa['position'])+frac*np.array(pb['position']);att=np.array(pa['rotation']) if frac==0 else Slerp([0,1],Rotation.from_matrix([pa['rotation'],pb['rotation']]))([frac]).as_matrix()[0];F=poses[side+'_ankle_roll_link'];pe.append(np.linalg.norm(F[:3,3]-target));re.append(Rotation.from_matrix(att.T@F[:3,:3]).magnitude());ref=a['actual_foot_reference'][ix,sides.index(side)];fe.append(np.linalg.norm(F[:3,3]-ref[:3,3]));fr.append(Rotation.from_matrix(ref[:3,:3].T@F[:3,:3]).magnitude())
  loc=(pv-center)@sr;mask=pd<pd.min()+.002;patch=float(min(.25-abs(loc[mask,0]).max(),.45-abs(loc[mask,1]).max()))
  vals=dict(clearance_m=min(clear) if clear else 1.,foot_cop_margin_m=cop,pelvis_gap_m=float(pd.min()),floor_clearance_m=min(floor),patch_margin_m=patch,foot_position_error_m=float(max(pe)),foot_orientation_error_rad=float(max(re)),interpolated_foot_position_defect_m=float(max(fe)),interpolated_foot_orientation_defect_rad=float(max(fr)),com_interpolation_defect_m=float(max(abs(com[:2]-a['com_reference'][ix,:2]))))
  errors['geometry']=max(errors['geometry'],max(abs(v-rec[key]) for key,v in vals.items()));metrics.append(vals)
  # Independent central virtual work: force application points move with the
  # appropriate child foot under each joint perturbation. Seat load acts on root.
  local=[np.linalg.inv(poses[s+'_ankle_roll_link'])@np.r_[p,1.] for p,s in zip(P,nodes[0]['corner_sides'])];tau=[]
  for joint in moving:
   vv=[]
   for sign in [-1,1]:
    qq=q.copy();qq[joint]+=sign*1e-5;ff=m.fk(qq);potential=sum(mm*9.81*apply(T@ff[b],cc)[2] for b,mm,cc in zip(ine['body_names'],mass,ine['local_com']));contact=sum(W*ww*(T@ff[s+'_ankle_roll_link']@ll)[2] for ww,s,ll in zip(w,nodes[0]['corner_sides'],local));vv.append(potential-contact)
   tau.append((vv[1]-vv[0])/2e-5)
  tau=np.asarray(tau);finite_work.append(tau);errors['work']=max(errors['work'],float(abs(tau-a['torque'][ix]).max()));expected=c0.copy();expected[moving]+=q[moving]-q0[moving]+(tau-tau0)/kp
  if k==0 and frac==0:expected=c0.copy()
  errors['command']=max(errors['command'],float(abs(expected-a['command'][ix]).max()));assert np.array_equal(a['q'][ix,fixed],q0[fixed]) and np.array_equal(a['command'][ix,fixed],c0[fixed])
  if (ix+1)%200==0:print(json.dumps(dict(audited_records=ix+1,maximum_work_error_Nm=errors['work'])),flush=True)
 assert errors['interpolation']<1e-12 and errors['geometry']<1e-12 and errors['equilibrium']<1e-12 and errors['work']<1e-6 and errors['command']<2e-8,errors
 initial=dict(np.load(D/'lattice_path_430/initial.npz'));clock=a['command'][:676,moving];dd=np.diff(clock,axis=0);target=10*clock[1:]-9*clock[:-1];rate=np.c_[.02*initial['rates']-dd,.02*initial['rates']+dd];head=np.c_[initial['command_hi']-target,target-initial['command_lo']];errors['derived_clock']=float(max(abs(rate-a['clock_rate']).max(),abs(head-a['clock_target']).max()))
 ideal=np.load(D/'lattice_path_430/event_clock.npz')['clock'];rounded=ideal.astype(np.float32);diff=np.diff(rounded.astype(float),axis=0);rr=np.c_[.02*initial['rates']-diff,.02*initial['rates']+diff];assert np.array_equal(rounded,a['rounded_ideal']);errors['rounding']=float(abs(rr-a['rounded_ideal_rate']).max())
 assert errors['derived_clock']==errors['rounding']==0.
 result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,scope=__doc__,records=len(records),finite_work_columns=len(records)*16,errors=errors,reported_geometry_failures=data['sampled_geometry_failures'],reported_nonlinear_lift_compatible=data['nonlinear_lift_compatible'],inherited430_extra_buffer_passed=False,arithmetic_pass_not_admission=True,actual_float32_filter_ticks=0,physics_ticks=0,policy_updates=0,physical_admission=False,source_sha256={str(D/'clock_lift_v2_434/results.json'):sha(D/'clock_lift_v2_434/results.json'),str(Path(__file__).resolve()):sha(__file__)})
 np.save(O/'finite_work.npy',finite_work);(O/'independent_audit.json').write_text(json.dumps(result,indent=2));(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2));print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
