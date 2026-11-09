"""Independent saved-state audit of NEW440 construction, not434's interpolant.
No root, projection, optimizer or forward filter is rerun. Full geometry and
contact/COM identities are checked for1467 records. Central virtual work is new
for changed states; unchanged cardinal states inherit the sealed431 work check
only after explicit pose/foot-wrench/command equivalence. Exact duplicate new
states share one saved finite-work result. No435 finite-work loop is repeated.
"""
from pathlib import Path
from fractions import Fraction
import json,hashlib,datetime
import numpy as np
from scipy.spatial import ConvexHull,QhullError
from scipy.spatial.transform import Rotation,Slerp
from humanoid_sitting.shared_frame import BodyModel,apply,transform
from plan_seat_load_targets import clipped_vertices
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');O=D/'contact_lift_audit_441'
read=lambda p:json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 O.mkdir(exist_ok=False)
 for p,h in read(D/'contact_lift_440/SHA256.json').items():assert sha(D/'contact_lift_440'/p)==h,p
 result=read(D/'contact_lift_440/results.json');records=read(D/'contact_lift_440/records.json');a=dict(np.load(D/'contact_lift_440/states.npz'));prep=read(D/'contact_lift_preparation_438/preparation.json');pf=dict(np.load(D/'contact_lift_preparation_438/fixture.npz'));initial=dict(np.load(D/'lattice_path_430/initial.npz'));previous=read(D/'lattice_path_430/results.json');path=previous['dense']['path'];nodes=previous['dense']['command_nodes'];source_lift=np.load(D/'clock_lift_v2_434/lift.npz');n=source_lift['plane_normal'];d=float(source_lift['plane_d'])
 m=BodyModel(D/'cooked_model_50/model.json');ine=read(D.parent/'crouch_rest_bridge_01/balanced_standing10_01/static_inertial_model.json');mass=np.array(ine['mass']);W=sum(mass)*9.81;sides=['left','right'];legs=[m.names.index(s+'_'+j+'_joint') for s in sides for j in ['hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch','ankle_roll']];arms=[m.names.index(s+'_'+j+'_joint') for s in sides for j in ['shoulder_pitch','elbow_pitch']];moving=legs+arms;fixed=[j for j in range(37) if j not in moving];kp=np.array([200,150,150,200,160,160]*2+[40]*4);effort=np.array([300,300,300,300,20,20]*2+[300]*4)
 q0=np.array(path[0]['q']);c0=np.array(nodes[0]['command']);tau0=np.array(nodes[0]['planned_static_torque_Nm']);limits={j['name']:np.deg2rad(j['limits']) for j in m.joints if j['type']=='PhysicsRevoluteJoint'}
 sr=Rotation.from_euler('y',np.rad2deg(np.arctan2(n[0],n[2])),degrees=True).as_matrix();center=np.array([-.4,0.,0.]);center[2]=-(d+n[0]*center[0])/n[2]
 errors=dict(geometry=0.,equilibrium=0.,frame=0.,force_interpolation=0.,work=0.,command=0.,source_equivalence=0.,clock=0.,prior_rows=0.);cache={};work=[];work_kind=[];columns=0;inherited=0;duplicates=0;geometry=[]
 assert read(D/'lattice_path_audit_431/independent_audit.json')['passed']
 times=[Fraction(int(x),int(y)) for x,y in prep['vertex_times']];changed={x['knot'] for x in prep['changed_nominal_nodes']}
 for ix,rec in enumerate(records):
  k,l=rec['knot0'],rec['knot1'];frac=rec['fraction'];q=a['q'][ix];root=a['root'][ix,:3];R=Rotation.from_quat(a['root'][ix,3:]).as_matrix();T=transform(R,root);fk=m.fk(q);poses={b:T@v for b,v in fk.items()};com=np.average([apply(poses[b],cc) for b,cc in zip(ine['body_names'],ine['local_com'])],axis=0,weights=mass);w=a['weights'][ix];P=a['points'][ix];normal=a['normal'][ix]
  # Persistent local corners must move with the DECLARED foot frame, never with
  # an interpolated world array slot. Actual FK must match that frame independently.
  for j,side in enumerate(sides):
   A=pf['feet'][k,j];B=pf['feet'][l,j];att=A[:3,:3] if frac==0 else Slerp([0,1],Rotation.from_matrix([A[:3,:3],B[:3,:3]]))([frac]).as_matrix()[0];pos=(1-frac)*A[:3,3]+frac*B[:3,3];F=transform(att,pos);points=apply(F,pf['local'][j]);errors['frame']=max(errors['frame'],float(abs(F-a['foot_frames'][ix,j]).max()),float(abs(points-P[j*4:(j+1)*4]).max()))
  if k<104:errors['force_interpolation']=max(errors['force_interpolation'],float(abs(w-((1-frac)*pf['weights'][k]+frac*pf['weights'][l])).max()))
  if rec['stance']==['left'] and k>=16:assert np.all(w[4:]==0)
  assert np.all(w>=0)
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
  for j,side in enumerate(sides):
   pa=path[k]['foot_targets'][side];pb=path[l]['foot_targets'][side];target=(1-frac)*np.array(pa['position'])+frac*np.array(pb['position']);att=np.array(pa['rotation']) if frac==0 else Slerp([0,1],Rotation.from_matrix([pa['rotation'],pb['rotation']]))([frac]).as_matrix()[0];F=poses[side+'_ankle_roll_link'];pe.append(np.linalg.norm(F[:3,3]-target));re.append(Rotation.from_matrix(att.T@F[:3,:3]).magnitude());ref=a['foot_frames'][ix,j];fe.append(np.linalg.norm(F[:3,3]-ref[:3,3]));fr.append(Rotation.from_matrix(ref[:3,:3].T@F[:3,:3]).magnitude())
  loc=(pv-center)@sr;mask=pd<pd.min()+.002;patch=float(min(.25-abs(loc[mask,0]).max(),.45-abs(loc[mask,1]).max()));vals=dict(clearance_m=min(clear) if clear else 1.,foot_cop_margin_m=cop,pelvis_gap_m=float(pd.min()),floor_clearance_m=min(floor),patch_margin_m=patch,foot_position_error_m=float(max(pe)),foot_orientation_error_rad=float(max(re)),interpolated_foot_position_defect_m=float(max(fe)),interpolated_foot_orientation_defect_rad=float(max(fr)),com_interpolation_defect_m=float(max(abs(com[:2]-a['com_target'][ix]))))
  errors['geometry']=max(errors['geometry'],max(abs(v-rec[key]) for key,v in vals.items()));geometry.append(vals)
  tilt=np.rad2deg(np.arccos(np.clip(R[2,2],-1,1)));geo=bool(max(pe)<=.003 and max(re)<=np.deg2rad(.5) and cop>=.005 and min(floor)>=.005 and tilt<=38 and (pd.min()>=.02 or patch>=.015) and pd.min()>=-1e-8 and vals['clearance_m']>=.003);assert geo==rec['sampled_geometry_passed']
  # Inherit only cardinal-equivalent source states, with explicit wrench check.
  source_ok=False
  if k==l and k not in changed:
   op=np.array(nodes[k]['corner_world_points']);ow=np.array(nodes[k]['corner_fractions']);sides0=nodes[k]['corner_sides'];moment=[]
   for j,side in enumerate(sides):
    sel=np.array([x==side for x in sides0]);moment.append(np.r_[ow[sel].sum(),ow[sel]@op[sel,:2]])
   current=np.array([np.r_[w[j*4:(j+1)*4].sum(),w[j*4:(j+1)*4]@P[j*4:(j+1)*4,:2]] for j in range(2)])
   err=max(float(abs(q-np.array(path[k]['q'])).max()),float(abs(root-np.array(path[k]['root'][:3])).max()),float(abs(R-Rotation.from_quat(path[k]['root'][3:]).as_matrix()).max()),float(abs(current-np.array(moment)).max()),float(abs(a['torque'][ix]-np.array(nodes[k]['planned_static_torque_Nm'])).max()))
   source_ok=err<2e-10
   if source_ok:errors['source_equivalence']=max(errors['source_equivalence'],err)
  key=hashlib.sha256(np.r_[q,root,R.ravel(),w,P.ravel()].tobytes()).hexdigest()
  if source_ok:fd=np.array(nodes[k]['planned_static_torque_Nm']);label='inherited431_cardinal_equivalence';inherited+=1
  elif key in cache:fd=cache[key];label='duplicate_new_state';duplicates+=1
  else:
   local=np.array([(P[j*4:(j+1)*4]-poses[side+'_ankle_roll_link'][:3,3])@poses[side+'_ankle_roll_link'][:3,:3] for j,side in enumerate(sides)])
   def positions(qtest):
    ff=m.fk(qtest);pts=np.array([apply(T@ff[b],cc) for b,cc in zip(ine['body_names'],ine['local_com'])]);contact=np.concatenate([apply(T@ff[side+'_ankle_roll_link'],local[j]) for j,side in enumerate(sides)]);return pts,contact
   fd=[];h=1e-5
   for joint in moving:
    qp=q.copy();qm=q.copy();qp[joint]+=h;qm[joint]-=h;bp,fp=positions(qp);bm,fm=positions(qm)
    fd.append(float((np.sum(mass*9.81*(bp[:,2]-bm[:,2]))-W*np.sum(w*(fp[:,2]-fm[:,2])))/(2*h)))
   fd=np.array(fd);cache[key]=fd;columns+=16;label='new_finite_work'
   with (O/'finite_work.jsonl').open('a') as journal:journal.write(json.dumps(dict(record=ix,torque=fd.tolist(),state_sha256=key))+'\n')
  work.append(fd);work_kind.append(label);errors['work']=max(errors['work'],float(abs(fd-a['torque'][ix]).max()))
  expected=c0.copy();expected[moving]+=q[moving]-q0[moving]+(fd-tau0)/kp
  if k==0 and frac==0:expected=c0.copy()
  errors['command']=max(errors['command'],float(abs(expected-a['command'][ix]).max()));assert np.array_equal(q[fixed],q0[fixed]) and np.array_equal(a['command'][ix,fixed],c0[fixed])
  if (ix+1)%200==0:print(json.dumps(dict(audited_records=ix+1,new_finite_work_columns=columns,maximum_errors=errors)),flush=True)
 # Persist all completed finite-work evidence before aggregate assertions.
 np.savez_compressed(O/'independent_work.npz',torque=np.array(work),kind=np.array(work_kind));(O/'geometry.json').write_text(json.dumps(geometry))
 assert max(errors[k] for k in ['geometry','equilibrium','frame','force_interpolation'])<1e-10 and errors['work']<1e-6 and errors['command']<2e-8,errors
 c=a['command'][:676,moving];dc=np.diff(c,axis=0);u=10*c[1:]-9*c[:-1];raw=np.c_[.02*initial['rates']-dc,.02*initial['rates']+dc,initial['command_hi']-u,u-initial['command_lo']];prior=raw-np.r_[4*initial['ulp'],4*initial['ulp'],32*initial['ulp'],32*initial['ulp']];buffer=prior/np.r_[.02*initial['rates'],.02*initial['rates'],np.full(32,.03)]-2.**-23
 errors['clock']=float(max(abs(raw-a['clock_raw']).max(),abs(prior-a['clock_prior']).max(),abs(buffer-a['clock_buffer']).max()));assert errors['clock']==0.
 final=read(D/'lattice_path_430/results.json');frozen=np.load(D/'lattice_path_430/final.npz');coord=a['node_coordinate_bounds'];rebuilt=np.r_[a['node_prior'][frozen['keep'].astype(bool)]/frozen['scale'][frozen['keep'].astype(bool)]-2.**-23,coord-2.**-23];errors['prior_rows']=float(abs(rebuilt-a['node_buffered']).max());assert errors['prior_rows']==0
 from audit_low_seat_contact_rows_441 import audit_rows
 row_audit=audit_rows(a,records,initial,pf)
 out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,scope=__doc__,records=1467,new_finite_work_columns=columns,inherited_cardinal_records=inherited,duplicate_new_state_records=duplicates,detailed_new_finite_work_array_available=True,maximum_errors=errors,node_row_audit=row_audit,sampled_geometry_failures=result['sampled_geometry_failures'],new_clock_rate_minimum=float(raw[:,:32].min()),new_clock_target_minimum=float(raw[:,32:].min()),complete_independent_original_row_rebuild=True,prior_buffer_algebra_rebuilt=True,actual_forward_filter_ticks=0,physics_ticks=0,physical_admission=False)
 (O/'independent_audit.json').write_text(json.dumps(out,indent=2));(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2));print(json.dumps(out,indent=2),flush=True)
if __name__=='__main__':main()
