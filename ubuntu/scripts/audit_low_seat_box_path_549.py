"""Independent actual-X geometry/virtual-work audit (auxiliary Y is never substituted) of NEW548 final diagnostic or accepted path.
No root/optimizer/AD fixture or old finite-work loop rerun. Exactly unchanged
states inherit sealed441 work; new states receive central finite work, saved
before aggregate assertions. All1467 labels remain, with exact duplicate work
cached by complete pose and wrench identity. Arithmetic pass is not admission.
"""
from pathlib import Path
import json,datetime,hashlib
import numpy as np
from scipy import sparse
from scipy.spatial import ConvexHull,QhullError
from scipy.spatial.transform import Rotation,Slerp
from humanoid_sitting.shared_frame import BodyModel,apply,transform
from plan_seat_load_targets import clipped_vertices
from low_seat_clock_transcription_444 import load_inputs,sha
from audit_low_seat_box_steps_549 import audit_steps
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');O=D/'box_path_audit_549'
read=lambda p:json.loads(Path(p).read_text())
def main():
 assert read(D/'box_path_exit_548.json')['actual_exit']==0
 for p,h in read(D/'box_path_launch_548.json')['source_input_sha256'].items():assert sha(p)==h,p
 for p,h in read(D/'box_path_548/SHA256.json').items():assert sha(D/'box_path_548'/p)==h,p
 for folder in ['joint_ray_path_audit_479','free_gap_path_audit_473','contact_lift_440','contact_lift_audit_441','clock_transcription_preparation_444']:
  for p,h in read(D/folder/'SHA256.json').items():assert sha(D/folder/p)==h,p
 O.mkdir(exist_ok=False);step=audit_steps(O);s=load_inputs();mesh=dict(np.load(D/'clock_transcription_preparation_444/mesh.npz'));a=dict(np.load(D/'box_path_548/final.npz'));ls=mesh['label_to_state'];X=a['X'][ls];Y=a['Y'][ls];m=s['m'];ine=s['ine'];mass=s['mass'];W=s['weight'];n=s['n'];d=s['plane_d'];moving=s['moving'];legs=s['legs'];arms=s['arms'];fixed=s['fixed'];q0=s['q0'];c0=s['c0'];tau0=s['tau0'];kp=s['kp'];source=s['z'];records=s['records'];path=read(D/'lattice_path_430/results.json')['dense']['path'];ini=s['ini'];sr=Rotation.from_euler('y',np.arctan2(n[0],n[2])).as_matrix();center=np.array([-.4,0.,0.]);center[2]=-(d+n[0]*center[0])/n[2]
 sealed=np.load(D/'free_gap_path_audit_473/independent_states.npz');inherited473=0
 sealed479=np.load(D/'joint_ray_path_audit_479/independent_states.npz');inherited479=0
 errors=dict(com=0.,gap=0.,frame=0.,equilibrium=0.,torque=0.,command=0.);actual_frame_max=0.;samples=[];cache={};work=[];kinds=[];newcolumns=0;inherited=0;duplicates=0;Q=[];ROOT=[];C=[]
 for ix,rec in enumerate(records):
  x=X[ix];f=x[2];p=x[3];q=q0.copy();q[legs]=x[4:16];q[arms]=q0[arms]-.35*f;R=s['baseR'][ix]@Rotation.from_rotvec(p*s['rotvec']).as_matrix();root=np.r_[x[:2],s['root0'][2]+s['dz']*f];T=transform(R,root);fk=m.fk(q);poses={b:T@v for b,v in fk.items()};com=np.average([apply(poses[b],cc) for b,cc in zip(ine['body_names'],ine['local_com'])],axis=0,weights=mass);w=x[16:24];P=source['points'][ix];normal=source['normal'][ix];pv=apply(T,m.vertices['pelvis']);pd=pv@n+d;seat=pv[pd<=pd.min()+1e-7].mean(0);seat-=n*(seat@n+d);equ=np.r_[w.sum()+normal/n[2]-1,w@P[:,:2]+normal/n[2]*seat[:2]-com[:2]]
  errors['com']=max(errors['com'],float(abs(com-Y[ix,48:51]).max()));errors['gap']=max(errors['gap'],abs(float(pd.min())-Y[ix,47]));errors['equilibrium']=max(errors['equilibrium'],float(abs(equ-np.r_[Y[ix,14],Y[ix,12:14]]).max()))
  feet={side:apply(poses[side+'_ankle_roll_link'],m.vertices[side+'_ankle_roll_link']) for side in s['sides']};st=np.concatenate([feet[side] for side in rec['stance']]);sole=st[st[:,2]<st[:,2].min()+.003]
  try:H=ConvexHull(sole[:,:2]).equations;cop=float(min(-(H[:,:2]@com[:2]+H[:,2])))
  except QhullError:cop=-1.
  clear=[];floor=[]
  for body,v in m.vertices.items():
   z=apply(poses[body],v)
   if body not in ['left_ankle_roll_link','right_ankle_roll_link']:floor.append(float(z[:,2].min()))
   if body in ['pelvis','left_hip_yaw_link','right_hip_yaw_link','torso_link','logo_link','left_ankle_roll_link','right_ankle_roll_link']:continue
   z=(z-center)@sr;z[:,0]-=.4;z=clipped_vertices(z)
   if len(z):clear.append(float(z[:,2].min()))
  pe=[];re=[];fe=[];fr=[];k=rec['knot0'];l=rec['knot1'];fraction=rec['fraction']
  for j,side in enumerate(s['sides']):
   A=path[k]['foot_targets'][side];B=path[l]['foot_targets'][side];target=(1-fraction)*np.array(A['position'])+fraction*np.array(B['position']);att=np.array(A['rotation']) if fraction==0 else Slerp([0,1],Rotation.from_matrix([A['rotation'],B['rotation']]))([fraction]).as_matrix()[0];F=poses[side+'_ankle_roll_link'];ref=source['foot_frames'][ix,j];pe.append(np.linalg.norm(F[:3,3]-target));re.append(Rotation.from_matrix(att.T@F[:3,:3]).magnitude());fe.append(np.linalg.norm(F[:3,3]-ref[:3,3]));fr.append(Rotation.from_matrix(ref[:3,:3].T@F[:3,:3]).magnitude())
  actual_frame_max=max(actual_frame_max,max(fe),max(fr))
  predicted=[]
  for j,side in enumerate(s['sides']):
   actualfoot=poses[side+'_ankle_roll_link'];ref=source['foot_frames'][ix,j];rel=ref[:3,:3].T@actualfoot[:3,:3];predicted.extend(actualfoot[:3,3]-ref[:3,3]);predicted.extend(.1*np.array([rel[2,1]-rel[1,2],rel[0,2]-rel[2,0],rel[1,0]-rel[0,1]]))
  errors['frame']=max(errors['frame'],float(abs(np.asarray(predicted)-Y[ix,:12]).max()));loc=(pv-center)@sr;mask=pd<pd.min()+.002;patch=float(min(.25-abs(loc[mask,0]).max(),.45-abs(loc[mask,1]).max()));tilt=float(np.rad2deg(np.arccos(np.clip(R[2,2],-1,1))));housing=min(clear) if clear else 1.;geo=bool(max(pe)<=.003 and max(re)<=np.deg2rad(.5) and cop>=.005 and min(floor)>=.005 and tilt<=38 and (pd.min()>=.02 or patch>=.015) and pd.min()>=-1e-8 and housing>=.003)
  original_same=np.array_equal(x,s['X'][ix]);key=hashlib.sha256(np.r_[q,root,R.ravel(),w,P.ravel()].tobytes()).hexdigest()
  if np.array_equal(x,sealed479['X'][ix]):fd=sealed479['work'][ix].copy();label='inherited479_exact_input';inherited479+=1
  elif np.array_equal(x,sealed['X'][ix]):fd=sealed['work'][ix].copy();label='inherited473_exact_input';inherited473+=1
  elif original_same:fd=source['torque'][ix].copy();label='inherited441_exact_input';inherited+=1
  elif key in cache:fd=cache[key];label='duplicate_new_state';duplicates+=1
  else:
   local=np.array([(P[j*4:(j+1)*4]-poses[side+'_ankle_roll_link'][:3,3])@poses[side+'_ankle_roll_link'][:3,:3] for j,side in enumerate(s['sides'])])
   def positions(qt):
    ff=m.fk(qt);bp=np.array([apply(T@ff[b],cc) for b,cc in zip(ine['body_names'],ine['local_com'])]);fp=np.concatenate([apply(T@ff[side+'_ankle_roll_link'],local[j]) for j,side in enumerate(s['sides'])]);return bp,fp
   fd=[];h=1e-5
   for joint in moving:
    qp=q.copy();qm=q.copy();qp[joint]+=h;qm[joint]-=h;bp,fp=positions(qp);bm,fm=positions(qm);fd.append(float((np.sum(mass*9.81*(bp[:,2]-bm[:,2]))-W*np.sum(w*(fp[:,2]-fm[:,2])))/(2*h)))
   fd=np.array(fd);cache[key]=fd;label='new_finite_work';newcolumns+=16
   with (O/'finite_work.jsonl').open('a') as out:out.write(json.dumps(dict(record=ix,state_sha256=key,torque=fd.tolist()))+'\n')
  errors['torque']=max(errors['torque'],float(abs(fd-Y[ix,31:47]).max()));expected=c0.copy();expected[moving]+=q[moving]-q0[moving]+(fd-tau0)/kp;command=c0.copy();command[moving]=Y[ix,15:31]
  if ix==0:expected=c0.copy();command=c0.copy()
  errors['command']=max(errors['command'],float(abs(expected-command).max()));assert np.array_equal(q[fixed],q0[fixed]) and np.array_equal(command[fixed],c0[fixed]);assert np.all(w>=0)
  margin=min(min(command[j]-s['limits'][m.names[j]][0],s['limits'][m.names[j]][1]-command[j]) for j in range(37));moving_margin=min(min(command[j]-s['limits'][m.names[j]][0],s['limits'][m.names[j]][1]-command[j]) for j in moving);samples.append(dict(record=ix,time_s=rec['time_s'],phase=float(f),orientation_phase=float(p),housing_m=housing,cop_margin_m=cop,gap_m=float(pd.min()),floor_m=min(floor),patch_m=patch,tilt_deg=tilt,foot_position_m=max(pe),foot_orientation_rad=max(re),declared_frame_position_m=max(fe),declared_frame_orientation_rad=max(fr),command_margin_rad=float(margin),moving_command_margin_rad=float(moving_margin),sampled_geometry_passed=geo));work.append(fd);kinds.append(label);Q.append(q);ROOT.append(np.r_[root,Rotation.from_matrix(R).as_quat()]);C.append(command)
  if (ix+1)%250==0:print(json.dumps(dict(records=ix+1,new_finite_work_columns=newcolumns,errors=errors)),flush=True)
 np.savez_compressed(O/'independent_states.npz',q=Q,root=ROOT,command=C,work=work,work_kind=kinds,X=X,Y=Y);(O/'geometry.json').write_text(json.dumps(samples))
 # Persist all new detailed work before aggregate arithmetic assertions.
 assert max(errors[k] for k in ['com','gap','equilibrium'])<1e-10 and errors['frame']<2e-10 and errors['torque']<1e-6 and errors['command']<2e-8,errors
 old=sparse.load_npz(D/'clock_row_preparation_446/original_rows_matrix.npz');rows=np.load(D/'clock_row_preparation_v2_446/row_arrays.npz');raw=old@np.c_[a['X'],a['Y']].ravel()+rows['old_constant'];prior=raw-rows['old_tightening'];c=np.array(C)[:676][:,moving];dc=np.diff(c,axis=0);u=10*c[1:]-9*c[:-1];clockraw=np.c_[.02*ini['rates']-dc,.02*ini['rates']+dc,ini['command_hi']-u,u-ini['command_lo']];clockprior=clockraw-np.r_[4*ini['ulp'],4*ini['ulp'],32*ini['ulp'],32*ini['ulp']]
 np.savez_compressed(O/'independent_rows.npz',old_raw=raw,old_prior=prior,clock_raw=clockraw,clock_prior=clockprior)
 out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope=__doc__,passed=True,arithmetic_only=True,records=len(samples),new_finite_work_columns=newcolumns,inherited441_exact_records=inherited,inherited473_exact_records=inherited473,inherited479_exact_records=inherited479,duplicate_new_records=duplicates,maximum_errors=errors,actual_declared_frame_mixed_max=actual_frame_max,auxiliary_consistency_is_not_actual_feasibility=True,sampled_geometry_passed=all(x['sampled_geometry_passed'] for x in samples),sampled_geometry_failures=[x for x in samples if not x['sampled_geometry_passed']],minimum_housing_m=min(x['housing_m'] for x in samples),minimum_cop_m=min(x['cop_margin_m'] for x in samples),minimum_command_margin_rad=min(x['command_margin_rad'] for x in samples),minimum_moving_command_margin_rad=min(x['moving_command_margin_rad'] for x in samples),all37_margin_includes_fixed_fingers=True,original_clock_rate_minimum=float(clockraw[:,:32].min()),original_clock_target_minimum=float(clockraw[:,32:].min()),prior_clock_rate_minimum=float(clockprior[:,:32].min()),prior_clock_target_minimum=float(clockprior[:,32:].min()),step_audit=step,actual_forward_filter_ticks=0,physics_ticks=0,policy_updates=0,physical_admission=False)
 (O/'independent_audit.json').write_text(json.dumps(out,indent=2));(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2));print(json.dumps(out,indent=2),flush=True)
if __name__=='__main__':main()
