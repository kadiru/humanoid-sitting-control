"""Offline joint-limit-aware swing attitude and spatial terminal contact65.

A bounded alternative to failed fixed-height, two-foot stepping53: retain the
left foot, lift the right foot35mm before translating60/100mm, and move COM only
to the closest point inside the stance sole's5.5mm inset. Crouch begins during
the swing, then approach the legal pelvis patch with both feet planted.
No runtime, simulation, actual actuator command or physical criterion changes.
"""
from pathlib import Path
import json,hashlib,datetime
import numpy as np
from scipy.optimize import least_squares,minimize
from scipy.spatial import ConvexHull,QhullError
from scipy.spatial.transform import Rotation,Slerp
from humanoid_sitting.shared_frame import BodyModel,apply,transform,level_rotation
from low_seat_joint_path_controller_17 import measured
from low_seat_terminal_contract_42 import terminal_posture_check
from plan_seat_load_targets import clipped_vertices
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');O=D/'single_step_spatial_screen_65'
read=lambda p:json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def smooth(u):return u*u*(3-2*u)
def main():
 O.mkdir(exist_ok=False);mp=D/'cooked_model_50/model.json';ip=D.parent/'crouch_rest_bridge_01/balanced_standing10_01/static_inertial_model.json';cp=D.parent/'contact_effort_repair_01/plane_calibration.json';src=D/'normal_landing3s_06/candidate';m=BodyModel(mp);meta=read(mp);inertia=read(ip);cal=read(cp);tr=np.load(src/'trace.npz');ss=np.load(src/'sensor_inputs.npz')['sensors'];mass=np.asarray(inertia['mass']);sides=['left','right'];ids={s:[m.names.index(s+'_'+j+'_joint') for j in ['hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch','ankle_roll']] for s in sides};legs=ids['left']+ids['right'];arms=[m.names.index(s+'_'+j+'_joint') for s in sides for j in ['shoulder_pitch','elbow_pitch']];limits={j['name']:np.deg2rad(j['limits']) for j in meta['joints'] if j['type']=='PhysicsRevoluteJoint'}
 c=2;at=int(np.argmin(abs(tr['t']-1)));q0=tr['q'][at,c].astype(float);g=ss[0,10,c,339:342];assert np.array_equal(q0[:23].astype(np.float32),ss[0,10,c,270:293]);n,d,origin,h,slope,_,_=measured(m,q0,g,tr['geometry'][at,c],[tr['estimated_root_x'][at,c],0,0],cal);R0=level_rotation(g);T0=transform(R0,origin);f0=m.fk(q0);feet0={s:(R0@f0[s+'_ankle_roll_link'][:3,3]+origin,R0@f0[s+'_ankle_roll_link'][:3,:3]) for s in sides};SR=Rotation.from_euler('y',slope,degrees=True).as_matrix();center=np.array([-.4,0,h]);C0=np.average([apply(T0@f0[b],v) for b,v in zip(inertia['body_names'],inertia['local_com'])],axis=0,weights=mass)
 def ik(root,R,feet,seed,swing_bound=0.):
  T=transform(R,root);q=seed.copy()
  for side in sides:
   ii=ids[side];pos,rot=feet[side];lo=np.array([limits[m.names[j]][0]+.031 for j in ii]);hi=np.array([limits[m.names[j]][1]-.031 for j in ii])
   free=side=='right' and swing_bound>1e-7
   if free:
    base_rot=feet0['right'][1]
    def residual(a):
     qq=q.copy();qq[ii]=a[:6];F=T@m.fk(qq)[side+'_ankle_roll_link'];desired=base_rot@Rotation.from_euler('xy',a[6:]).as_matrix();return np.r_[F[:3,3]-pos,.2*Rotation.from_matrix(desired.T@F[:3,:3]).as_rotvec(),.002*a[6:]]
    fit=least_squares(residual,np.r_[np.clip(q[ii],lo,hi),[0.,0.]],bounds=(np.r_[lo,[-swing_bound,-swing_bound]],np.r_[hi,[swing_bound,swing_bound]]),max_nfev=80,xtol=1e-9,ftol=1e-9,gtol=1e-9);q[ii]=fit.x[:6];feet[side]=(pos,base_rot@Rotation.from_euler('xy',fit.x[6:]).as_matrix())
   else:
    def residual(a):
     qq=q.copy();qq[ii]=a;F=T@m.fk(qq)[side+'_ankle_roll_link'];return np.r_[F[:3,3]-pos,.2*Rotation.from_matrix(rot.T@F[:3,:3]).as_rotvec()]
    fit=least_squares(residual,np.clip(q[ii],lo,hi),bounds=(lo,hi),max_nfev=45,xtol=1e-9,ftol=1e-9,gtol=1e-9);q[ii]=fit.x
  return q
 def metrics(root,R,q,feet,stance):
  T=transform(R,root);fk=m.fk(q);C=np.average([apply(T@fk[b],v) for b,v in zip(inertia['body_names'],inertia['local_com'])],axis=0,weights=mass);foot={s:apply(T@fk[s+'_ankle_roll_link'],m.vertices[s+'_ankle_roll_link']) for s in sides};st=np.concatenate([foot[s] for s in stance]);sole=st[st[:,2]<st[:,2].min()+.003]
  try:H=ConvexHull(sole[:,:2]).equations;cop=float(np.min(-(H[:,:2]@C[:2]+H[:,2])));deg=False
  except QhullError:cop=-1.;deg=True
  pelvis=apply(T,m.vertices['pelvis']);dep=pelvis@n+d;patch=dep<dep.min()+.002;loc=(pelvis-center)@SR;pm=float(min(.25-abs(loc[patch,0]).max(),.45-abs(loc[patch,1]).max()));clear=[];floor=[]
  for b,v in m.vertices.items():
   w=apply(T@fk[b],v)
   if b not in ['left_ankle_roll_link','right_ankle_roll_link']:floor.append(float(w[:,2].min()))
   if b in ['pelvis','left_hip_yaw_link','right_hip_yaw_link','torso_link','logo_link','left_ankle_roll_link','right_ankle_roll_link']:continue
   v=(w-center)@SR;v[:,0]-=.4;v=clipped_vertices(v)
   if len(v):clear.append((float(v[:,2].min()),b))
  pe=max(np.linalg.norm((T@fk[s+'_ankle_roll_link'])[:3,3]-feet[s][0]) for s in sides);re=max(np.linalg.norm(Rotation.from_matrix(feet[s][1].T@(T@fk[s+'_ankle_roll_link'])[:3,:3]).as_rotvec()) for s in sides);pose=np.r_[root,Rotation.from_matrix(R).as_quat()];term=terminal_posture_check(pose);mi=min(clear) if clear else (1.,'none_over_finite_seat')
  out=dict(root=pose.tolist(),q=q.tolist(),com=C.tolist(),foot_targets={s:dict(position=feet[s][0].tolist(),rotation=feet[s][1].tolist()) for s in sides},stance=stance,clearance_m=mi[0],nearest_body=mi[1],pelvis_gap_m=float(dep.min()),patch_margin_m=pm,foot_cop_margin_m=cop,support_degenerate=deg,floor_clearance_m=min(floor),foot_position_error_m=float(pe),foot_orientation_error_rad=float(re),foot_min_z={s:float(v[:,2].min()) for s,v in foot.items()},terminal=term)
  out['other_model_passed']=bool(pe<=.003 and re<=np.deg2rad(.5) and cop>=.005 and min(floor)>=.005 and term['passed'] and (dep.min()>=.02 or pm>=.015) and dep.min()>=-1e-8)
  out['model_pose_passed']=bool(out['other_model_passed'] and mi[0]>=.003);return out
 results=[]
 for length in [.10]:
  endfeet={s:(p.copy(),r.copy()) for s,(p,r) in feet0.items()};endfeet['right'][0][0]-=length;seed=q0.copy();seed[arms]-=.35;cache={};history=[]
  def endpoint(x):
   key=tuple(float(v) for v in x)
   if key in cache:return cache[key]
   px,py,roll,pitch,yaw=key;R=Rotation.from_euler('xyz',[roll,pitch,yaw]).as_matrix();z=(.001-min((m.vertices['pelvis']@R.T)@n)-n[0]*px-n[1]*py-d)/n[2];root=np.array([px,py,z]);q=ik(root,R,endfeet,seed);a=metrics(root,R,q,endfeet,sides);a['parameters']=key;cache[key]=a;history.append(a);return a
  def cons(x):
   a=endpoint(x);return np.array([(a['foot_cop_margin_m']-.0055)*100,(a['patch_margin_m']-.0155)*100,(.0029-a['foot_position_error_m'])*100,(np.deg2rad(.49)-a['foot_orientation_error_rad'])*10,(a['floor_clearance_m']-.0055)*100,(38-a['terminal']['model_tilt_deg'])*.1])
  searches=[]
  for start in [(-.265,-.06,0.,np.deg2rad(22.5),0.),(-.265,-.08,-.03,np.deg2rad(22.5),0.),(-.24,-.08,.03,np.deg2rad(25),.1)]:
   opt=minimize(lambda x:-100*endpoint(x)['clearance_m'],start,method='SLSQP',bounds=[(-.30,-.1),(-.12,.06),(-np.deg2rad(12),np.deg2rad(12)),(np.deg2rad(5),np.deg2rad(38)),(-np.deg2rad(20),np.deg2rad(20))],constraints=[dict(type='ineq',fun=cons)],options=dict(maxiter=60,ftol=1e-8,eps=2e-5));searches.append(dict(start=list(start),success=bool(opt.success),message=str(opt.message),endpoint=endpoint(opt.x)));print('endpoint',length,opt.message,flush=True)
  eligible=[a for a in history if a['other_model_passed']];end=max(eligible,key=lambda a:a['clearance_m']) if eligible else None
  if end is None:results.append(dict(length_m=length,case=c,endpoint=None,searches=searches,path=[],whole_path_screen_passed=False));continue
  # Closest5.5mm-inset stance point, rather than the previously overrestrictive sole center.
  lf=apply(T0@f0['left_ankle_roll_link'],m.vertices['left_ankle_roll_link']);sole=lf[lf[:,2]<lf[:,2].min()+.003];H=ConvexHull(sole[:,:2]).equations;proj=minimize(lambda xy:float(np.sum((xy-C0[:2])**2)),sole[:,:2].mean(0),constraints=[dict(type='ineq',fun=lambda xy:-(H[:,:2]@xy+H[:,2])-.0055)],method='SLSQP',options=dict(ftol=1e-12,maxiter=50));assert proj.success;supportxy=proj.x
  endpoint_pose=np.array(end['root']);endpointC=np.array(end['com']);endR=Rotation.from_quat(endpoint_pose[3:]);interp=Slerp([0,1],Rotation.concatenate([Rotation.from_matrix(R0),endR]));path=[];previous=q0.copy();previous_root=origin.copy()
  spec=[]
  for u in np.linspace(0,1,17):spec.append(('shift',float(u),0.,C0[:2]+smooth(u)*(supportxy-C0[:2])))
  for u in np.linspace(0,1,49)[1:]:spec.append(('swing',float(u),.4*smooth(u),supportxy))
  for u in np.linspace(0,1,41)[1:]:spec.append(('approach',float(u),.4+.6*smooth(u),supportxy+smooth(u)*(endpointC[:2]-supportxy)))
  for k,(phase,u,f,xy) in enumerate(spec):
   feet={s:(p.copy(),r.copy()) for s,(p,r) in feet0.items()};stance=sides
   if phase=='swing':
    lift=.035*smooth(u/.2) if u<.2 else (.035*smooth((1-u)/.2) if u>.8 else .035);travel=smooth(np.clip((u-.2)/.6,0,1));feet['right'][0][0]-=length*travel;feet['right'][0][2]+=lift;stance=['left'] if 0<u<1 else sides
   elif phase=='approach':feet=endfeet
   R=interp(float(f)).as_matrix();root=previous_root.copy();root[2]=origin[2]+f*(endpoint_pose[2]-origin[2]);seed=previous.copy();seed[arms]=q0[arms]-.35*f
   for _ in range(14):
    bound=np.deg2rad(5)*np.sin(np.pi*u)**2 if phase=='swing' else 0.;q=ik(root,R,feet,seed,swing_bound=bound);a=metrics(root,R,q,feet,stance);err=np.asarray(xy)-np.asarray(a['com'][:2]);
    if np.linalg.norm(err)<1e-7:break
    root[:2]+=.8*err;seed=q
   a.update(phase=phase,u=u,swing_attitude_bound_rad=float(bound),approach_fraction=f,com_target_xy=np.asarray(xy).tolist(),com_target_error_m=float(np.linalg.norm(np.asarray(xy)-a['com'][:2])));path.append(a);previous=q;previous_root=root;assert np.isfinite(q).all()
  Q=np.array([a['q'] for a in path]);arc=float(np.max(abs(np.diff(Q,axis=0)),axis=1).sum());failed=[i for i,a in enumerate(path) if not a['model_pose_passed']];movements=[]
  for before,after in zip(path,path[1:]):
   p=np.array(before['foot_targets']['right']['position']);q=np.array(after['foot_targets']['right']['position'])
   if np.linalg.norm((q-p)[:2])>1e-8:movements.append(min(p[2],q[2])-feet0['right'][0][2])
  row=dict(length_m=length,case=c,endpoint=end,searches=searches,endpoint_history=history,stance_target_xy=supportxy.tolist(),path=path,failed_knots=len(failed),first_failed_knot=failed[0] if failed else None,min_clearance_m=min(a['clearance_m'] for a in path),min_cop_margin_m=min(a['foot_cop_margin_m'] for a in path),max_foot_position_error_m=max(a['foot_position_error_m'] for a in path),max_foot_orientation_error_rad=max(a['foot_orientation_error_rad'] for a in path),minimum_swing_target_lift_during_horizontal_motion_m=min(movements),minimum_actual_model_swing_sole_height_m=min(a['foot_min_z']['right'] for a in path if a['phase']=='swing' and .2<a['u']<.8),pose_arc_rad=arc,pose_only_necessary_time_at_existing_cap_s=arc/.15,first_to_last_pose_delta_rad=float(np.max(abs(Q[-1]-Q[0]))),whole_path_screen_passed=bool(not failed and arc/.15<=13.5),actual_command_rate_checked=False,physical_admission=False)
  results.append(row);(O/f'case2_spatial_step{int(length*1000)}.json').write_text(json.dumps(row,indent=2));print(json.dumps({k:v for k,v in row.items() if k not in ['path','endpoint','searches']}),flush=True)
 paths=[Path(__file__),mp,ip,cp,src/'trace.npz',src/'sensor_inputs.npz',Path('scripts/low_seat_terminal_contract_42.py')];result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='OFFLINE revision65 complete kinematic contact sequence for deepest development case only. Reuse the100mm right rear step,35mm lift and left sole closest5.5mm inset support; allow unloaded swing-foot roll/pitch within5deg times sin(pi*u)^2, optimized jointly with IK, exactly flat again at landing. Full-foot target tracking tolerance unchanged. Spatial endpoint search x[-.30,-.10],y[-.12,.06],roll±12deg,yaw±20deg,pitch5–38deg with totaltilt<=38deg; three bounded starts60iterations. Addresses recorded ankle-pitch/roll limits and terminal housing clearance, not a blind gain/rate change. Coupled40percent crouch during swing unchanged. Corrected cooked collision/frozen inertial/current1s sensor model, fixed nonleg nominal poses except previously tested-.35rad arms. No dragging/teleporting/runtime or asset change. Static COM and pose-rate lower bound only; actual command feedforward/rates, dynamics, force feasibility, self-collision and sensed landing NOT certified. Endpoint-only outcomes never authorize physics. Physical criteria unchanged; local search failure is not impossibility.',source_input_sha256={str(p.resolve()):sha(p) for p in paths},physics_ticks=0,optimizer_updates=0,holdout_used=False,physical_admission=False,rows=results);(O/'results.json').write_text(json.dumps(result,indent=2));(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2))
if __name__=='__main__':main()
