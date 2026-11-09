"""Offline coupled crouch/stance-transfer contact path67; no physical execution.

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
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');O=D/'coupled_step_timing_screen_67'
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

 previous_result=D/'single_step_spatial_screen_65/case2_spatial_step100.json';prior=read(previous_result);end=prior['endpoint'];length=.10;endfeet={s:(p.copy(),r.copy()) for s,(p,r) in feet0.items()};endfeet['right'][0][0]-=length
 supportxy=np.asarray(prior['stance_target_xy']);endpoint_pose=np.array(end['root']);endpointC=np.array(end['com']);endR=Rotation.from_quat(endpoint_pose[3:]);interp=Slerp([0,1],Rotation.concatenate([Rotation.from_matrix(R0),endR]));hist=[];cache={}
 def trajectory(x,dense=False):
  initial,landing=map(float,x);key=(initial,landing,dense)
  if key in cache:return cache[key]
  path=[];previous=q0.copy();previous_root=origin.copy();spec=[]
  for u in np.linspace(0,1,17 if dense else 9):spec.append(('shift',float(u),initial*smooth(u),C0[:2]+smooth(u)*(supportxy-C0[:2])))
  for u in np.linspace(0,1,49 if dense else 25)[1:]:spec.append(('swing',float(u),initial+(landing-initial)*smooth(u),supportxy))
  for u in np.linspace(0,1,41 if dense else 21)[1:]:spec.append(('approach',float(u),landing+(1-landing)*smooth(u),supportxy+smooth(u)*(endpointC[:2]-supportxy)))
  for phase,u,f,xy in spec:
   feet={s:(p.copy(),r.copy()) for s,(p,r) in feet0.items()};stance=sides;bound=0.
   if phase=='swing':
    lift=.035*smooth(u/.2) if u<.2 else (.035*smooth((1-u)/.2) if u>.8 else .035);travel=smooth(np.clip((u-.2)/.6,0,1));feet['right'][0][0]-=length*travel;feet['right'][0][2]+=lift;stance=['left'] if 0<u<1 else sides
    bound=np.deg2rad(8)*min(smooth(min(u/.2,1)),smooth(min((1-u)/.2,1)))
   elif phase=='approach':feet=endfeet
   R=interp(float(f)).as_matrix();root=previous_root.copy();root[2]=origin[2]+f*(endpoint_pose[2]-origin[2]);seed=previous.copy();seed[arms]=q0[arms]-.35*f
   for _ in range(14):
    q=ik(root,R,feet,seed,swing_bound=bound);a=metrics(root,R,q,feet,stance);err=np.asarray(xy)-np.asarray(a['com'][:2])
    if np.linalg.norm(err)<1e-7:break
    root[:2]+=.8*err;seed=q
   a.update(phase=phase,u=u,swing_attitude_bound_rad=float(bound),approach_fraction=f,com_target_xy=np.asarray(xy).tolist(),com_target_error_m=float(np.linalg.norm(np.asarray(xy)-a['com'][:2])));path.append(a);previous=q;previous_root=root
  Q=np.array([a['q'] for a in path]);arc=float(np.max(abs(np.diff(Q,axis=0)),axis=1).sum());failed=[i for i,a in enumerate(path) if not a['model_pose_passed']];moves=[]
  for a,b in zip(path,path[1:]):
   p=np.array(a['foot_targets']['right']['position']);v=np.array(b['foot_targets']['right']['position'])
   if np.linalg.norm((p-v)[:2])>1e-8:moves.append(min(p[2],v[2])-feet0['right'][0][2])
  pe=max(a['foot_position_error_m'] for a in path);re=max(a['foot_orientation_error_rad'] for a in path);clear=min(a['clearance_m'] for a in path);cop=min(a['foot_cop_margin_m'] for a in path);swingheight=min(a['foot_min_z']['right'] for a in path if a['phase']=='swing' and .2<a['u']<.8)
  penalty=1000*max(pe-.003,0)+100*max(re-np.deg2rad(.5),0)+1000*max(.003-clear,0)+1000*max(.005-cop,0)+1000*max(.005-swingheight,0)
  result=dict(parameters=[initial,landing],knots=len(path),path=path,failed_knots=failed,pose_arc_rad=arc,nominal_pose_schedule_at_0_15_rad_s=arc/.15,min_clearance_m=clear,min_cop_margin_m=cop,max_foot_position_error_m=pe,max_foot_orientation_error_rad=re,min_actual_swing_sole_height_m=swingheight,minimum_target_lift_while_horizontal_m=min(moves),objective=arc/.15+penalty,model_geometry_passed=bool(not failed and swingheight>=.005),nominal_pose_schedule_passed=bool(arc/.15<=13.5),actual_command_rate_checked=False,physical_admission=False)
  cache[key]=result
  if not dense:
   hist.append({k:v for k,v in result.items() if k!='path'});print(json.dumps(hist[-1]),flush=True)
  return result
 # Bounded two-variable planning search: move part of the required crouch into
 # double-support load transfer, and choose the landing crouch fraction.
 # No control gains, material, physical criteria, rate cap or arm pose sweep.
 bounds=[(.05,.25),(.35,.60)]
 opt=minimize(lambda x:trajectory(x)['objective'],[.15,.45],method='Powell',bounds=bounds,options=dict(maxiter=3,maxfev=24,xtol=.008,ftol=.001))
 best=min(cache.values(),key=lambda a:a['objective']);dense=trajectory(best['parameters'],True)
 paths=[Path(__file__),previous_result,D/'single_step_spatial_screen_65/results.json',mp,ip,cp,src/'trace.npz',src/'sensor_inputs.npz',Path('scripts/low_seat_terminal_contract_42.py')]
 result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='Offline constrained planning revision67. Reuses the saved corrected-model spatial endpoint65 without rerunning endpoint search. Right100mm rear step/35mm lift; left stance unchanged. Two bounded crouch fractions: during initial double-support COM transfer .05–.25, at landing .35–.60, one Powell search at most24 evaluations. This overlaps previously sequential joint motion. Planned unloaded foot attitude may roll/pitch up to8deg, with cubic rise during first20percent and return during last20percent; exactly flat at landing. Addresses documented bound saturation in65; target tracking tolerance, joint reserve, static support/clearance and original physical criteria unchanged. No runtime, simulator, optimizer, holdout or promotion. Pose schedule is not an actual PD-command certificate; all dynamic, force, command and self-collision checks remain necessary.',search=dict(success=bool(opt.success),message=str(opt.message),evaluations=int(opt.nfev),bounds=bounds,history=hist),endpoint=end,dense=dense,source_input_sha256={str(p.resolve()):sha(p) for p in paths},physics_ticks=0,optimizer_updates=0,physical_admission=False)
 (O/'results.json').write_text(json.dumps(result,indent=2));(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2));print(json.dumps(dict(dense={k:v for k,v in dense.items() if k!='path'})),flush=True)
if __name__=='__main__':main()
