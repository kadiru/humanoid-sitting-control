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
from scipy.optimize import least_squares,minimize,linprog
from scipy.spatial import ConvexHull,QhullError
from scipy.spatial.transform import Rotation,Slerp
from humanoid_sitting.shared_frame import BodyModel,apply,transform,level_rotation
from low_seat_joint_path_controller_17 import measured
from low_seat_terminal_contract_42 import terminal_posture_check
from plan_seat_load_targets import clipped_vertices
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');O=D/'landing_ankle_path_95'
read=lambda p:json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def smooth(u):return u*u*(3-2*u)
def main():
 O.mkdir(exist_ok=False);mp=D/'cooked_model_50/model.json';ip=D.parent/'crouch_rest_bridge_01/balanced_standing10_01/static_inertial_model.json';cp=D.parent/'contact_effort_repair_01/plane_calibration.json';src=D/'normal_landing3s_06/candidate';m=BodyModel(mp);meta=read(mp);inertia=read(ip);cal=read(cp);tr=np.load(src/'trace.npz');ss=np.load(src/'sensor_inputs.npz')['sensors'];mass=np.asarray(inertia['mass']);sides=['left','right'];ids={s:[m.names.index(s+'_'+j+'_joint') for j in ['hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch','ankle_roll']] for s in sides};legs=ids['left']+ids['right'];arms=[m.names.index(s+'_'+j+'_joint') for s in sides for j in ['shoulder_pitch','elbow_pitch']];limits={j['name']:np.deg2rad(j['limits']) for j in meta['joints'] if j['type']=='PhysicsRevoluteJoint'}
 c=2;at=int(np.argmin(abs(tr['t']-1)));q0=tr['q'][at,c].astype(float);g=ss[0,10,c,339:342];assert np.array_equal(q0[:23].astype(np.float32),ss[0,10,c,270:293]);n,d,origin,h,slope,_,_=measured(m,q0,g,tr['geometry'][at,c],[tr['estimated_root_x'][at,c],0,0],cal);R0=level_rotation(g);T0=transform(R0,origin);f0=m.fk(q0);feet0={s:(R0@f0[s+'_ankle_roll_link'][:3,3]+origin,R0@f0[s+'_ankle_roll_link'][:3,:3]) for s in sides};SR=Rotation.from_euler('y',slope,degrees=True).as_matrix();center=np.array([-.4,0,h]);C0=np.average([apply(T0@f0[b],v) for b,v in zip(inertia['body_names'],inertia['local_com'])],axis=0,weights=mass)
 def ik(root,R,feet,seed,swing_bound=0.,swing_base_rotation=None):
  T=transform(R,root);q=seed.copy()
  for side in sides:
   ii=ids[side];pos,rot=feet[side];lo=np.array([limits[m.names[j]][0]+.031 for j in ii]);hi=np.array([limits[m.names[j]][1]-.031 for j in ii])
   free=side=='right' and swing_bound>1e-7
   if free:
    base_rot=feet0['right'][1] if swing_base_rotation is None else swing_base_rotation
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

 ine=inertia;W=sum(ine['mass'])*9.81;names=m.names;moving=legs+arms;kp=np.array([200,150,150,200,160,160]*2+[40]*4,dtype=float);effort=np.array([300,300,300,300,20,20]*2+[300]*4,dtype=float);byname={j['name']:j for j in m.joints};children={}
 for j in m.joints:
  if j['body0']:children.setdefault(j['body0'][0],[]).append(j['body1'][0])
 def descend(b):
  r={b}
  for c in children.get(b,[]):r|=descend(c)
  return r
 sub={name:descend(byname[name]['body1'][0]) for name in names};command0=tr['command'][at,c].astype(float)
 def state(e):
  q=np.asarray(e['q']);root=np.asarray(e['root']);T=transform(Rotation.from_quat(root[3:]).as_matrix(),root[:3]);fk=m.fk(q);C=np.average([apply(T@fk[b],v) for b,v in zip(ine['body_names'],ine['local_com'])],axis=0,weights=ine['mass']);pts=[];ss=[];local=[]
  for s in sides:
   F=T@fk[s+'_ankle_roll_link'];v=apply(F,m.vertices[s+'_ankle_roll_link']);sole=v[v[:,2]<v[:,2].min()+.003]
   try:sole=sole[ConvexHull(sole[:,:2]).vertices]
   except QhullError:pass
   for p in sole:pts.append(p);ss.append(s);local.append(np.linalg.inv(F)@np.r_[p,1.])
  return q,T,fk,C,np.array(pts),ss,np.array(local)
 def allocation(st,right_fraction=None,seat_normal_fraction=0.):
  q,T,fk,C,P,ss,local=st;v=apply(T,m.vertices['pelvis']);dep=v@n+d;seat=v[dep<=dep.min()+1e-7].mean(0);seat-=n*(seat@n+d);sv=seat_normal_fraction/n[2];assert seat_normal_fraction==0. or abs(dep.min())<1e-8;A=np.stack([np.ones(len(P)),P[:,0],P[:,1]]);b=np.r_[1.-sv,C[:2]-sv*seat[:2]]
  if right_fraction is not None:A=np.vstack([A,np.array([s=='right' for s in ss],float)]);b=np.r_[b,right_fraction]
  lp=linprog(np.zeros(len(P)),A_eq=A,b_eq=b,bounds=(0,None),method='highs')
  if not lp.success:return None,dict(passed=False,message=lp.message)
  op=minimize(lambda w:.5*np.dot(w,w),lp.x,jac=lambda w:w,method='SLSQP',constraints=[dict(type='eq',fun=lambda w:A@w-b,jac=lambda w:A)],bounds=[(0,None)]*len(P),options=dict(ftol=1e-12,maxiter=100));w=op.x;error=float(max(abs(A@w-b)));return w,dict(passed=bool(op.success and error<1e-8),max_normalized_equilibrium_residual=error,message=op.message,right_fraction=float(sum(w[i] for i,s in enumerate(ss) if s=='right')),minimum_corner_fraction=float(min(w)))
 def torques(st,w):
  q,T,fk,C,P,ss,local=st;tau=[];finite=[]
  for j in moving:
   name=names[j];joint=byname[name];J=T@fk[joint['body0'][0]]@m.frames[name][0];axis=J[:3,'XYZ'.index(joint['axis'])];moment=np.zeros(3)
   for b,mm,cc in zip(ine['body_names'],ine['mass'],ine['local_com']):
    if b in sub[name]:moment+=np.cross(apply(T@fk[b],cc)-J[:3,3],[0,0,-mm*9.81])
   for i,s in enumerate(ss):
    if s+'_ankle_roll_link' in sub[name]:moment+=np.cross(P[i]-J[:3,3],[0,0,W*w[i]])
   tau.append(float(-axis@moment))
  return np.array(tau)

 previous_result=D/'contact_stance_screen_89/results.json';prior=read(previous_result);end=prior['best'];assert end['all_endpoint_model_checks_passed'];length=end['parameters'][5];landing_yaw=end['parameters'][6];endfeet={s:(p.copy(),r.copy()) for s,(p,r) in feet0.items()};endfeet['right']=(endfeet['right'][0]+np.array([-length,0.,0.]),Rotation.from_euler('z',landing_yaw).as_matrix()@feet0['right'][1])
 old69=read(D/'coupled_step_refined_69/results.json');initial=state(old69['dense']['path'][0]);w0,a0=allocation(initial);assert a0['passed'];tau0=torques(initial,w0);rend_w,rend_a=allocation(state(end));assert rend_a['passed'];r0=a0['right_fraction'];rend=rend_a['right_fraction'];fixed=[j for j in range(len(names)) if j not in moving];fingers=[i for i,v in enumerate(names) if any('_'+n+'_joint' in v for n in ['zero','one','two','three','four','five','six'])];assert len(fingers)==14
 supportxy=np.asarray(old69['dense']['path'][16]['com_target_xy']);endpoint_pose=np.array(end['root']);endpointC=np.array(end['com']);endR=Rotation.from_quat(endpoint_pose[3:]);interp=Slerp([0,1],Rotation.concatenate([Rotation.from_matrix(R0),endR]));hist=[];cache={}
 def trajectory(x,dense=False):
  initial,landing,lean,pitch_bump=map(float,x);key=(initial,landing,lean,pitch_bump,dense)
  if key in cache:return cache[key]
  path=[];previous=q0.copy();previous_root=origin.copy();spec=[]
  for u in np.linspace(0,1,17 if dense else 9):spec.append(('shift',float(u),initial*smooth(u),C0[:2]+smooth(u)*(supportxy-C0[:2])))
  for u in np.linspace(0,1,49 if dense else 25)[1:]:spec.append(('swing',float(u),initial+(landing-initial)*smooth(u),supportxy))
  for u in np.linspace(0,1,41 if dense else 21)[1:]:spec.append(('approach',float(u),landing+(1-landing)*smooth(u),supportxy+smooth(u)*(endpointC[:2]-supportxy)))
  for u in np.linspace(0,1,11)[1:]:spec.append(('load',float(u),1.,endpointC[:2]))
  for phase,u,f,xy in spec:
   feet={s:(p.copy(),r.copy()) for s,(p,r) in feet0.items()};stance=sides;bound=0.
   if phase=='swing':
    lift=.035*smooth(u/.2) if u<.2 else (.035*smooth((1-u)/.2) if u>.8 else .035);travel=smooth(np.clip((u-.2)/.6,0,1));feet['right'][0][0]-=length*travel;feet['right'][0][2]+=lift;feet['right']=(feet['right'][0],Rotation.from_euler('z',landing_yaw*travel).as_matrix()@feet0['right'][1]);stance=['left'] if 0<u<1 else sides
    bound=np.deg2rad(8)*min(smooth(min(u/.2,1)),smooth(min((1-u)/.2,1)))
   elif phase in ['approach','load']:feet=endfeet
   lean_fraction=smooth(u) if phase=='shift' else (1. if phase=='swing' else 1-smooth(u) if phase=='approach' else 0.);R=Rotation.from_rotvec([0,pitch_bump*lean_fraction,0]).as_matrix()@Rotation.from_rotvec([lean*lean_fraction,0,0]).as_matrix()@interp(float(f)).as_matrix();root=previous_root.copy();root[2]=origin[2]+f*(endpoint_pose[2]-origin[2]);seed=previous.copy();seed[arms]=q0[arms]-.35*f
   for _ in range(14):
    q=ik(root,R,feet,seed,swing_bound=bound,swing_base_rotation=Rotation.from_euler('z',landing_yaw*travel).as_matrix()@feet0['right'][1] if phase=='swing' else None);a=metrics(root,R,q,feet,stance);err=np.asarray(xy)-np.asarray(a['com'][:2])
    if np.linalg.norm(err)<1e-7:break
    root[:2]+=.8*err;seed=q
   a.update(phase=phase,u=u,swing_attitude_bound_rad=float(bound),approach_fraction=f,com_target_xy=np.asarray(xy).tolist(),com_target_error_m=float(np.linalg.norm(np.asarray(xy)-a['com'][:2])));path.append(a);previous=q;previous_root=root
  Q=np.array([a['q'] for a in path]);arc=float(np.max(abs(np.diff(Q,axis=0)),axis=1).sum());failed=[i for i,a in enumerate(path) if not a['model_pose_passed']];moves=[]
  for a,b in zip(path,path[1:]):
   p=np.array(a['foot_targets']['right']['position']);v=np.array(b['foot_targets']['right']['position'])
   if np.linalg.norm((p-v)[:2])>1e-8:moves.append(min(p[2],v[2])-feet0['right'][0][2])
  pe=max(a['foot_position_error_m'] for a in path);re=max(a['foot_orientation_error_rad'] for a in path);clear=min(a['clearance_m'] for a in path);cop=min(a['foot_cop_margin_m'] for a in path);swingheight=min(a['foot_min_z']['right'] for a in path if a['phase']=='swing' and .2<a['u']<.8)
  commands=[];nodes=[];margins=[];utils=[];allocation_failed=[]
  for k,e in enumerate(path):
   u=e['u'];right=(1-smooth(u))*r0 if e['phase']=='shift' else (0. if e['phase']=='swing' else smooth(u)*rend);st=state(e);normal=.1*smooth(u) if e['phase']=='load' else 0.;w,al=allocation(st,None if normal>0 else right,normal);node=dict(knot=k,allocation=al,planned_right_load_fraction=al.get('right_fraction') if normal>0 else right,assumed_seat_normal_fraction=normal)
   if w is None:allocation_failed.append(k)
   else:
    tau=torques(st,w);cmd=command0.copy();cmd[moving]+=st[0][moving]-q0[moving]+(tau-tau0)/kp
    if k==0:cmd=command0.copy()
    assert np.array_equal(cmd[fixed],command0[fixed]) and np.array_equal(cmd[fingers],command0[fingers]);cm=float(min(min(cmd[j]-limits[names[j]][0],limits[names[j]][1]-cmd[j]) for j in moving));eu=float(max(abs(tau)/effort));margins.append(cm);utils.append(eu);commands.append(cmd);node.update(command=cmd.tolist(),planned_static_torque_Nm=tau.tolist(),corner_fractions=w.tolist(),corner_world_points=st[4].tolist(),corner_sides=st[5],min_command_margin_rad=cm,maximum_effort_utilization=eu)
   nodes.append(node)
  command_arc=float(np.max(abs(np.diff(commands,axis=0)),axis=1).sum()) if not allocation_failed else None;duration=command_arc/.15 if command_arc is not None else 1000.;cm=min(margins) if margins else -1.;eu=max(utils) if utils else 100.
  penalty=10000*max(pe-.003,0)+1000*max(re-np.deg2rad(.5),0)+10000*max(.003-clear,0)+10000*max(.005-cop,0)+10000*max(.005-swingheight,0)+1000*max(.03-cm,0)+1000*max(eu-1,0)+1000*len(allocation_failed)
  result=dict(parameters=[initial,landing,lean,pitch_bump],knots=len(path),path=path,command_nodes=nodes,failed_knots=failed,pose_arc_rad=arc,nominal_pose_schedule_at_0_15_rad_s=arc/.15,min_clearance_m=clear,min_cop_margin_m=cop,max_foot_position_error_m=pe,max_foot_orientation_error_rad=re,min_actual_swing_sole_height_m=swingheight,minimum_target_lift_while_horizontal_m=min(moves),min_command_margin_rad=cm,maximum_effort_utilization=eu,force_allocation_failures=allocation_failed,command_arc_rad=command_arc,minimum_time_for_this_command_polyline_at015_s=duration,objective=duration+penalty,model_geometry_passed=bool(not failed and swingheight>=.005),model_command_passed=bool(not allocation_failed and cm>=.03 and eu<=1),command_schedule_passed=bool(duration<=13.5),model_command_polyline_checked=True,actual_command_rate_checked=False,physical_admission=False)
  cache[key]=result
  if not dense:
   hist.append({k:v for k,v in result.items() if k not in ['path','command_nodes']});print(json.dumps(hist[-1]),flush=True)
  return result
 previous_path=D/'contact_path_screen_v2_91/results.json';old91=read(previous_path);assert read(D/'contact_path_audit_v2_92/independent_audit.json')['passed'];bounds=[(.05,.25),(.30,.55),(-np.deg2rad(12),0.),(0.,np.deg2rad(15))];limit=40
 class BudgetReached(Exception):pass
 def evaluate(x):
  key=tuple(map(float,x))+(False,)
  if key not in cache and len(hist)>=limit:raise BudgetReached('40 unique coarse path evaluations completed')
  return trajectory(x)
 def constraints(x):
  a=evaluate(x);p=a['path'];return np.array([(a['min_command_margin_rad']-.03001)*10,1-a['maximum_effort_utilization'],(.00299-a['max_foot_position_error_m'])*100,(np.deg2rad(.499)-a['max_foot_orientation_error_rad'])*10,(a['min_clearance_m']-.003)*100,(a['min_cop_margin_m']-.005)*100,(a['min_actual_swing_sole_height_m']-.005)*100,(min(e['floor_clearance_m'] for e in p)-.005)*100,min(e['pelvis_gap_m'] for e in p)*100,(38-max(e['terminal']['model_tilt_deg'] for e in p))*.1,0. if not a['force_allocation_failures'] else -1.])
 try:
  fit=minimize(lambda x:evaluate(x)['minimum_time_for_this_command_polyline_at015_s'],old91['dense']['parameters']+[np.deg2rad(10)],method='SLSQP',bounds=bounds,constraints=[dict(type='ineq',fun=constraints)],options=dict(maxiter=10,ftol=1e-6,eps=.001));search=dict(success=bool(fit.success),message=str(fit.message))
 except BudgetReached as ex:search=dict(success=False,message=str(ex))
 eligible=[a for a in cache.values() if a['model_geometry_passed'] and a['model_command_passed']];best=min(eligible,key=lambda a:a['minimum_time_for_this_command_polyline_at015_s']) if eligible else min(cache.values(),key=lambda a:a['objective']);dense=trajectory(best['parameters'],True)
 files=[Path(__file__).resolve(),previous_result,previous_path,D/'contact_path_audit_v2_92/independent_audit.json',D/'progress_report_93/path_diagnosis.json',D/'coupled_step_refined_69/results.json',mp,ip,cp,src/'trace.npz',src/'sensor_inputs.npz',Path('scripts/low_seat_terminal_contract_42.py')]
 result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='Bounded OFFLINE landing-ankle trajectory optimization95. Audited91v2 identifies right ankle pitch at its existing-.649rad planning reserve as the foot returns flat, with5.49mm/2.37degree foot error. Adds a smooth0–15degree temporary world-pitch shaping term during shift/swing, released during approach, to change pelvis position under the unchanged COM/support constraint; terminal pose89 is exactly unchanged. This is a diagnosed whole-path planning change, not a gain/rate/arm or physical-criterion sweep. Jointly optimize initial/landing crouch and temporary roll using prior bounds, at most40 unique63-knot evaluations/10iterations plus one115-knot dense check. Every knot retains38degree research tilt reserve/40degree physical limit,3mm housing/foot tolerances,0.5degree foot orientation,5mm support margin,.031rad IK reserve,.03rad command margin, configured efforts and0.15rad/s. Bounded swing tilt uses immutable yaw base from corrected91v2. Assumed seat load only at zero-gap contact. Actual command filtering, sensed support and dynamic behavior still require separate checks. No simulation, PPO, holdout or admission.',search=dict(**search,evaluations=len(hist),maximum_evaluations=40,bounds=bounds,history=hist),endpoint=end,dense=dense,source_input_sha256={str(p):sha(p) for p in files},physics_ticks=0,optimizer_updates=0,holdout_used=False,physical_admission=False);(O/'results.json').write_text(json.dumps(result,indent=2));(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2));print(json.dumps({k:v for k,v in dense.items() if k not in ['path','command_nodes']}),flush=True)
if __name__=='__main__':main()
