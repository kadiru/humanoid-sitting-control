"""Bounded offline contact-pose search for command/filter headroom114.
No motion, physical criteria, gains, rates or material changes.
"""
from pathlib import Path
import json,hashlib,datetime,argparse
import numpy as np
from scipy.optimize import least_squares,minimize,linprog
from scipy.spatial import ConvexHull,QhullError
from scipy.spatial.transform import Rotation,Slerp
from humanoid_sitting.shared_frame import BodyModel,apply,transform,level_rotation
from low_seat_joint_path_controller_17 import measured
from low_seat_terminal_contract_42 import terminal_posture_check
from plan_seat_load_targets import clipped_vertices
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');O=D/'fixed_foot_contact_paths_118'
read=lambda p:json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def smooth(u):return u*u*(3-2*u)
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--case',type=int,choices=[0,2,6,10],required=True);args=parser.parse_args();O=D/'fixed_foot_contact_paths_118'/f'case_{args.case}';O.parent.mkdir(exist_ok=True)
 O.mkdir(exist_ok=False);mp=D/'cooked_model_50/model.json';ip=D.parent/'crouch_rest_bridge_01/balanced_standing10_01/static_inertial_model.json';cp=D.parent/'contact_effort_repair_01/plane_calibration.json';src=D/'normal_landing3s_06/candidate';m=BodyModel(mp);meta=read(mp);inertia=read(ip);cal=read(cp);tr=np.load(src/'trace.npz');ss=np.load(src/'sensor_inputs.npz')['sensors'];mass=np.asarray(inertia['mass']);sides=['left','right'];ids={s:[m.names.index(s+'_'+j+'_joint') for j in ['hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch','ankle_roll']] for s in sides};legs=ids['left']+ids['right'];arms=[m.names.index(s+'_'+j+'_joint') for s in sides for j in ['shoulder_pitch','elbow_pitch']];limits={j['name']:np.deg2rad(j['limits']) for j in meta['joints'] if j['type']=='PhysicsRevoluteJoint'}
 c=args.case;at=int(np.argmin(abs(tr['t']-1)));q0=tr['q'][at,c].astype(float);g=ss[0,10,c,339:342];assert np.array_equal(q0[:23].astype(np.float32),ss[0,10,c,270:293]);n,d,origin,h,slope,_,_=measured(m,q0,g,tr['geometry'][at,c],[tr['estimated_root_x'][at,c],0,0],cal);R0=level_rotation(g);T0=transform(R0,origin);f0=m.fk(q0);feet0={s:(R0@f0[s+'_ankle_roll_link'][:3,3]+origin,R0@f0[s+'_ankle_roll_link'][:3,:3]) for s in sides};SR=Rotation.from_euler('y',slope,degrees=True).as_matrix();center=np.array([-.4,0,h]);C0=np.average([apply(T0@f0[b],v) for b,v in zip(inertia['body_names'],inertia['local_com'])],axis=0,weights=mass)
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
 def allocation(st,right_fraction=None):
  q,T,fk,C,P,ss,local=st;A=np.stack([np.ones(len(P)),P[:,0],P[:,1]]);b=np.r_[1.,C[:2]]
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

 previous58=D/'spatial_contact_screen_v2_58'/f'case_{c}.json';endpoint58=read(previous58)['best'];assert endpoint58 is not None
 initial_pose=dict(q=q0.tolist(),root=np.r_[origin,Rotation.from_matrix(R0).as_quat()].tolist());initial=state(initial_pose);w0,al0=allocation(initial);assert al0['passed'];tau0=torques(initial,w0)
 feet={s:(p.copy(),r.copy()) for s,(p,r) in feet0.items()};seed=q0.copy()
 def loaded(a,normal_fraction):
  st=state(a);q,T,fk,C,P,ss,local=st;v=apply(T,m.vertices['pelvis']);dep=v@n+d;assert normal_fraction==0 or abs(dep.min())<1e-8;contact=v[dep<=dep.min()+1e-7].mean(0);contact-=n*(contact@n+d);vertical=normal_fraction/n[2];A=np.stack([np.ones(len(P)),P[:,0],P[:,1]]);b=np.r_[1-vertical,C[:2]-vertical*contact[:2]];grav=torques(st,np.zeros(len(P)));B=[]
  for j in moving:
   joint=byname[names[j]];J=T@fk[joint['body0'][0]]@m.frames[names[j]][0];axis=J[:3,'XYZ'.index(joint['axis'])];B.append([-axis@np.cross(point-J[:3,3],[0,0,W]) if side+'_ankle_roll_link' in sub[names[j]] else 0. for point,side in zip(P,ss)])
  B=np.array(B);base=command0[moving]+q[moving]-q0[moving]+(grav-tau0)/kp;M=B/kp[:,None];lo=np.array([limits[names[j]][0] for j in moving]);hi=np.array([limits[names[j]][1] for j in moving]);U=np.vstack([np.c_[M,np.ones(len(moving))],np.c_[-M,np.ones(len(moving))],np.c_[B,np.zeros(len(moving))],np.c_[-B,np.zeros(len(moving))]]);rhs=np.r_[hi-base,base-lo,effort-grav,effort+grav]
  lp=linprog(np.r_[np.zeros(len(P)),-1.],A_ub=U,b_ub=rhs,A_eq=np.c_[A,np.zeros(3)],b_eq=b,bounds=[(0,1)]*len(P)+[(-1,1)],method='highs',options=dict(primal_feasibility_tolerance=1e-10,dual_feasibility_tolerance=1e-10))
  if not lp.success:return dict(passed=False,message=lp.message,min_command_margin_rad=-1.,maximum_effort_utilization=100.)
  w=lp.x[:-1];err=max(abs(A@w-b));tau=grav+B@w;cmd=command0.copy();cmd[moving]=base+M@w;cm=float(min(np.r_[cmd[moving]-lo,hi-cmd[moving]]));eu=float(max(abs(tau)/effort));assert abs(cm-lp.x[-1])<1e-8
  return dict(passed=bool(err<1e-8 and min(w)>=-1e-10 and cm>=.03 and eu<=1),message=lp.message,normal_load_fraction=normal_fraction,vertical_seat_fraction=float(vertical),seat_contact_point=contact.tolist(),seat_force_normalized=[0.,0.,float(vertical)],required_seat_tangent_over_normal=float(np.sqrt(1-n[2]**2)/n[2]),max_normalized_equilibrium_residual=float(err),corner_fractions=w.tolist(),corner_world_points=P.tolist(),corner_sides=ss,command=cmd.tolist(),static_torque_Nm=tau.tolist(),min_command_margin_rad=cm,maximum_effort_utilization=eu,force_objective='maximize minimum actual command margin at this fixed endpoint')
 params=endpoint58['parameters'];Rend=Rotation.from_euler('xyz',params[2:5]);interp=Slerp([0,1],Rotation.concatenate([Rotation.from_matrix(R0),Rend]));gap0=float(min(apply(T0,m.vertices['pelvis'])@n+d));path=[];previous=q0.copy()
 for k in range(91):
  phase='approach' if k<=80 else 'load';u=k/80 if phase=='approach' else (k-80)/10;f=smooth(u) if phase=='approach' else 1.;normal=0. if phase=='approach' else .1*smooth(u);R=interp(f).as_matrix();root=origin.copy();root[:2]=(1-f)*origin[:2]+f*np.array(params[:2]);gap=(1-f)*gap0;root[2]=(gap-min((m.vertices['pelvis']@R.T)@n)-n[0]*root[0]-n[1]*root[1]-d)/n[2];seed=previous.copy();seed[arms]=q0[arms]-.35*f;q=ik(root,R,feet,seed);e=metrics(root,R,q,feet,sides);previous=q;e.update(parameters=list(map(float,[root[0],root[1],*Rotation.from_matrix(R).as_euler('xyz'),0.,0.])),phase=phase,u=u,approach_fraction=f,assumed_seat_normal_fraction=normal,case=c)
  load=loaded(e,normal)
  if k==0:
   assert max(abs(q-q0))<1e-6
   load.update(corner_fractions=w0.tolist(),corner_world_points=initial[4].tolist(),corner_sides=initial[5],command=command0.tolist(),static_torque_Nm=tau0.tolist(),min_command_margin_rad=float(min(min(command0[j]-limits[names[j]][0],limits[names[j]][1]-command0[j]) for j in moving)),maximum_effort_utilization=float(max(abs(tau0)/effort)),max_normalized_equilibrium_residual=al0['max_normalized_equilibrium_residual'],force_objective='exact original initial force and command')
   load['passed']=bool(load['min_command_margin_rad']>=.03 and load['maximum_effort_utilization']<=1)
  e.update(load_checks=[load],min_command_margin_rad=load['min_command_margin_rad'],maximum_effort_utilization=load['maximum_effort_utilization'],all_endpoint_model_checks_passed=bool(e['model_pose_passed'] and load['passed']));path.append(e)
 failed=[i for i,e in enumerate(path) if not e['model_pose_passed']];commands=[e['load_checks'][0].get('command') for e in path];allcmd=all(x is not None for x in commands);ct=float(np.max(abs(np.diff(np.array(commands),axis=0)),axis=1).sum()/.15) if allcmd else None
 summary=dict(case=c,geometry_passed=not failed,failed_geometry_knots=failed,knots=len(path),minimum_housing_clearance_mm=min(e['clearance_m'] for e in path)*1000,max_foot_position_error_mm=max(e['foot_position_error_m'] for e in path)*1000,max_foot_orientation_error_deg=float(np.rad2deg(max(e['foot_orientation_error_rad'] for e in path))),minimum_command_margin_rad=min(e['min_command_margin_rad'] for e in path),commands_passed=all(e['load_checks'][0]['passed'] for e in path),command_polyline_time_s=ct,time_gate_passed=bool(ct is not None and ct<=13.5),endpoint_model_passed=path[-1]['all_endpoint_model_checks_passed'],actual_filter_checked=False,load_is_measured=False,whole_sampled_path_checked=True)
 files=[Path(__file__).resolve(),mp,ip,cp,src/'trace.npz',src/'sensor_inputs.npz',previous58,Path('scripts/low_seat_terminal_contract_42.py')]
 result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='One OFFLINE complete fixed-foot approach/contact path per previously unresolved geometry, no search. Uses preserved spatial58 endpoint x/y/rotation but approaches actual zero-gap cooked pelvis contact rather than ending1mm above it. Both full foot targets stay fixed; all91samples use measured1s start and frozen inertia/arms/fingers, original full-foot,clearance,support,terminal and command limits. Smooth coupled rootxy/rotation and normal gap descent then explicitly assumed0–.1BW normal seat load only atzero gap. Each static foot-force allocation maximizes command margin; that is not realized dynamic loading. No physical/PPO/holdout, no endpoint-only launch, no filter or rate changes. Allfour geometries are included even if prior endpoint screen failed; rejected paths remain rejected.',case=c,best=path[-1],searches=[],path=path,initial_static_torque_Nm=tau0.tolist(),summary=summary,source_sha256={str(p.resolve()):sha(p) for p in files},physics_ticks=0,optimizer_updates=0,holdout_used=False,physical_admission=False)
 (O/'results.json').write_text(json.dumps(result,indent=2));(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2));print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
