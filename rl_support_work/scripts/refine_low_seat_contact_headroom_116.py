"""Bounded offline feasible trust-region contact-pose search116.
No motion, physical criteria, gains, rates or material changes.
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
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');O=D/'contact_headroom_trust_116'
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

 oldpath=D/'coupled_step_refined_69/results.json';old=read(oldpath);initial=state(old['dense']['path'][0]);w0,al0=allocation(initial);assert al0['passed'];tau0=torques(initial,w0)
 prior_path=D/'actuator_endpoint_screen_72/results.json';prior=read(prior_path);approach=prior['best'];assert approach['all_endpoint_model_checks_passed'];feet={s:(p.copy(),r.copy()) for s,(p,r) in feet0.items()};feet['right'][0][0]-=.10;seed=np.array(approach['q']);history=[];cache={}
 def loaded(a,normal_fraction):
  st=state(a);q,T,fk,C,P,ss,local=st;v=apply(T,m.vertices['pelvis']);dep=v@n+d;assert abs(dep.min())<1e-8;contact=v[dep<=dep.min()+1e-7].mean(0);contact-=n*(contact@n+d);vertical=normal_fraction/n[2];A=np.stack([np.ones(len(P)),P[:,0],P[:,1]]);b=np.r_[1-vertical,C[:2]-vertical*contact[:2]];grav=torques(st,np.zeros(len(P)));B=[]
  for j in moving:
   joint=byname[names[j]];J=T@fk[joint['body0'][0]]@m.frames[names[j]][0];axis=J[:3,'XYZ'.index(joint['axis'])];B.append([-axis@np.cross(point-J[:3,3],[0,0,W]) if side+'_ankle_roll_link' in sub[names[j]] else 0. for point,side in zip(P,ss)])
  B=np.array(B);base=command0[moving]+q[moving]-q0[moving]+(grav-tau0)/kp;M=B/kp[:,None];lo=np.array([limits[names[j]][0] for j in moving]);hi=np.array([limits[names[j]][1] for j in moving]);U=np.vstack([np.c_[M,np.ones(len(moving))],np.c_[-M,np.ones(len(moving))],np.c_[B,np.zeros(len(moving))],np.c_[-B,np.zeros(len(moving))]]);rhs=np.r_[hi-base,base-lo,effort-grav,effort+grav]
  lp=linprog(np.r_[np.zeros(len(P)),-1.],A_ub=U,b_ub=rhs,A_eq=np.c_[A,np.zeros(3)],b_eq=b,bounds=[(0,1)]*len(P)+[(-1,1)],method='highs',options=dict(primal_feasibility_tolerance=1e-10,dual_feasibility_tolerance=1e-10))
  if not lp.success:return dict(passed=False,message=lp.message,min_command_margin_rad=-1.,maximum_effort_utilization=100.)
  w=lp.x[:-1];err=max(abs(A@w-b));tau=grav+B@w;cmd=command0.copy();cmd[moving]=base+M@w;cm=float(min(np.r_[cmd[moving]-lo,hi-cmd[moving]]));eu=float(max(abs(tau)/effort));assert abs(cm-lp.x[-1])<1e-8
  return dict(passed=bool(err<1e-8 and min(w)>=-1e-10 and cm>=.03 and eu<=1),message=lp.message,normal_load_fraction=normal_fraction,vertical_seat_fraction=float(vertical),seat_contact_point=contact.tolist(),seat_force_normalized=[0.,0.,float(vertical)],required_seat_tangent_over_normal=float(np.sqrt(1-n[2]**2)/n[2]),max_normalized_equilibrium_residual=float(err),corner_fractions=w.tolist(),corner_world_points=P.tolist(),corner_sides=ss,command=cmd.tolist(),static_torque_Nm=tau.tolist(),min_command_margin_rad=cm,maximum_effort_utilization=eu,force_objective='maximize minimum actual command margin at this fixed endpoint')
 def pose(x,gap):
  px,py,roll,pitch,yaw,length,footyaw=map(float,x);feet['right']=(feet0['right'][0]+np.array([-length,0.,0.]),Rotation.from_euler('z',footyaw).as_matrix()@feet0['right'][1]);R=Rotation.from_euler('xyz',[roll,pitch,yaw]).as_matrix();z=(gap-min((m.vertices['pelvis']@R.T)@n)-n[0]*px-n[1]*py-d)/n[2];root=np.array([px,py,z]);q=ik(root,R,feet,seed);a=metrics(root,R,q,feet,sides);a['parameters']=list(map(float,x));return a
 class BudgetReached(Exception):pass
 def endpoint(x):
  key=tuple(map(float,x))
  if key in cache:return cache[key]
  if len(history)>=100:raise BudgetReached('100 unique endpoint evaluations completed')
  a=pose(x,0.);loads=[loaded(a,f) for f in [0.,.1]];a.update(load_checks=loads,min_command_margin_rad=min(v['min_command_margin_rad'] for v in loads),maximum_effort_utilization=max(v['maximum_effort_utilization'] for v in loads),all_endpoint_model_checks_passed=bool(a['model_pose_passed'] and all(v['passed'] for v in loads)));cache[key]=a;history.append(a);return a
 def cons(x):
  a=endpoint(x);return np.array([(a['foot_cop_margin_m']-.0055)*100,(a['patch_margin_m']-.0155)*100,(.0029-a['foot_position_error_m'])*100,(np.deg2rad(.49)-a['foot_orientation_error_rad'])*10,(a['floor_clearance_m']-.0055)*100,(38-a['terminal']['model_tilt_deg'])*.1,(a['min_command_margin_rad']-.0305)*10,1-a['maximum_effort_utilization'],(a['clearance_m']-.00305)*100])
 previous89=D/'contact_stance_screen_89/results.json';previous=read(previous89);prior114=D/'contact_headroom_endpoint_114/results.json';old114=read(prior114);assert read(D/'contact_headroom_audit_115/independent_audit.json')['passed'];start=old114['best']['parameters'];searches=[]
 bounds=np.array([(-.30,-.1),(-.12,.06),(-np.deg2rad(12),np.deg2rad(12)),(np.deg2rad(5),np.deg2rad(38)),(-np.deg2rad(20),np.deg2rad(20)),(.08,.14),(-np.deg2rad(15),np.deg2rad(15))])
 # Small physical-coordinate boxes avoid the large infeasible steps of114.
 base_radius=np.array([.003,.003,.01,.01,.01,.002,.01]);radius=base_radius.copy();x=np.array(start);iterations=[];search_status=dict(success=False,message='Eight bounded trust-region iterations exhausted')
 try:
  for iteration in range(8):
   current=endpoint(x);c0=cons(x);assert np.min(c0)>=0 and current['all_endpoint_model_checks_passed']
   f0=current['min_command_margin_rad'];gf=[];gc=[];probes=[]
   for j in range(7):
    h=1e-5 if x[j]+1e-5<=bounds[j,1] else -1e-5;y=x.copy();y[j]+=h;a=endpoint(y);gf.append((a['min_command_margin_rad']-f0)/h);gc.append((cons(y)-c0)/h);probes.append(dict(parameters=y.tolist(),delta=h))
   gf=np.array(gf);gc=np.array(gc).T
   step_bounds=[(max(-radius[j],bounds[j,0]-x[j]),min(radius[j],bounds[j,1]-x[j])) for j in range(7)]
   lp=linprog(-gf,A_ub=-gc,b_ub=c0,bounds=step_bounds,method='highs',options=dict(primal_feasibility_tolerance=1e-10,dual_feasibility_tolerance=1e-10));item=dict(iteration=iteration,start=x.tolist(),radius=radius.tolist(),margin_before=f0,linear_solver_success=bool(lp.success),linear_solver_message=lp.message,probes=probes,trials=[])
   if not lp.success:iterations.append(item);break
   item.update(step=lp.x.tolist(),linear_predicted_gain=float(gf@lp.x),constraint_jacobian=gc.tolist(),objective_gradient=gf.tolist(),constraints_before=c0.tolist())
   accepted=False
   for fraction in [1.,.5,.25,.125]:
    y=x+fraction*lp.x;a=endpoint(y);cy=cons(y);ok=bool(np.min(cy)>=0 and a['all_endpoint_model_checks_passed'] and a['min_command_margin_rad']>f0+1e-7)
    item['trials'].append(dict(fraction=fraction,parameters=y.tolist(),constraint_min=float(np.min(cy)),margin=a['min_command_margin_rad'],accepted=ok))
    if ok:x=y;accepted=True;break
   item['accepted']=accepted;iterations.append(item);radius=np.minimum(base_radius*2,radius*1.25) if accepted else radius*.5
   print(json.dumps(dict(iteration=iteration,evaluations=len(history),accepted=accepted,margin=endpoint(x)['min_command_margin_rad'],clearance_mm=endpoint(x)['clearance_m']*1000)),flush=True)
   if endpoint(x)['min_command_margin_rad']>=.06:search_status=dict(success=True,message='Static endpoint research margin reached; no motion admission');break
 except BudgetReached as ex:search_status=dict(success=False,message=str(ex))
 eligible=[a for a in history if a['all_endpoint_model_checks_passed']];best=max(eligible,key=lambda a:a['min_command_margin_rad']) if eligible else max(history,key=lambda a:a['min_command_margin_rad']);searches.append(dict(start=start,**search_status,endpoint=best));path=[]
 # This result is an endpoint proposal only. Never interpolate a planted foot
 # to a different location/yaw; a lifted swing trajectory must be built later.
 files=[Path(__file__).resolve(),mp,ip,cp,src/'trace.npz',src/'sensor_inputs.npz',oldpath,prior_path,previous89,prior114,D/'contact_headroom_audit_115/independent_audit.json',D/'contact_command_bound_audit_113/independent_audit.json',Path('scripts/low_seat_terminal_contract_42.py')]
 result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='Bounded OFFLINE feasible trust-region endpoint search116 after114 took large invalid steps. Same seven-variable89 bounds, same static force LP, cooked collision/current1s sensors/frozen inertia, unchanged arms/fingers/materials/physical/clearance/foot/command/effort limits. Start from114 best; eight trust-region iterations and at most100unique endpoint evaluations, finite differences and up tofour half-step trials each. Accept only improvements satisfying unchanged internal research reserves; derivatives may sample rejected poses and all are retained. The .06rad command-margin research target is stricter than actual.03 and is not a filter/rate/criterion change. No path, actual filtering, sensed load, dynamics, simulator or holdout; no impossibility proof',searches=searches,iterations=iterations,history=history,best=best,summary=dict(endpoint_model_passed=best['all_endpoint_model_checks_passed'],housing_clearance_mm=best['clearance_m']*1000,parameters=best['parameters'],command_margin_rad=best['min_command_margin_rad'],headroom_research_target_rad=.06,headroom_research_target_passed=bool(best['min_command_margin_rad']>=.06),evaluations=len(history),maximum_evaluations=100,actual_command_filter_checked=False,load_is_measured=False,whole_path_checked=False),source_sha256={str(p.resolve()):sha(p) for p in files},physics_ticks=0,optimizer_updates=0,holdout_used=False,physical_admission=False)
 (O/'results.json').write_text(json.dumps(result,indent=2));(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2));print(json.dumps(result['summary']),flush=True)
if __name__=='__main__':main()
