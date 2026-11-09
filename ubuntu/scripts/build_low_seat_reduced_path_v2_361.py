"""One reduced-coordinate complete-path nominal phase/force/guarded-time construction.
No physics, runtime, learning, or physical admission. Local NLP failure is not impossibility.
"""
import copy,sys
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
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');O=D/('reduced_path_preparation_v2_361' if '--prepare-only' in sys.argv else 'reduced_path_v2_361')
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
 old69=read(D/'coupled_step_refined_69/results.json');tau0=np.array(read(D/'clearance_lift_path_330/results.json')['dense']['command_nodes'][0]['planned_static_torque_Nm']);fixed=[j for j in range(len(names)) if j not in moving];fingers=[i for i,v in enumerate(names) if any('_'+n+'_joint' in v for n in ['zero','one','two','three','four','five','six'])];assert len(fingers)==14
 base_supportxy=np.asarray(old69['dense']['path'][16]['com_target_xy']);endpoint_pose=np.array(end['root']);endpointC=np.array(end['com']);endR=Rotation.from_quat(endpoint_pose[3:]);interp=Slerp([0,1],Rotation.concatenate([Rotation.from_matrix(R0),endR]));hist=[];cache={}
 leftworld=apply(T0@f0['left_ankle_roll_link'],m.vertices['left_ankle_roll_link']);leftsole=leftworld[leftworld[:,2]<leftworld[:,2].min()+.003];leftH=ConvexHull(leftsole[:,:2]).equations;leftmargin=-(leftH[:,:2]@base_supportxy+leftH[:,2]);near=leftH[int(np.argmin(leftmargin)),:2];longitudinal=np.array([-near[1],near[0]])
 if longitudinal[0]<0:longitudinal=-longitudinal
 direction=leftH[:,:2]@longitudinal;max_rearward=min(.08,min((margin-.0055)/(-dv) for margin,dv in zip(leftmargin,direction) if dv< -1e-12));assert max_rearward>0
 import torch,time
 torch.set_num_threads(2);torch.set_default_dtype(torch.float64)
 Tn=lambda a:torch.as_tensor(np.asarray(a),dtype=torch.float64)
 prior_path=D/'clearance_lift_path_330/results.json';prior=read(prior_path);frozen=prior['dense']['path'];oldnodes=prior['dense']['command_nodes'];seed_data=read(D/'projected_complete_path_v2_359/results.json');seed_path=seed_data['dense']['path'];seed_nodes=seed_data['dense']['command_nodes'];basecmd=np.array([e['command'] for e in seed_nodes]);baseline_weights=[np.array(e['corner_fractions']) for e in seed_nodes]
 assert len(frozen)==115;change=np.arange(1,115);pose_change=np.arange(1,104);N=len(change);NP=len(pose_change);nvlocal=21;nv=N*nvlocal+N
 X=np.zeros((N,nvlocal));null=np.zeros((N,8,5));wbase=np.zeros((N,8));PP=np.zeros((N,8,3));ssmask=np.zeros((N,8,2));counts=[];nullsizes=[];footR=[];footP=[];comxy=[];leanRs=[]
 rootR0=Tn(R0);delta=Rotation.from_matrix(R0.T@endR.as_matrix()).as_rotvec();angle=np.linalg.norm(delta);axis=delta/angle;cross=np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
 lower=np.full((N,nvlocal),-1.);upper=np.full((N,nvlocal),1.);lower[:,2:4]=0.;lower[:,4:16]=[limits[names[j]][0]+.031 for j in legs];upper[:,4:16]=[limits[names[j]][1]-.031 for j in legs]
 for i,k in enumerate(change):
  e=frozen[k];se=seed_path[k];v=oldnodes[k];q=np.array(se['q']);pose=np.array(se['root']);TT=transform(Rotation.from_quat(e['root'][3:]).as_matrix(),e['root'][:3]);ff=m.fk(np.array(e['q']));X[i,:16]=np.r_[pose[:2],se['approach_fraction'],se.get('orientation_fraction',se['approach_fraction']),q[legs]]
  P=np.array(v['corner_world_points']);s=v['corner_sides'];cnt=len(P);counts.append(cnt);PP[i,:cnt]=P;wbase[i,:cnt]=baseline_weights[k]
  active=np.array([j for j,side in enumerate(s) if not (16<=k<=64) or side=='left']);A=np.vstack([np.ones(len(active)),P[active,:2].T]);uu,sv,vh=np.linalg.svd(A,full_matrices=True);rank=int((sv>1e-13).sum());B=vh[rank:].T;assert rank==3 and B.shape[1]<=5
  null[i,active,:B.shape[1]]=B;nullsizes.append(B.shape[1]);lower[i,16+B.shape[1]:]=0.;upper[i,16+B.shape[1]:]=0.
  if k>=104:lower[i,:16]=X[i,:16];upper[i,:16]=X[i,:16]
  for j,side in enumerate(s):ssmask[i,j,sides.index(side)]=1.
  footP.append([(TT@ff[side+'_ankle_roll_link'])[:3,3] for side in sides]);footR.append([(TT@ff[side+'_ankle_roll_link'])[:3,:3] for side in sides]);comxy.append(e['com'][:2])
  blend=smooth(e['u']) if e['phase']=='shift' else 1. if e['phase']=='swing' else 1-smooth(e['u']) if e['phase']=='approach' else 0.
  leanRs.append(Rotation.from_rotvec([prior['dense']['parameters'][2]*blend,0,0]).as_matrix())
 txq0=Tn(q0);tp=Tn(PP);tnull=Tn(null);twbase=Tn(wbase);tfootP=Tn(footP);tfootR=Tn(footR);tcomxy=Tn(comxy);tlean=Tn(leanRs);tcross=Tn(cross);eye=torch.eye(3)
 ordered=[];known={'pelvis'};pending=m.joints.copy()
 while pending:
  done=[]
  for j in pending:
   if j['body0'] and j['body0'][0] in known:ordered.append(j);known.add(j['body1'][0]);done.append(j)
  assert done;pending=[j for j in pending if j not in done]
 frames={j['name']:(Tn(m.frames[j['name']][0]),Tn(m.frames[j['name']][1])) for j in ordered};axis_cross={}
 for j in ordered:
  a=np.eye(3)['XYZ'.index((j.get('axis') or 'X'))];axis_cross[j['name']]=Tn(np.array([[0,-a[2],a[1]],[a[2],0,-a[0]],[-a[1],a[0],0]]))
 def calc(xx,extra=False):
  xx=Tn(xx);f=xx[:,2];p=xx[:,3];q=txq0.repeat(N,1);q[:,legs]=xx[:,4:16];q[:,arms]=txq0[arms]-.35*f[:,None]
  rr=eye+torch.sin(p*angle)[:,None,None]*tcross+(1-torch.cos(p*angle))[:,None,None]*(tcross@tcross);rr=tlean@rootR0@rr
  root=torch.cat([xx[:,:2],(origin[2]+f*(endpoint_pose[2]-origin[2]))[:,None]],1);T=torch.eye(4).repeat(N,1,1);T[:,:3,:3]=rr;T[:,:3,3]=root;poses={'pelvis':T};Jworld={}
  for j in ordered:
   name=j['name'];A,B=frames[name];J=poses[j['body0'][0]]@A;Jworld[name]=J;theta=q[:,names.index(name)] if name in names else torch.zeros(N);K=axis_cross[name];r=eye+torch.sin(theta)[:,None,None]*K+(1-torch.cos(theta))[:,None,None]*(K@K);Q=torch.eye(4).repeat(N,1,1);Q[:,:3,:3]=r;poses[j['body1'][0]]=J@Q@B
  bodypts={b:(poses[b][:,:3,:3]@Tn(cc)[:,None]).squeeze(-1)+poses[b][:,:3,3] for b,cc in zip(ine['body_names'],ine['local_com'])};C=sum(mm*bodypts[b] for b,mm in zip(ine['body_names'],ine['mass']))/sum(ine['mass']);eq=[]
  for k,side in enumerate(sides):
   F=poses[side+'_ankle_roll_link'];dR=tfootR[:,k].transpose(1,2)@F[:,:3,:3];omega=.5*torch.stack([dR[:,2,1]-dR[:,1,2],dR[:,0,2]-dR[:,2,0],dR[:,1,0]-dR[:,0,1]],1);eq.extend([F[:,:3,3]-tfootP[:,k],.2*omega])
  eq.append(C[:,:2]-tcomxy);w=twbase+torch.einsum('nik,nk->ni',tnull,xx[:,16:]);tau=[]
  for j in moving:
   name=names[j];J=Jworld[name];a=J[:,:3,'XYZ'.index(byname[name]['axis'])];o=J[:,:3,3];mom=torch.zeros((N,3))
   for b,mm in zip(ine['body_names'],ine['mass']):
    if b in sub[name]:mom+=torch.linalg.cross(bodypts[b]-o,Tn([0,0,-mm*9.81]).expand(N,3))
   mask=sum(ssmask[:,:,k] for k,side in enumerate(sides) if side+'_ankle_roll_link' in sub[name])
   if not np.isscalar(mask):
    forces=torch.zeros((N,8,3));forces[:,:,2]=W*w*Tn(mask);mom+=torch.linalg.cross(tp-o[:,None,:],forces).sum(1)
   tau.append(-(a*mom).sum(1))
  tau=torch.stack(tau,1);cmd=Tn(command0[moving])+q[:,moving]-Tn(q0[moving])+(tau-Tn(tau0))/Tn(kp)
  pv=(T[:,:3,:3]@Tn(m.vertices['pelvis']).T).transpose(1,2)+root[:,None,:];gap=(pv@Tn(n)+d).min(1).values
  out=torch.cat(eq+[cmd,tau,w,gap[:,None]],1).numpy()
  return (out,q.numpy(),T.numpy(),C.numpy()) if extra else out
 #357/358 establish a nonsingular, moderately conditioned dependent-coordinate block.
 assert read(D/'constraint_coordinates_audit_358/independent_audit.json')['passed']
 assert read(D/'projected_path_audit_360/independent_audit.json')['passed']
 dependent=np.array([0,1]+list(range(4,16)));phase=np.array([2,3]);template=X.copy();nvlocal_reduced=7;nr=N*7+N
 V=np.c_[X[:,2:4],X[:,16:]];reduced_lower=np.c_[lower[:,2:4],lower[:,16:]];reduced_upper=np.c_[upper[:,2:4],upper[:,16:]]
 def local_jac(xx,columns=None,h=1e-6):
  columns=list(range(21)) if columns is None else list(columns);jac=np.empty((N,55,len(columns)))
  for a,j in enumerate(columns):
   plus=xx.copy();minus=xx.copy();plus[:,j]+=h;minus[:,j]-=h;jac[:,:,a]=(calc(plus)-calc(minus))/(2*h)
  return jac
 def on_manifold(v):
  xx=template.copy();xx[:,2:4]=v[:,:2];xx[:,16:]=v[:,2:];max_steps=10;used=0
  for it in range(max_steps):
   yy=calc(xx);r=yy[:NP,:14];bad=np.max(abs(r),axis=1)>1e-13
   if not bad.any():return xx,yy,used
   J=local_jac(xx,dependent)[:NP,:14];delta=np.linalg.solve(J,-r[...,None])[...,0];accepted=~bad;norm=np.linalg.norm(r,axis=1);used+=1
   #Finite local equation solves may leave joint bounds internally; outer constraints and
   #final full screens enforce the original bounds. No trial is a command or runtime input.
   for half in range(9):
    trial=xx.copy();active=np.where(~accepted)[0];trial[active[:,None],dependent]+=delta[active]*(2.**(-half));ty=calc(trial);better=np.linalg.norm(ty[:NP,:14],axis=1)<norm;take=(~accepted)&better
    ii=np.where(take)[0];xx[ii]=trial[ii];accepted|=take
    if accepted.all():break
   if not accepted.all():raise RuntimeError('dependent-coordinate domain failed bounded Newton descent')
  yy=calc(xx)
  if np.max(abs(yy[:NP,:14]))>1e-13:raise RuntimeError('dependent-coordinate domain failed ten-iteration cap')
  return xx,yy,used
 rates=np.array([.1125 if ('ankle_pitch' in names[j] or 'ankle_roll' in names[j]) else .15 for j in moving]);desired_lo=np.array([limits[names[j]][0]+.03 for j in moving],np.float32);desired_hi=np.array([limits[names[j]][1]-.03 for j in moving],np.float32);command_lo=np.maximum(np.nextafter(desired_lo,np.float32(np.inf)).astype(float),np.array([limits[names[j]][0]+.03 for j in moving]));command_hi=np.minimum(np.nextafter(desired_hi,np.float32(-np.inf)).astype(float),np.array([limits[names[j]][1]-.03 for j in moving]))
 seedX,seedY,_=on_manifold(V);seedcmd=np.vstack([basecmd[0,moving],seedY[:,14:30]]);seedgap=np.r_[frozen[0]['pelvis_gap_m'],seedY[:,54]];dt0=np.maximum(np.max(abs(np.diff(seedcmd,axis=0))/rates,axis=1),.02)
 for k in range(100,105):dt0[k-1]=max(dt0[k-1],(seedgap[k-1]-seedgap[k])/.006)
 x0=np.r_[V.ravel(),dt0];bounds=list(zip(reduced_lower.ravel(),reduced_upper.ravel()))+[(.02,13.5)]*N;free=np.array([k for k,(lo,hi) in enumerate(bounds) if lo!=hi]);fixed_x=x0.copy();last={};history=[];tstart=time.monotonic();cols=np.arange(N*7).reshape(N,7)
 def evaluate(x,derivative=False):
  if 'x' not in last or not np.array_equal(last['x'],x):
   xx,yy,its=on_manifold(x[:N*7].reshape(N,7));last.clear();last.update(x=x.copy(),X=xx,Y=yy,newton_steps=its)
  xx=last['X'];Y=last['Y'];cmd=np.vstack([basecmd[0,moving],Y[:,14:30]]);dd=np.diff(cmd,axis=0);dt=x[N*7:];gap=np.r_[frozen[0]['pelvis_gap_m'],Y[:,54]];vv=x[:N*7].reshape(N,7)
  eq=np.array([(gap[99]-.012)/.012]);mono=np.diff(np.vstack([[0.,0.],vv[:NP,:2],[1.,1.]]),axis=0).ravel();local=np.c_[Y[:,14:30]-command_lo,command_hi-Y[:,14:30],effort-Y[:,30:46],effort+Y[:,30:46],Y[:,46:54],1-Y[:,46:54]];dep_bounds=np.c_[xx[:NP,dependent]-lower[:NP,dependent],upper[:NP,dependent]-xx[:NP,dependent]]
  hrhi=(command_hi-cmd[1:])*dt[:,None]-.18*dd;hrlo=(cmd[1:]-command_lo)*dt[:,None]+.18*dd
  iq=np.r_[local.ravel(),dep_bounds.ravel(),mono,(rates*dt[:,None]-dd).ravel(),(rates*dt[:,None]+dd).ravel(),hrhi.ravel(),hrlo.ravel(),gap[1:99]-.012,.012-gap[100:105],gap[1:105]+1e-8,gap[65:104]-gap[66:105],.006*dt[99:104]-(gap[99:104]-gap[100:105]),13.5-.5-dt.sum()]
  if not derivative:return eq,iq
  if 'J' not in last:
   full=local_jac(xx);K=full[:NP,:14][:,:,dependent];B=full[:NP,:14][:,:,phase];tangent=-np.linalg.solve(K,B);J=np.zeros((N,55,7));J[:NP,:,:2]=full[:NP,:,phase]+np.einsum('nra,nab->nrb',full[:NP,:,dependent],tangent);J[:,:,2:]=full[:,:,16:];last.update(J=J,tangent=tangent)
  J=last['J'];tangent=last['tangent'];E=np.zeros((1,nr));I=np.zeros((len(iq),nr));E[0,cols[98]]=J[98,54]/.012;lc=local.shape[1]
  for i in range(N):I[i*lc:(i+1)*lc,cols[i]]=np.r_[J[i,14:30],-J[i,14:30],-J[i,30:46],J[i,30:46],J[i,46:54],-J[i,46:54]]
  off=N*lc
  for i in range(NP):I[off+i*28:off+i*28+14,cols[i,:2]]=tangent[i];I[off+i*28+14:off+(i+1)*28,cols[i,:2]]=-tangent[i]
  off+=NP*28
  for i in range(NP+1):
   for j in range(2):
    if i<NP:I[off+i*2+j,i*7+j]=1
    if i>0:I[off+i*2+j,(i-1)*7+j]=-1
  off+=2*(NP+1)
  for sign in [-1,1]:
   for k in range(N):
    sl=slice(off+k*16,off+(k+1)*16);I[sl,N*7+k]=rates;I[sl,cols[k]]+=sign*J[k,14:30]
    if k>0:I[sl,cols[k-1]]-=sign*J[k-1,14:30]
   off+=N*16
  for sign in [-1,1]:
   for k in range(N):
    sl=slice(off+k*16,off+(k+1)*16);I[sl,N*7+k]=(command_hi-cmd[k+1]) if sign==-1 else (cmd[k+1]-command_lo);I[sl,cols[k]]=sign*(dt[k]+.18)*J[k,14:30]
    if k>0:I[sl,cols[k-1]]=-sign*.18*J[k-1,14:30]
   off+=N*16
  for k in range(1,99):I[off,cols[k-1]]=J[k-1,54];off+=1
  for k in range(100,105):I[off,cols[k-1]]=-J[k-1,54];off+=1
  for k in range(1,105):I[off,cols[k-1]]=J[k-1,54];off+=1
  for k in range(66,105):I[off,cols[k-2]]=J[k-2,54];I[off,cols[k-1]]=-J[k-1,54];off+=1
  for k in range(100,105):I[off,N*7+k-1]=.006;I[off,cols[k-2]]=-J[k-2,54];I[off,cols[k-1]]=J[k-1,54];off+=1
  assert off==len(iq)-1;I[-1,N*7:]=-1;return E,I
 e0,i0=evaluate(x0);sourcefiles=[Path(__file__),D/'constraint_coordinates_357/results.json',D/'constraint_coordinates_audit_358/independent_audit.json',D/'projected_complete_path_v2_359/results.json',D/'projected_path_audit_360/independent_audit.json',prior_path,mp,ip,cp,src/'trace.npz',src/'sensor_inputs.npz'];hashes={str(p.resolve()):sha(p) for p in sourcefiles}
 scope='One reduced-coordinate COMPLETE115-knot formulation. Eliminate rootxy+12leg angles using all14 original per-knot fixed-foot/COM equations at each function evaluation; retain206 independent descent/orientation phases, force-null coordinates and114times. Initial and endpoint/load poses exact. One max30iteration SLSQP, not restarted. Dependent Newton max10iterations/9halvings,1e-13 numerical residual; domain failure terminates the one solve and preserves last valid candidate. Implicit-function derivatives, explicit original dependent joint/root bounds, original static command/effort/corner/phase/rate/headroom/hover/normal/time constraints. Hover equality divided by12mm for numerical scaling; original12mm guard unchanged. Original full geometry mandatory final rejection. No runtime, extra feedback authority, physics or policy training.'
 if '--prepare-only' in sys.argv:
  E,I=evaluate(x0,True);rng=np.random.default_rng(361);v=np.zeros(nr);v[free]=rng.normal(size=len(free));v/=np.linalg.norm(v);ep,ipp=evaluate(x0+1e-6*v);em,imm=evaluate(x0-1e-6*v);derr=float(max(abs((ep-em)/2e-6-E@v).max(),abs((ipp-imm)/2e-6-I@v).max()));np.savez_compressed(O/'derivative_fixture.npz',direction=v,equality_analytic=E@v,inequality_analytic=I@v,equality_fd=(ep-em)/2e-6,inequality_fd=(ipp-imm)/2e-6);print(json.dumps(dict(global_directional_derivative_error=derr,fixed_direction_exact_zero=bool(np.all(v[np.setdiff1d(np.arange(nr),free)]==0)),inner_numerical_residual_tolerance=1e-13)),flush=True);assert derr<2e-6
  record=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope=scope,case=2,maximum_optimizer_runs=1,maximum_iterations=30,declared_variables=nr,free_variables=len(free),equalities=len(e0),inequalities=len(i0),maximum_initial_kinematic_residual=float(abs(seedY[:NP,:14]).max()),maximum_initial_command_error=float(abs(seedcmd-basecmd[:,moving]).max()),global_directional_derivative_error=derr,initial_time_s=float(dt0.sum()+.5),initial_equality=float(abs(e0).max()),initial_minimum_inequality=float(i0.min()),source_sha256=hashes,physics_ticks=0,policy_optimizer_updates=0,physical_admission=False)
  np.savez_compressed(O/'initial.npz',x=x0,free=free,template=template,null=null,base_weights=wbase,points=PP,counts=counts,rates=rates,command_lo=command_lo,command_hi=command_hi,desired_lo=desired_lo,desired_hi=desired_hi,lower=lower,upper=upper,reduced_lower=reduced_lower,reduced_upper=reduced_upper);(O/'preparation.json').write_text(json.dumps(record,indent=2));(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2));print(json.dumps(record,indent=2),flush=True);return
 preparation=read(D/'reduced_path_preparation_v2_361/preparation.json');assert preparation['source_sha256']==hashes;assert preparation['global_directional_derivative_error']<2e-6
 for p,h in hashes.items():assert sha(p)==h
 grad=np.r_[np.zeros(N*7),np.ones(N)];last_valid=x0.copy();domain_error=None
 def expand(z):
  x=fixed_x.copy();x[free]=z;return x
 def callback(z):
  nonlocal last_valid
  x=expand(z);eq,iq=evaluate(x);last_valid=x.copy();row=dict(iteration=len(history)+1,time_s=float(.5+x[N*7:].sum()),max_equality=float(abs(eq).max()),min_inequality=float(iq.min()),max_kinematic_residual=float(abs(last['Y'][:NP,:14]).max()),elapsed_s=time.monotonic()-tstart);history.append(row);np.save(O/'last_iterate.npy',x);(O/'iterations.json').write_text(json.dumps(history,indent=2));print(json.dumps(row),flush=True)
 (O/'revision.json').write_text(json.dumps(preparation,indent=2));np.savez_compressed(O/'initial.npz',x=x0,free=free,template=template,null=null,base_weights=wbase,points=PP,counts=counts,rates=rates,command_lo=command_lo,command_hi=command_hi,desired_lo=desired_lo,desired_hi=desired_hi,lower=lower,upper=upper)
 try:
  fit=minimize(lambda z:.5+expand(z)[N*7:].sum(),x0[free],jac=lambda z:grad[free],method='SLSQP',bounds=[bounds[k] for k in free],constraints=[dict(type='eq',fun=lambda z:evaluate(expand(z))[0],jac=lambda z:evaluate(expand(z),True)[0][:,free]),dict(type='ineq',fun=lambda z:evaluate(expand(z))[1],jac=lambda z:evaluate(expand(z),True)[1][:,free])],callback=callback,options=dict(maxiter=30,ftol=1e-10));solution=expand(fit.x);opt=dict(success=bool(fit.success),message=fit.message,iterations=int(fit.nit),evaluations=int(fit.nfev))
 except RuntimeError as ex:
  domain_error=str(ex);solution=last_valid;opt=dict(success=False,message='terminated on dependent-coordinate domain failure; last valid callback retained',iterations=len(history),evaluations=None,domain_error=domain_error)
 eq,iq=evaluate(solution);XX=last['X'];Y,Q,Ts,Cs=calc(XX,True);path=copy.deepcopy(frozen);nodes=[];allcmd=basecmd.copy();allcmd[change[:,None],np.array(moving)[None,:]]=Y[:,14:30]
 for i,k in enumerate(change):
  if k>=104:continue
  old=frozen[k];feet={s:(np.array(old['foot_targets'][s]['position']),np.array(old['foot_targets'][s]['rotation'])) for s in sides};a=metrics(Ts[i,:3,3],Ts[i,:3,:3],Q[i],feet,old['stance']);f,p=XX[i,2:4];a.update(phase=old['phase'],u=old['u'],swing_attitude_bound_rad=old['swing_attitude_bound_rad'],approach_fraction=float(f),orientation_fraction=float(p),com_target_xy=old['com_target_xy'],com_target_error_m=float(np.linalg.norm(np.array(old['com_target_xy'])-Cs[i,:2])),multiknot_constraint_residual=float(abs(Y[i,:14]).max()));path[k]=a
 for k,e in enumerate(path):
  old=oldnodes[k];v=seed_nodes[k];P=np.array(old['corner_world_points']);ss=old['corner_sides'];cnt=len(P);normal=v['assumed_seat_normal_fraction']
  if k:
   i=k-1;w=Y[i,46:46+cnt];tau=Y[i,30:46];T=transform(Rotation.from_quat(e['root'][3:]).as_matrix(),e['root'][:3]);pv=apply(T,m.vertices['pelvis']);pd=pv@n+d;seat=pv[pd<=pd.min()+1e-7].mean(0);seat-=n*(seat@n+d);sv=normal/n[2];com=np.array(e['com']);res=float(max(abs(w.sum()+sv-1),abs(w@P[:,:2]+sv*seat[:2]-com[:2]).max()))
  else:w=np.array(v['corner_fractions']);tau=np.array(v['planned_static_torque_Nm']);res=v['max_equilibrium_residual']
  cmd=allcmd[k];margin=float(min(min(cmd[j]-limits[names[j]][0],limits[names[j]][1]-cmd[j]) for j in moving));eu=float(max(abs(tau)/effort));nodes.append(dict(knot=k,command=cmd.tolist(),planned_static_torque_Nm=tau.tolist(),corner_fractions=w.tolist(),corner_world_points=P.tolist(),corner_sides=ss,assumed_seat_normal_fraction=normal,planned_right_load_fraction=float(sum(ww for ww,side in zip(w,ss) if side=='right')),min_command_margin_rad=margin,maximum_effort_utilization=eu,max_equilibrium_residual=res))
 failed=[k for k,e in enumerate(path) if not e['model_pose_passed']];duration=float(.5+solution[N*7:].sum());margin=min(v['min_command_margin_rad'] for v in nodes);util=max(v['maximum_effort_utilization'] for v in nodes);forceok=all(min(v['corner_fractions'])>=0 and v['max_equilibrium_residual']<1e-8 for v in nodes);constraints=bool(abs(eq).max()<=1e-8 and iq.min()>=-1e-8);swingheight=min(e['foot_min_z']['right'] for e in path if e['phase']=='swing' and .2<e['u']<.8)
 opt.update(objective_time_s=duration,max_equality_residual=float(abs(eq).max()),minimum_inequality=float(iq.min()),numerical_constraints_passed=constraints,local_not_global=True)
 dense=dict(parameters=prior['dense']['parameters'],knots=115,path=path,command_nodes=nodes,segment_durations_s=solution[N*7:].tolist(),minimum_synthetic_quiet_s=.5,failed_knots=failed,min_clearance_m=min(e['clearance_m'] for e in path),min_cop_margin_m=min(e['foot_cop_margin_m'] for e in path),max_foot_position_error_m=max(e['foot_position_error_m'] for e in path),max_foot_orientation_error_rad=max(e['foot_orientation_error_rad'] for e in path),min_actual_swing_sole_height_m=swingheight,min_command_margin_rad=margin,maximum_effort_utilization=util,model_geometry_passed=bool(not failed and swingheight>=.005),model_command_passed=bool(margin>=.03 and util<=1 and forceok),guarded_continuous_timing_s=duration,guarded_timing_constraints_passed=bool(constraints and duration<=13.5),actual_filter_or_dynamic_check=False,physical_admission=False)
 result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),revision=dict(preparation,changed_pose_knots=pose_change.tolist()),endpoint=prior['endpoint'],stance_target_constraint=prior['stance_target_constraint'],dense=dense,unchanged_pose_knots=[0]+list(range(104,115)),optimizer=opt,source_input_sha256=hashes,physics_ticks=0,policy_optimizer_updates=0,physical_admission=False)
 np.savez_compressed(O/'final.npz',x=solution,full_coordinates=XX,eq=eq,ineq=iq,local_outputs=Y);(O/'results.json').write_text(json.dumps(result,indent=2));(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2));print(json.dumps(opt),flush=True);print(json.dumps({k:v for k,v in dense.items() if k not in ['path','command_nodes']}),flush=True)
if __name__=='__main__':main()
