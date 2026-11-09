"""One bounded simultaneous48-knot nominal pose/force/command-time optimization.
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
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');O=D/('multiknot_preparation_344' if '--prepare-only' in sys.argv else 'multiknot_path_344')
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
 base_supportxy=np.asarray(old69['dense']['path'][16]['com_target_xy']);endpoint_pose=np.array(end['root']);endpointC=np.array(end['com']);endR=Rotation.from_quat(endpoint_pose[3:]);interp=Slerp([0,1],Rotation.concatenate([Rotation.from_matrix(R0),endR]));hist=[];cache={}
 leftworld=apply(T0@f0['left_ankle_roll_link'],m.vertices['left_ankle_roll_link']);leftsole=leftworld[leftworld[:,2]<leftworld[:,2].min()+.003];leftH=ConvexHull(leftsole[:,:2]).equations;leftmargin=-(leftH[:,:2]@base_supportxy+leftH[:,2]);near=leftH[int(np.argmin(leftmargin)),:2];longitudinal=np.array([-near[1],near[0]])
 if longitudinal[0]<0:longitudinal=-longitudinal
 direction=leftH[:,:2]@longitudinal;max_rearward=min(.08,min((margin-.0055)/(-dv) for margin,dv in zip(leftmargin,direction) if dv< -1e-12));assert max_rearward>0
 import torch,time
 torch.set_num_threads(2);torch.set_default_dtype(torch.float64)
 Tn=lambda a:torch.as_tensor(np.asarray(a),dtype=torch.float64)
 prior_path=D/'clearance_lift_path_330/results.json';prior=read(prior_path);frozen=prior['dense']['path'];oldnodes=prior['dense']['command_nodes'];bound=read(D/'clearance_lift_command_bound_332/results.json');basecmd=np.array([e['command'] for e in bound['nodes']]);baseline_weights=[np.array(e['corner_fractions']) for e in bound['nodes']]
 assert len(frozen)==115 and bound['summary']['success'];change=np.arange(56,104);N=len(change);assert N==48
 # Direct simultaneous variables: each node's rootxy,descent,orientation,12legq,5padded force-null coordinates.
 # No extra knee bound/equality; no greedy prefix choice between changed nodes.
 nvlocal=21;nv=N*nvlocal+49;X=np.zeros((N,nvlocal));null=np.zeros((N,8,5));wbase=np.zeros((N,8));PP=np.zeros((N,8,3));ssmask=np.zeros((N,8,2));counts=[];nullsizes=[];footR=[];footP=[];comxy=[];leanRs=[]
 rootR0=Tn(R0);delta=Rotation.from_matrix(R0.T@endR.as_matrix()).as_rotvec();angle=np.linalg.norm(delta);axis=delta/angle;cross=np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
 lower=np.full((N,nvlocal),-1.);upper=np.full((N,nvlocal),1.);lower[:,2:4]=frozen[55]['approach_fraction'];lower[:,4:16]=[limits[names[j]][0]+.031 for j in legs];upper[:,4:16]=[limits[names[j]][1]-.031 for j in legs]
 for i,k in enumerate(change):
  e=frozen[k];v=oldnodes[k];q=np.array(e['q']);pose=np.array(e['root']);TT=transform(Rotation.from_quat(pose[3:]).as_matrix(),pose[:3]);ff=m.fk(q);X[i,:16]=np.r_[pose[:2],e['approach_fraction'],e['approach_fraction'],q[legs]]
  P=np.array(v['corner_world_points']);s=v['corner_sides'];cnt=len(P);counts.append(cnt);PP[i,:cnt]=P;wbase[i,:cnt]=baseline_weights[k];active=np.array([j for j,side in enumerate(s) if e['phase']!='swing' or side=='left']);A=np.vstack([np.ones(len(active)),P[active,:2].T]);uu,sv,vh=np.linalg.svd(A,full_matrices=True);rank=int((sv>1e-10).sum());B=vh[rank:].T;assert rank==3 and B.shape[1]<=5
  null[i,active,:B.shape[1]]=B;nullsizes.append(B.shape[1]);lower[i,16+B.shape[1]:]=0;upper[i,16+B.shape[1]:]=0
  for j,side in enumerate(s):ssmask[i,j,sides.index(side)]=1.
  footP.append([(TT@ff[side+'_ankle_roll_link'])[:3,3] for side in sides]);footR.append([(TT@ff[side+'_ankle_roll_link'])[:3,:3] for side in sides]);comxy.append(e['com'][:2]);blend=1. if e['phase']=='swing' else 1-smooth(e['u']);leanRs.append(Rotation.from_rotvec([prior['dense']['parameters'][2]*blend,0,0]).as_matrix())
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
  out=torch.cat(eq+[cmd,tau,w],1).numpy()
  return (out,q.numpy(),T.numpy(),C.numpy()) if extra else out
 # Frozen initial data independently reconstruct everychangednode before optimization.
 y0=calc(X);expectedcmd=basecmd[change][:,moving];expectedtau=np.array([bound['nodes'][k]['torque_Nm'] for k in change]);assert abs(y0[:,:14]).max()<1e-12;assert abs(y0[:,14:30]-expectedcmd).max()<1e-11;assert abs(y0[:,30:46]-expectedtau).max()<1e-10
 # Block derivatives are centered1e-6, identical perturbation per independent node.
 def local_jac(xx,h=1e-6):
  jac=np.empty((N,54,nvlocal))
  for j in range(nvlocal):
   plus=xx.copy();minus=xx.copy();plus[:,j]+=h;minus[:,j]-=h;jac[:,:,j]=(calc(plus)-calc(minus))/(2*h)
  return jac
 jac0=local_jac(X);jac_half=local_jac(X,5e-7);derivative_error=float(abs(jac0-jac_half).max());assert derivative_error<2e-6
 dt0=np.max(abs(np.diff(basecmd[55:105][:,moving],axis=0)),axis=1)/.15;fixed_time=float(np.max(abs(np.diff(basecmd[:56],axis=0)),axis=1).sum()/.15+np.max(abs(np.diff(basecmd[104:],axis=0)),axis=1).sum()/.15);x0=np.r_[X.ravel(),dt0+1e-8];bounds=list(zip(lower.ravel(),upper.ravel()))+[(0,13.5)]*49;last={};history=[];tstart=time.monotonic()
 # Static force equilibrium is eliminated in exact null coordinates using fixed330feet/COM.
 # Full clipped-mesh geometry remains an unchanged FINAL admission gate, not a smooth NLP surrogate.
 command_lo=np.array([limits[names[j]][0]+.03 for j in moving]);command_hi=np.array([limits[names[j]][1]-.03 for j in moving]);columns=np.arange(N*nvlocal).reshape(N,nvlocal)
 def evaluate(x,derivative=False):
  if 'x' not in last or not np.array_equal(last['x'],x):last.clear();last.update(x=x.copy(),Y=calc(x[:N*nvlocal].reshape(N,nvlocal)))
  Y=last['Y'];cmd=np.vstack([basecmd[55,moving],Y[:,14:30],basecmd[104,moving]]);dd=np.diff(cmd,axis=0);dt=x[N*nvlocal:];xx=x[:N*nvlocal].reshape(N,nvlocal)
  eq=Y[:,:14].ravel();mono=np.diff(np.vstack([[frozen[55]['approach_fraction']]*2,xx[:,2:4],[1.,1.]]),axis=0).ravel();local=np.c_[Y[:,14:30]-command_lo,command_hi-Y[:,14:30],effort-Y[:,30:46],effort+Y[:,30:46],Y[:,46:54],1-Y[:,46:54]];ineq=np.r_[local.ravel(),mono,(.15*dt[:,None]-dd).ravel(),(.15*dt[:,None]+dd).ravel(),13.5-fixed_time-dt.sum()]
  if not derivative:return eq,ineq
  if 'J' not in last:last['J']=local_jac(xx)
  J=last['J'];E=np.zeros((N*14,nv));I=np.zeros((len(ineq),nv));lc=local.shape[1]
  for i in range(N):
   E[i*14:(i+1)*14,columns[i]]=J[i,:14];JL=np.r_[J[i,14:30],-J[i,14:30],-J[i,30:46],J[i,30:46],J[i,46:54],-J[i,46:54]];I[i*lc:(i+1)*lc,columns[i]]=JL
  off=N*lc
  for i in range(N+1):
   for j in range(2):
    if i<N:I[off+i*2+j,i*nvlocal+2+j]=1
    if i>0:I[off+i*2+j,(i-1)*nvlocal+2+j]=-1
  off+=2*(N+1)
  for sign in [-1,1]:
   for k in range(N+1):
    sl=slice(off+k*16,off+(k+1)*16);I[sl,N*nvlocal+k]=.15
    if k<N:I[sl,columns[k]]+=sign*J[k,14:30]
    if k>0:I[sl,columns[k-1]]-=sign*J[k-1,14:30]
   off+=(N+1)*16
  I[-1,N*nvlocal:]=-1;return E,I
 # Directional derivative of global assembled constraints, including inter-knot coupling.
 rng=np.random.default_rng(344);v=rng.normal(size=nv);v/=np.linalg.norm(v);e0,i0=evaluate(x0);E,I=evaluate(x0,True);ep,ineq_plus=evaluate(x0+1e-6*v);em,ineq_minus=evaluate(x0-1e-6*v);global_derivative_error=float(max(abs((ep-em)/2e-6-E@v).max(),abs((ineq_plus-ineq_minus)/2e-6-I@v).max()));assert global_derivative_error<2e-6
 revision=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope=__doc__,case=2,maximum_optimizer_runs=1,maximum_iterations=40,changed_knots=change.tolist(),variables=nv,construction='Simultaneous48nodes56..103 rootxy,independent monotone descent/orientation,12legangles and nullspace-footforces plus49segmenttimes. Preserve actual MODELED330feet/COM atnodes; original target/screens remain. No extra knee equality/bound. Frozen330remainingposes,332remainingcommands. Minimize full115-node commandpolyline time, hard .03margin/effort/nonnegativeforce/static-equilibrium/.15necessaryrate/13.5time and originaljointbounds. No actualfilter or dynamics. Full clippedmesh/finitepatch/CoP/floor/foot/tilt are FINAL rejection gates, not differentiated optimizer constraints; no admission if any fail.',inputs='saved1s current-sensor development fixture, frozen staticmodel; never runtime injection',source_sha256={str(p.resolve()):sha(p) for p in [Path(__file__),prior_path,D/'clearance_lift_command_bound_332/results.json',D/'clearance_lift_bound_audit_333/independent_audit.json',mp,ip,cp,src/'trace.npz',src/'sensor_inputs.npz']},fixture=dict(max_kinematic_error=float(abs(y0[:,:14]).max()),max_command_error=float(abs(y0[:,14:30]-expectedcmd).max()),max_static_torque_error=float(abs(y0[:,30:46]-expectedtau).max()),local_half_step_derivative_error=derivative_error,global_directional_derivative_error=global_derivative_error,initial_eq=float(abs(e0).max()),initial_min_inequality=float(min(i0)),initial_time_s=fixed_time+dt0.sum(),null_dimensions=nullsizes),physics_ticks=0,physical_admission=False)
 (O/'revision.json').write_text(json.dumps(revision,indent=2));np.savez_compressed(O/'initial.npz',x=x0,lower=np.array([v[0] for v in bounds]),upper=np.array([v[1] for v in bounds]),null=null,base_weights=wbase,points=PP,counts=counts)
 print(json.dumps(dict(preparation=revision['fixture'],variables=nv,equalities=len(e0),inequalities=len(i0))),flush=True)
 if '--prepare-only' in sys.argv:
  (O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2));return
 grad=np.r_[np.zeros(N*nvlocal),np.ones(N+1)]
 def callback(x):
  eq,iq=evaluate(x);row=dict(iteration=len(history)+1,objective_time_s=float(fixed_time+x[N*nvlocal:].sum()),max_equality=float(abs(eq).max()),min_inequality=float(iq.min()),elapsed_s=time.monotonic()-tstart);history.append(row);np.save(O/'last_iterate.npy',x);(O/'iterations.json').write_text(json.dumps(history,indent=2));print(json.dumps(row),flush=True)
 fit=minimize(lambda x:fixed_time+x[N*nvlocal:].sum(),x0,jac=lambda x:grad,method='SLSQP',bounds=bounds,constraints=[dict(type='eq',fun=lambda x:evaluate(x)[0],jac=lambda x:evaluate(x,True)[0]),dict(type='ineq',fun=lambda x:evaluate(x)[1],jac=lambda x:evaluate(x,True)[1])],callback=callback,options=dict(maxiter=40,ftol=1e-10))
 Y,Q,Ts,Cs=calc(fit.x[:N*nvlocal].reshape(N,nvlocal),True);eq,iq=evaluate(fit.x);path=copy.deepcopy(frozen);nodes=[];allcmd=basecmd.copy();allcmd[change[:,None],np.array(moving)[None,:]]=Y[:,14:30];force_equilibrium=[];force_contact_error=[]
 for i,k in enumerate(change):
  old=frozen[k];feet={s:(np.array(old['foot_targets'][s]['position']),np.array(old['foot_targets'][s]['rotation'])) for s in sides};a=metrics(Ts[i,:3,3],Ts[i,:3,:3],Q[i],feet,old['stance']);f,p=fit.x[i*nvlocal+2:i*nvlocal+4];a.update(phase=old['phase'],u=old['u'],swing_attitude_bound_rad=old['swing_attitude_bound_rad'],approach_fraction=float(f),orientation_fraction=float(p),com_target_xy=old['com_target_xy'],com_target_error_m=float(np.linalg.norm(np.array(old['com_target_xy'])-Cs[i,:2])),multiknot_constraint_residual=float(abs(Y[i,:14]).max()));path[k]=a
 for k,e in enumerate(path):
  old=oldnodes[k];v=bound['nodes'][k];P=np.array(old['corner_world_points']);ss=old['corner_sides'];cnt=len(P)
  if k in change:
   i=int(k-change[0]);w=Y[i,46:46+cnt];tau=Y[i,30:46];normal=0.;com=np.array(e['com']);res=float(max(abs(w.sum()-1),abs(w@P[:,:2]-com[:2]).max()));force_equilibrium.append(res)
  else:w=np.array(v['corner_fractions']);tau=np.array(v['torque_Nm']);normal=v['assumed_seat_normal_fraction'];res=v['max_equilibrium_residual']
  cmd=allcmd[k];margin=float(min(min(cmd[j]-limits[names[j]][0],limits[names[j]][1]-cmd[j]) for j in moving));eu=float(max(abs(tau)/effort));nodes.append(dict(knot=k,command=cmd.tolist(),planned_static_torque_Nm=tau.tolist(),corner_fractions=w.tolist(),corner_world_points=P.tolist(),corner_sides=ss,assumed_seat_normal_fraction=normal,planned_right_load_fraction=float(sum(ww for ww,side in zip(w,ss) if side=='right')),min_command_margin_rad=margin,maximum_effort_utilization=eu,max_equilibrium_residual=res))
 failed=[k for k,e in enumerate(path) if not e['model_pose_passed']];duration=float(np.max(abs(np.diff(allcmd,axis=0)),axis=1).sum()/.15);margin=min(v['min_command_margin_rad'] for v in nodes);util=max(v['maximum_effort_utilization'] for v in nodes);force_ok=all(min(v['corner_fractions'])>=0 and v['max_equilibrium_residual']<1e-8 for v in nodes);changed_constraints=bool(abs(eq).max()<=1e-8 and iq.min()>=-1e-8);swingheight=min(e['foot_min_z']['right'] for e in path if e['phase']=='swing' and .2<e['u']<.8)
 dense=dict(parameters=prior['dense']['parameters'],knots=115,path=path,command_nodes=nodes,failed_knots=failed,min_clearance_m=min(e['clearance_m'] for e in path),min_cop_margin_m=min(e['foot_cop_margin_m'] for e in path),max_foot_position_error_m=max(e['foot_position_error_m'] for e in path),max_foot_orientation_error_rad=max(e['foot_orientation_error_rad'] for e in path),min_actual_swing_sole_height_m=swingheight,min_command_margin_rad=margin,maximum_effort_utilization=util,model_geometry_passed=bool(not failed and swingheight>=.005),model_command_passed=bool(margin>=.03 and util<=1 and force_ok),command_schedule_passed=bool(duration<=13.5),minimum_time_for_this_command_polyline_at015_s=duration,actual_filter_or_dynamic_check=False,physical_admission=False)
 result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),revision=revision,endpoint=prior['endpoint'],stance_target_constraint=prior['stance_target_constraint'],dense=dense,unchanged_knots=[k for k in range(115) if k not in change],optimizer=dict(success=bool(fit.success),message=fit.message,iterations=int(fit.nit),evaluations=int(fit.nfev),objective_time_s=float(fit.fun),max_equality_residual=float(abs(eq).max()),minimum_inequality=float(iq.min()),numerical_constraints_passed=changed_constraints,local_not_global=True),source_input_sha256=revision['source_sha256'],physics_ticks=0,optimizer_updates=0,physical_admission=False)
 np.savez_compressed(O/'final.npz',x=fit.x,eq=eq,ineq=iq,local_outputs=Y);(O/'results.json').write_text(json.dumps(result,indent=2));(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ['dense','revision','source_input_sha256','endpoint','stance_target_constraint']}));print(json.dumps({k:v for k,v in dense.items() if k not in ['path','command_nodes']}),flush=True)
if __name__=='__main__':main()
