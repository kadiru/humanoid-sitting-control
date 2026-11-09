# Extracted unchanged model/residual/derivative functions from preserved361v2.
# Original SHA256: 9dbdb39072623305ed755bd6f72d6902685f23e0a060cdbe27ae35ae7200bd2b
"""One reduced-coordinate complete-path nominal phase/force/guarded-time construction.
No physics, runtime, learning, or physical admission. Local NLP failure is not impossibility.
"""
import copy,sys
from pathlib import Path
import json,hashlib,datetime
import numpy as np
from scipy.spatial import ConvexHull,QhullError
from scipy.spatial.transform import Rotation,Slerp
from humanoid_sitting.shared_frame import BodyModel,apply,transform,level_rotation
from low_seat_joint_path_controller_17 import measured
from low_seat_terminal_contract_42 import terminal_posture_check
from plan_seat_load_targets import clipped_vertices
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')
read=lambda p:json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def smooth(u):return u*u*(3-2*u)
def make_problem():
 mp=D/'cooked_model_50/model.json';ip=D.parent/'crouch_rest_bridge_01/balanced_standing10_01/static_inertial_model.json';cp=D.parent/'contact_effort_repair_01/plane_calibration.json';src=D/'normal_landing3s_06/candidate';m=BodyModel(mp);meta=read(mp);inertia=read(ip);cal=read(cp);tr=np.load(src/'trace.npz');ss=np.load(src/'sensor_inputs.npz')['sensors'];mass=np.asarray(inertia['mass']);sides=['left','right'];ids={s:[m.names.index(s+'_'+j+'_joint') for j in ['hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch','ankle_roll']] for s in sides};legs=ids['left']+ids['right'];arms=[m.names.index(s+'_'+j+'_joint') for s in sides for j in ['shoulder_pitch','elbow_pitch']];limits={j['name']:np.deg2rad(j['limits']) for j in meta['joints'] if j['type']=='PhysicsRevoluteJoint'}
 c=2;at=int(np.argmin(abs(tr['t']-1)));q0=tr['q'][at,c].astype(float);g=ss[0,10,c,339:342];assert np.array_equal(q0[:23].astype(np.float32),ss[0,10,c,270:293]);n,d,origin,h,slope,_,_=measured(m,q0,g,tr['geometry'][at,c],[tr['estimated_root_x'][at,c],0,0],cal);R0=level_rotation(g);T0=transform(R0,origin);f0=m.fk(q0);feet0={s:(R0@f0[s+'_ankle_roll_link'][:3,3]+origin,R0@f0[s+'_ankle_roll_link'][:3,:3]) for s in sides};SR=Rotation.from_euler('y',slope,degrees=True).as_matrix();center=np.array([-.4,0,h]);C0=np.average([apply(T0@f0[b],v) for b,v in zip(inertia['body_names'],inertia['local_com'])],axis=0,weights=mass)
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
 return locals()
