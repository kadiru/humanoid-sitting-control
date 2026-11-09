"""One new complete nominal construction from saved430, not its optimizer restart.

Persistent four foot-local sole vertices replace WORLD array-slot interpolation.
Saved loads are copied into new identity slots; old files are never changed.
At nodes15..64 the nominal COM is projected onto the LEFT sole inset, preserving
each foot's total normal allocation. This explicitly includes one pre-lift segment.
Positive moment roots produce new loads only where that target changes. All other
loads are preserved. Between nodes, interpolate these persistent loads and declared
330 foot frames; their actual moments define the COM target. A constrained FK lift
solves rootxy/12legs (and descent during guarded normal approach). Root orientation,
arm rule, step/lift, initial and104..114 endpoint poses remain cardinal constraints.

This is a bounded construction, not trajectory optimization or an admitted schedule.
It retains the original imperfect330 foot FRAME path: approach frames drift within
the old pose errors. Matching that declared path is NOT a world-fixed foot or no-slip
certificate. No dynamics, forward filter or current quiet samples are generated.
"""
from pathlib import Path
from fractions import Fraction
import json,hashlib
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from humanoid_sitting.shared_frame import BodyModel,apply,transform
from low_seat_joint_path_controller_17 import measured
from assess_low_seat_clock_lift_v2_434 import assess,rotmix
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')
read=lambda p:json.loads(Path(p).read_text())
FLOOR=16*float(np.finfo(np.float32).eps)
BUFFER=2.**-23
SUPPORT=.005+8*float(np.finfo(np.float32).eps)+.005*BUFFER
# Ten times the unchanged1e-13 root equation accuracy, for a declared nominal target.
PROJECTION_INSET=SUPPORT+2e-12
GROUPS=[(0,16),(16,64),(64,99),(99,104),(104,114)]
BUDGETS=[93,194,245,101,17]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def setup():
 data=read(D/'lattice_path_430/results.json');path=data['dense']['path'];nodes=data['dense']['command_nodes']
 saved=dict(np.load(D/'clock_lift_v2_434/lift.npz'));sel=np.r_[np.arange(100),np.arange(101,116)]
 feet=saved['actual_foot_reference'][1351:][sel].copy()
 Q=np.array([v['q'] for v in path]);roots=np.array([v['root'] for v in path]);P=saved['points'][1351:][sel].copy();oldw=saved['weights'][1351:][sel].copy()
 local=np.array([(P[0,j*4:(j+1)*4]-feet[0,j,:3,3])@feet[0,j,:3,:3] for j in range(2)])
 weights=np.zeros_like(oldw);mapping=np.full((115,8),-1,int);identity_error=0.
 for k in range(115):
  for j in range(2):
   for h in range(4):
    slot=j*4+h
    if oldw[k,slot]==0:continue
    xyz=(P[k,slot]-feet[k,j,:3,3])@feet[k,j,:3,:3]
    dist=np.linalg.norm(local[j]-xyz,axis=1);idx=int(np.argmin(dist));assert dist[idx]<2e-10
    assert weights[k,j*4+idx]==0
    weights[k,j*4+idx]=oldw[k,slot];mapping[k,slot]=j*4+idx;identity_error=max(identity_error,float(dist[idx]))
 points=np.array([np.concatenate([apply(feet[k,j],local[j]) for j in range(2)]) for k in range(115)])
 moments=np.array([[(weights[k,j*4:(j+1)*4,None]*points[k,j*4:(j+1)*4,:2]).sum(0) for j in range(2)] for k in range(115)])
 oldmom=np.array([[(oldw[k,j*4:(j+1)*4,None]*P[k,j*4:(j+1)*4,:2]).sum(0) for j in range(2)] for k in range(115)])
 assert np.max(abs(moments-oldmom))<2e-12
 assert np.all(weights[16:65,4:]==0)
 # Exact rational event times. Interior proportions come only from saved430 dt.
 dt=np.array(data['dense']['segment_durations_s']);node_times=[Fraction(0)]*115;start=Fraction(0)
 for (a,b),budget in zip(GROUPS,BUDGETS):
  ds=[Fraction.from_float(float(x)) for x in dt[a:b]];total=sum(ds);elapsed=Fraction(0)
  for k,duration in enumerate(ds,a+1):
   elapsed+=duration;node_times[k]=start+Fraction(budget,50)*elapsed/total
  start+=Fraction(budget,50)
 assert node_times[-1]==13
 times=node_times[:100]+[node_times[99]+Fraction(1,2)]+[t+Fraction(1,2) for t in node_times[100:]]
 ids=np.r_[np.arange(100),99,np.arange(100,115)]
 queries=[(Fraction(i,50),None,'clock',i) for i in range(676)]+[(Fraction(2*i+1,100),None,'half_tick',i) for i in range(675)]+[(t,i,'vertex',i) for i,t in enumerate(times)]
 m=BodyModel(D/'cooked_model_50/model.json');ine=read(D.parent/'crouch_rest_bridge_01/balanced_standing10_01/static_inertial_model.json');mass=np.array(ine['mass']);W=mass.sum()*9.81
 cal=read(D.parent/'contact_effort_repair_01/plane_calibration.json');tr=np.load(D/'normal_landing3s_06/candidate/trace.npz');ss=np.load(D/'normal_landing3s_06/candidate/sensor_inputs.npz')['sensors'];at=int(np.argmin(abs(tr['t']-1)))
 q0=tr['q'][at,2].astype(float);command0=tr['command'][at,2].astype(float)
 n,plane_d,origin,h,slope,*_=measured(m,q0,ss[0,10,2,339:342],tr['geometry'][at,2],[tr['estimated_root_x'][at,2],0,0],cal)
 SR=Rotation.from_euler('y',slope,degrees=True).as_matrix();center=np.array([-.4,0,h]);sides=['left','right']
 legs=[m.names.index(s+'_'+v+'_joint') for s in sides for v in ['hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch','ankle_roll']];arms=[m.names.index(s+'_'+v+'_joint') for s in sides for v in ['shoulder_pitch','elbow_pitch']];moving=legs+arms;fixed=[j for j in range(37) if j not in moving]
 kp=np.array([200,150,150,200,160,160]*2+[40]*4);effort=np.array([300,300,300,300,20,20]*2+[300]*4)
 byname={j['name']:j for j in m.joints};parent={j['body1'][0]:j['body0'][0] for j in m.joints if j['body0']}
 def below(body,ancestor):
  while True:
   if body==ancestor:return True
   if body not in parent:return False
   body=parent[body]
 bodies=set(ine['body_names'])|set(m.vertices)
 sub={name:{b for b in bodies if below(b,byname[name]['body1'][0])} for name in m.names}
 limits={j['name']:np.deg2rad(j['limits']) for j in m.joints if j['type']=='PhysicsRevoluteJoint'}
 tau0=np.array(nodes[0]['planned_static_torque_Nm']);corner_sides=['left']*4+['right']*4
 f=np.array([v['approach_fraction'] for v in path]);p=np.r_[0.,np.load(D/'lattice_path_430/final.npz')['full_coordinates'].reshape(114,23)[:,3]]
 dz=roots[104,2]-roots[0,2];normal=np.array([v['assumed_seat_normal_fraction'] for v in nodes]);initial=dict(np.load(D/'lattice_path_430/initial.npz'))
 return locals()
def project_polygon(point,H,inset):
 """Exact finite active-set Euclidean projection in2D, not an allocation search."""
 A=H[:,:2];b=-H[:,2]-inset
 if np.all(A@point<=b):return point.copy(),[]
 options=[]
 for j,a in enumerate(A):
  x=point-a*((a@point-b[j])/(a@a))
  if np.min(b-A@x)>=-2e-15:options.append((float(np.sum((x-point)**2)),(j,),x))
 for j in range(len(A)):
  for k in range(j):
   M=A[[j,k]]
   if abs(np.linalg.det(M))<1e-12:continue
   x=np.linalg.solve(M,b[[j,k]])
   if np.min(b-A@x)>=-2e-15:options.append((float(np.sum((x-point)**2)),(k,j),x))
 assert options,'empty inset polygon';_,active,x=min(options,key=lambda v:(v[0],v[1]));return x,list(active)
def moment_root(reference,A,target):
 """Positive exponential-family root. Max30iterations/12halvings; no fallback."""
 assert np.all(reference>FLOOR);base=np.log(reference-FLOOR);lam=np.zeros(len(target))
 for it in range(30):
  e=np.exp(base+A.T@lam);w=FLOOR+e;r=A@w-target
  if np.max(abs(r))<=1e-13:return w,dict(iterations=it,residual=float(abs(r).max()),lambda_value=lam.tolist())
  step=np.linalg.solve((A*e)@A.T,-r);norm=np.linalg.norm(r)
  for half in range(12):
   trial=lam+step*2.**(-half);ee=np.exp(base+A.T@trial)
   if np.linalg.norm(A@(FLOOR+ee)-target)<norm:lam=trial;break
  else:raise RuntimeError('bounded moment root descent failed')
 raise RuntimeError('bounded moment root iteration cap')
def node_loads(s):
 w=s['weights'].copy();changes=[]
 for k in range(15,65):
  left=apply(s['feet'][k,0],s['local'][0]);H=ConvexHull(left[:,:2]).equations;point=(w[k,:,None]*s['points'][k,:,:2]).sum(0)
  target,edges=project_polygon(point,H,PROJECTION_INSET)
  if np.array_equal(point,target):continue
  aa=np.flatnonzero(w[k]>0);P=s['points'][k,aa,:2];origin=P.mean(0)
  active_sides=sorted(set((aa//4).tolist()));A=np.vstack([(aa//4==j).astype(float) for j in active_sides]+[((P-origin)/.1).T[0],((P-origin)/.1).T[1]])
  F=float(w[k].sum());t=np.r_[[w[k,j*4:(j+1)*4].sum() for j in active_sides],(target-origin*F)/.1]
  out,info=moment_root(w[k,aa],A,t);w[k,aa]=out
  changes.append(dict(knot=k,original_com_xy=point.tolist(),new_com_xy=target.tolist(),displacement_m=(target-point).tolist(),active_polygon_edges=edges,moment_root=info))
 return w,changes
def record(s,w,t,vertex,kind,index):
 import bisect
 if vertex is not None:i=int(vertex);k=l=int(s['ids'][i]);a=0.
 else:
  i=min(max(bisect.bisect_right(s['times'],t)-1,0),114);k=int(s['ids'][i]);l=int(s['ids'][i+1]);a=float((t-s['times'][i])/(s['times'][i+1]-s['times'][i]));assert 0<=a<=1
 if k==l:a=0.
 mix=lambda v:v[k].copy() if a==0 else v[l].copy() if a==1 else v[k]+a*(v[l]-v[k])
 R=rotmix(Rotation.from_quat(s['roots'][k,3:]).as_matrix(),Rotation.from_quat(s['roots'][l,3:]).as_matrix(),a)
 F=np.array([transform(rotmix(s['feet'][k,j,:3,:3],s['feet'][l,j,:3,:3],a),mix(s['feet'][:,j,:3,3])) for j in range(2)])
 W=mix(w);P=np.concatenate([apply(F[j],s['local'][j]) for j in range(2)])
 q=mix(s['Q']);root=mix(s['roots'][:,:3]);phase=float(mix(s['f']));q[s['arms']]=s['q0'][s['arms']]-.35*phase;q[s['fixed']]=s['q0'][s['fixed']]
 # Initial/end poses are copied exactly; changes at these nodes are forbidden.
 fixed_pose=(k==l and k==0) or k>=104
 if fixed_pose and k==l:q=s['Q'][k].copy();root=s['roots'][k,:3].copy()
 guarded=(k>=99 and l<=104) and not fixed_pose
 support_left=t>=s['times'][15] and t<=s['times'][64]
 stance=['left'] if support_left else ['left','right']
 feet=[]
 for j,side in enumerate(s['sides']):
  aa=s['path'][k]['foot_targets'][side];bb=s['path'][l]['foot_targets'][side]
  feet.append((np.array(aa['position'])+a*(np.array(bb['position'])-aa['position']),rotmix(np.array(aa['rotation']),np.array(bb['rotation']),a)))
 gap=(1-a)*s['path'][k]['pelvis_gap_m']+a*s['path'][l]['pelvis_gap_m']
 return dict(time_s=float(t),exact_time=[str(t.numerator),str(t.denominator)],vertex=vertex,segment=i,knot0=k,knot1=l,fraction=a,sample_kind=kind,sample_index=index,q=q,root=root,R=R,feet=feet,actual_foot_reference=F,stance=stance,weights=W,points=P,normal=float(mix(s['normal'])),linear_command=mix(np.array([n['command'] for n in s['nodes']])),com_reference=(W[:,None]*P[:,:2]).sum(0),phase=phase,orientation_phase=float(mix(s['p'])),gap_target=gap,guarded_gap=guarded,fixed_pose=fixed_pose)
def endpoint_forces(s,r):
 if r['knot0']<104:return r,None
 T=transform(r['R'],r['root']);fk=s['m'].fk(r['q']);C=sum(mm*apply(T@fk[b],cc) for b,cc,mm in zip(s['ine']['body_names'],s['ine']['local_com'],s['mass']))/s['mass'].sum()
 pv=apply(T,s['m'].vertices['pelvis']);depth=pv@s['n']+s['plane_d'];seat=pv[depth<=depth.min()+1e-7].mean(0);seat-=s['n']*(seat@s['n']+s['plane_d']);sv=r['normal']/s['n'][2]
 P=r['points'][:,:2];origin=P.mean(0);A=np.vstack([np.ones(8),((P-origin)/.1).T]);target=np.r_[1-sv,(C[:2]-sv*seat[:2]-origin*(1-sv))/.1]
 w,info=moment_root(r['weights'],A,target);out=r.copy();out['weights']=w;out['com_reference']=C[:2].copy();return out,info
def skew(a):return np.array([[0,-a[2],a[1]],[a[2],0,-a[0]],[-a[1],a[0],0]])
def kinematics(s,r,jacobian=False):
 m=s['m'];T=transform(r['R'],r['root']);fk=m.fk(r['q']);world={b:T@v for b,v in fk.items()};pts={b:apply(world[b],cc) for b,cc in zip(s['ine']['body_names'],s['ine']['local_com'])};C=sum(mm*pts[b] for b,mm in zip(s['ine']['body_names'],s['mass']))/s['mass'].sum();res=[]
 for j,side in enumerate(s['sides']):
  F=world[side+'_ankle_roll_link'];target=r['actual_foot_reference'][j];A=target[:3,:3].T@F[:3,:3]
  res.extend(F[:3,3]-target[:3,3]);res.extend(.1*np.array([A[2,1]-A[1,2],A[0,2]-A[2,0],A[1,0]-A[0,1]]))
 res.extend(C[:2]-r['com_reference']);pv=apply(T,m.vertices['pelvis']);gap=float((pv@s['n']+s['plane_d']).min())
 if r['guarded_gap']:res.append(gap-r['gap_target'])
 if not jacobian:return np.array(res),world,C,gap
 n=15 if r['guarded_gap'] else 14;J=np.zeros((n,n));geo={}
 for j in s['moving']:
  name=m.names[j];jj=s['byname'][name];JW=world[jj['body0'][0]]@m.frames[name][0];geo[j]=(JW[:3,'XYZ'.index(jj['axis'])],JW[:3,3])
 for col in range(n):
  rv=np.eye(3)[:,col] if col<2 else np.array([0.,0.,s['dz']]) if col==14 else np.zeros(3)
  joints=[(s['legs'][col-2],1.)] if 2<=col<14 else [(j,-.35) for j in s['arms']] if col==14 else []
  def vel(body,P):
   v=rv.copy();omega=np.zeros(3)
   for j,weight in joints:
    if body in s['sub'][m.names[j]]:
     a,o=geo[j];v+=weight*np.cross(a,P-o);omega+=weight*a
   return v,omega
  for j,side in enumerate(s['sides']):
   body=side+'_ankle_roll_link';F=world[body];v,omega=vel(body,F[:3,3]);dR=r['actual_foot_reference'][j,:3,:3].T@skew(omega)@F[:3,:3]
   J[j*6:j*6+3,col]=v;J[j*6+3:j*6+6,col]=.1*np.array([dR[2,1]-dR[1,2],dR[0,2]-dR[2,0],dR[1,0]-dR[0,1]])
  J[12:14,col]=(sum(mm*vel(b,pts[b])[0] for b,mm in zip(s['ine']['body_names'],s['mass']))/s['mass'].sum())[:2]
  if r['guarded_gap']:J[14,col]=s['n']@rv
 return np.array(res),J
def apply_step(s,r,step):
 out=r.copy();out['q']=r['q'].copy();out['root']=r['root'].copy();out['root'][:2]+=step[:2];out['q'][s['legs']]+=step[2:14]
 if r['guarded_gap']:
  out['phase']=r['phase']+step[14];out['root'][2]=s['roots'][0,2]+s['dz']*out['phase'];out['q'][s['arms']]=s['q0'][s['arms']]-.35*out['phase']
 return out
def solve_lift(s,r):
 if r['fixed_pose']:return r,dict(iterations=0,changed=False,fixed_pose=True)
 for it in range(10):
  residual,J=kinematics(s,r,True)
  if abs(residual).max()<=1e-13:return r,dict(iterations=it,residual=float(abs(residual).max()),fixed_pose=False)
  step=np.linalg.solve(J,-residual);norm=np.linalg.norm(residual)
  for half in range(9):
   trial=apply_step(s,r,step*2.**-half);rr,*_=kinematics(s,trial)
   if np.linalg.norm(rr)<norm:r=trial;break
  else:raise RuntimeError('bounded kinematic root descent failed')
 raise RuntimeError('bounded kinematic root iteration cap')
