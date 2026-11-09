"""Offline full-leg static force/command checks for saved contact paths.
No physics, new trajectory search, training or admission. Analytic virtual work
is cross-checked against central finite differences at every trajectory node.
"""
from pathlib import Path
import json,hashlib,datetime
import numpy as np
from scipy.optimize import minimize,linprog
from scipy.spatial import ConvexHull,QhullError
from scipy.spatial.transform import Rotation
from humanoid_sitting.shared_frame import BodyModel,apply,transform
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');O=D/'contact_command_screen_68'
read=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def main():
 O.mkdir(exist_ok=False);mp=D/'cooked_model_50/model.json';ip=D.parent/'crouch_rest_bridge_01/balanced_standing10_01/static_inertial_model.json';m=BodyModel(mp);ine=read(ip);W=sum(ine['mass'])*9.81;names=m.names;sides=['left','right'];legs=[names.index(s+'_'+j+'_joint') for s in sides for j in ['hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch','ankle_roll']];arms=[names.index(s+'_'+j+'_joint') for s in sides for j in ['shoulder_pitch','elbow_pitch']];moving=legs+arms;fixed=[j for j in range(len(names)) if j not in moving];fingers=[i for i,n in enumerate(names) if any('_'+v+'_joint' in n for v in ['zero','one','two','three','four','five','six'])];assert len(fingers)==14
 kp=np.array([200,150,150,200,160,160]*2+[40]*4,dtype=float);effort=np.array([300,300,300,300,20,20]*2+[300]*4,dtype=float)
 limits={j['name']:np.deg2rad(j['limits']) for j in m.joints if j['type']=='PhysicsRevoluteJoint'};byname={j['name']:j for j in m.joints};children={}
 for j in m.joints:
  if j['body0']:children.setdefault(j['body0'][0],[]).append(j['body1'][0])
 def descend(b):
  r={b}
  for c in children.get(b,[]):r|=descend(c)
  return r
 sub={name:descend(byname[name]['body1'][0]) for name in names}
 trp=D/'normal_landing3s_06/candidate/trace.npz';tr=np.load(trp);at=int(np.argmin(abs(tr['t']-1)));q0=tr['q'][at,2].astype(float);command0=tr['command'][at,2].astype(float);assert np.array_equal(tr['command'][:,2,fingers],np.broadcast_to(command0[fingers],tr['command'][:,2,fingers].shape))
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
   tau.append(float(-axis@moment));values=[]
   for sign in [-1,1]:
    qq=q.copy();qq[j]+=sign*1e-5;ff=m.fk(qq);V=sum(mm*9.81*apply(T@ff[b],cc)[2] for b,mm,cc in zip(ine['body_names'],ine['mass'],ine['local_com']));work=sum(W*w[i]*(T@ff[s+'_ankle_roll_link']@local[i])[2] for i,s in enumerate(ss));values.append(V-work)
   finite.append((values[1]-values[0])/2e-5)
  err=float(max(abs(np.array(tau)-finite)));assert err<1e-6,err
  return np.array(tau),err
 paths=[D/'single_step_spatial_screen_65/results.json',D/'coupled_step_refined_69/results.json'];rows=[]
 for p in paths:
  data=read(p);path=data['rows'][0]['path'] if 'rows' in data else data['dense']['path'];assert len(path)==105
  for x,h in data['source_input_sha256'].items():assert sha(x)==h,x
  initial=state(path[0]);final=state(path[-1]);w0,a0=allocation(initial);we,ae=allocation(final);assert a0['passed'] and ae['passed'];r0=a0['right_fraction'];rend=ae['right_fraction'];tau0,error0=torques(initial,w0);nodes=[];finite_errors=[error0]
  for k,e in enumerate(path):
   st=state(e);u=e['u'];f=u*u*(3-2*u)
   right=(1-f)*r0 if e['phase']=='shift' else (0. if e['phase']=='swing' else f*rend)
   w,a=allocation(st,right);node=dict(knot=k,phase=e['phase'],u=u,planned_right_load_fraction=right,allocation=a)
   if w is not None:
    tau,error=torques(st,w);finite_errors.append(error);cmd=command0.copy();cmd[moving]+=st[0][moving]-q0[moving]+(tau-tau0)/kp;assert np.array_equal(cmd[fixed],command0[fixed]) and np.array_equal(cmd[fingers],command0[fingers]);margin=float(min(min(cmd[j]-limits[names[j]][0],limits[names[j]][1]-cmd[j]) for j in moving));exceed=np.where(abs(tau)>effort+1e-9)[0]
    node.update(command=cmd.tolist(),planned_static_torque_Nm=tau.tolist(),corner_fractions=w.tolist(),corner_world_points=st[4].tolist(),corner_sides=st[5],joint_command_margin_rad=margin,torque_limit_exceed_names=[names[moving[i]] for i in exceed],maximum_effort_utilization=float(max(abs(tau)/effort)),finite_difference_max_error=error,static_model_passed=bool(a['passed'] and not len(exceed) and margin>=.03))
   else:node['static_model_passed']=False
   nodes.append(node)
  complete=all('command' in n for n in nodes);arc=None
  if complete:
   Q=np.array([n['command'] for n in nodes]);assert np.max(abs(Q[0]-command0))<1e-9;arc=float(np.max(abs(np.diff(Q,axis=0)),axis=1).sum())
  row=dict(source=str(p),source_sha256=sha(p),nodes=nodes,force_allocation_failures=[n['knot'] for n in nodes if not n['allocation']['passed']],failed_static_command_nodes=[n['knot'] for n in nodes if not n['static_model_passed']],max_finite_difference_error=max(finite_errors),moving_joint_names=[names[j] for j in moving],kp_Nm_rad=kp.tolist(),configured_effort_limits_Nm=effort.tolist(),all14_finger_commands_exact=True,initial_command_exact=complete,command_arc_rad=arc,minimum_time_for_this_command_polyline_at015_s=arc/.15 if arc is not None else None,scope='Specific anchored quasi-static command construction: recorded command0 plus desired pose change and full moving-joint gravity/reaction bias change. Corner vertical forces satisfy 3D static force/moment balance; prescribed right load fades to zero before swing and rises after planned touchdown. No measured contact or dynamics established. No seat load is assumed. The minimum traversal time applies only to this command polyline, not every possible controller.',physical_admission=False)
  rows.append(row);print(json.dumps({k:v for k,v in row.items() if k!='nodes'}),flush=True)
 sources=[Path(__file__),mp,ip,trp,*paths,Path('IsaacLab/source/isaaclab_assets/isaaclab_assets/robots/unitree.py'),Path('sitting_task/src/humanoid_sitting/tasks/full_transition.py')];out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='Offline additional necessary force/actuator checks. Failed model paths remain failed. No simulator/optimizer/holdout and no executable controller. Static vertical forces are a model assumption, not sensed support or friction/dynamic certification. No physical criteria or actuator setting modified.',rows=rows,source_sha256={str(p.resolve()):sha(p) for p in sources},physics_ticks=0,optimizer_updates=0,physical_admission=False)
 (O/'results.json').write_text(json.dumps(out,indent=2));(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2))
if __name__=='__main__':main()
