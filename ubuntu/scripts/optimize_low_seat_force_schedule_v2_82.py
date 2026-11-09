"""Offline bounded force allocation/schedule for the unchanged audited path77.
One linear program; no pose search, runtime, physics or physical admission.
"""
from pathlib import Path
import json,hashlib,datetime
import numpy as np
from scipy.optimize import linprog
from scipy.spatial.transform import Rotation
from scipy.sparse import lil_matrix,csr_matrix
from humanoid_sitting.shared_frame import BodyModel,apply,transform
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');O=D/'force_schedule_screen_v2_82'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 O.mkdir(exist_ok=False);source=D/'actuator_contact_path_refinement_77/results.json';data=read(source);assert read(D/'actuator_path_refinement_audit_79/independent_audit.json')['passed'];z=data['dense'];assert z['model_geometry_passed'] and z['model_command_passed'];mp=D/'cooked_model_50/model.json';ip=D.parent/'crouch_rest_bridge_01/balanced_standing10_01/static_inertial_model.json';m=BodyModel(mp);ine=read(ip);W=sum(ine['mass'])*9.81;names=m.names
 moving=[names.index(s+'_'+j+'_joint') for s in ['left','right'] for j in ['hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch','ankle_roll']]+[names.index(s+'_'+j+'_joint') for s in ['left','right'] for j in ['shoulder_pitch','elbow_pitch']];kp=np.array([200,150,150,200,160,160]*2+[40]*4);effort=np.array([300,300,300,300,20,20]*2+[300]*4);limits={j['name']:np.deg2rad(j['limits']) for j in m.joints if j['type']=='PhysicsRevoluteJoint'};by={j['name']:j for j in m.joints};children={}
 for j in m.joints:
  if j['body0']:children.setdefault(j['body0'][0],[]).append(j['body1'][0])
 def descend(b):
  ans={b}
  for c in children.get(b,[]):ans|=descend(c)
  return ans
 sub={n:descend(by[n]['body1'][0]) for n in names};N=len(z['path']);assert N==105;counts=[len(v['corner_fractions']) for v in z['command_nodes']];offset=np.r_[0,np.cumsum(counts)];nw=offset[-1];nv=nw+N-1;eq=[];beq=[];ub=[];bub=[];nodes=[];cmd0=np.array(z['command_nodes'][0]['command']);q0=np.array(z['path'][0]['q']);tau0=np.array(z['command_nodes'][0]['planned_static_torque_Nm'])
 def add(row,b,equality=False):
  (eq if equality else ub).append(row);(beq if equality else bub).append(float(b))
 for k,(e,v) in enumerate(zip(z['path'],z['command_nodes'])):
  q=np.array(e['q']);root=e['root'];T=transform(Rotation.from_quat(root[3:]).as_matrix(),root[:3]);fk=m.fk(q);P=np.array(v['corner_world_points']);sides=v['corner_sides'];C=np.array(e['com']);grav=[];B=[]
  for j in moving:
   joint=by[names[j]];J=T@fk[joint['body0'][0]]@m.frames[names[j]][0];axis=J[:3,'XYZ'.index(joint['axis'])];G=sum((-axis@np.cross(apply(T@fk[b],cc)-J[:3,3],[0,0,-mm*9.81])) for b,mm,cc in zip(ine['body_names'],ine['mass'],ine['local_com']) if b in sub[names[j]]);grav.append(G);B.append([-axis@np.cross(p-J[:3,3],[0,0,W]) if s+'_ankle_roll_link' in sub[names[j]] else 0. for p,s in zip(P,sides)])
  grav=np.array(grav);B=np.array(B);A=np.vstack([np.ones(len(P)),P[:,:2].T,np.array([s=='right' for s in sides])]);b=np.r_[1.,C[:2],v['planned_right_load_fraction']];sl=slice(offset[k],offset[k+1])
  for ar,bb in zip(A,b):row=np.zeros(nv);row[sl]=ar;add(row,bb,True)
  base=cmd0[moving]+q[moving]-q0[moving]+(grav-tau0)/kp;M=B/kp[:,None]
  for i,j in enumerate(moving):
   row=np.zeros(nv);row[sl]=M[i];lo,hi=limits[names[j]];add(row,hi-.03-base[i]);add(-row,base[i]-(lo+.03));row=np.zeros(nv);row[sl]=B[i];add(row,effort[i]-grav[i]);add(-row,effort[i]+grav[i])
  wold=np.array(v['corner_fractions']);assert max(abs(grav+B@wold-v['planned_static_torque_Nm']))<1e-10
  nodes.append(dict(B=B,gravity=grav,base=base,M=M,slice=sl,eq=A,b=b))
  if k==0 or k==N-1:
   for i,ww in enumerate(wold):row=np.zeros(nv);row[offset[k]+i]=1;add(row,ww,True)
  if k:
   prev=nodes[k-1]
   for i in range(len(moving)):
    row=np.zeros(nv);row[sl]=M[i];row[prev['slice']]-=prev['M'][i];row[nw+k-1]=-.15;delta=base[i]-prev['base'][i];add(row,-delta);rev=-row;rev[nw+k-1]=-.15;add(rev,delta)
 objective=np.zeros(nv);objective[nw:]=1.;fit=linprog(objective,A_ub=csr_matrix(np.array(ub)),b_ub=bub,A_eq=csr_matrix(np.array(eq)),b_eq=beq,bounds=[(0,1)]*nw+[(0,None)]*(N-1),method='highs',options=dict(primal_feasibility_tolerance=1e-10,dual_feasibility_tolerance=1e-10));rows=[];summary=dict(success=bool(fit.success),message=fit.message)
 if fit.success:
  commands=[];maxeq=0.;maxwork=0.;margins=[];use=[]
  for k,((e,v),a) in enumerate(zip(zip(z['path'],z['command_nodes']),nodes)):
   w=fit.x[a['slice']];tau=a['gravity']+a['B']@w;cmd=cmd0.copy();cmd[moving]=a['base']+a['M']@w
   if k==0:cmd=cmd0.copy()
   err=max(abs(a['eq']@w-a['b']));maxeq=max(maxeq,err);margin=min(min(cmd[j]-limits[names[j]][0],limits[names[j]][1]-cmd[j]) for j in moving);margins.append(margin);use.append(max(abs(tau)/effort));commands.append(cmd);rows.append(dict(knot=k,corner_fractions=w.tolist(),command=cmd.tolist(),torque_Nm=tau.tolist(),max_equilibrium_residual=float(err),command_margin_rad=float(margin),fixed_commands_exact=bool(np.array_equal(np.delete(cmd,moving),np.delete(cmd0,moving)))))
  arc=float(np.max(abs(np.diff(commands,axis=0)),axis=1).sum());duration=arc/.15;summary.update(unchanged_residual_checks_passed=bool(maxeq<1e-8 and min(margins)>=.03-1e-8 and max(use)<=1+1e-8),command_arc_rad=arc,minimum_command_polyline_time_s=duration,original_time_s=z['minimum_time_for_this_command_polyline_at015_s'],max_equilibrium_residual=float(maxeq),min_command_margin_rad=float(min(margins)),maximum_static_effort_utilization=float(max(use)),planning_allowance_s=13.5,command_time_gate=bool(duration<=13.5),actual_filter_or_dynamic_check=False)
 files=[Path(__file__).resolve(),source,D/'actuator_path_refinement_audit_79/independent_audit.json',mp,ip];out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='One OFFLINE linear program changes only nonnegative foot corner force distribution on the unchanged audited105-knot path77. Geometry, prescribed total right load at each knot, endpoints, initial command, all14fingers, joint margins, configured effort limits and0.15rad/s remain fixed. Objective minimizes specific command-polyline time. This tests the arbitrary per-frame minimum-norm force allocation used earlier; optimized forces are assumed, not sensed or validated by a controller. No seated contact added, no physical feasibility/admission claim.',summary=summary,nodes=rows,source_sha256={str(p):sha(p) for p in files},physics_ticks=0,optimizer_updates=0,holdout_used=False,physical_admission=False);(O/'results.json').write_text(json.dumps(out,indent=2));(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2));print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
