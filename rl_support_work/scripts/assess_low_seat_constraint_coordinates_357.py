"""Offline dependent-coordinate feasibility assessment; no path/time optimization or physics.
Tests elimination of 14 kinematic equalities with rootxy+12 legs at fixed phase f,p.
All three saved paths retained; bounded local Newton projection only on failed351 poses.
Numerical conditioning depends on the explicitly declared nondimensionalization.
"""
from pathlib import Path
import json,hashlib,datetime
import numpy as np
from scipy.spatial.transform import Rotation
from humanoid_sitting.shared_frame import BodyModel,apply,transform
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def smooth(u):return u*u*(3-2*u)
class Coordinates:
 def __init__(self):
  self.mp=D/'cooked_model_50/model.json';self.ip=D.parent/'crouch_rest_bridge_01/balanced_standing10_01/static_inertial_model.json';self.basefile=D/'clearance_lift_path_330/results.json'
  self.m=BodyModel(self.mp);self.i=read(self.ip);self.base=read(self.basefile);self.path=self.base['dense']['path'];self.names=self.m.names
  self.legs=[self.names.index(s+'_'+j+'_joint') for s in ['left','right'] for j in ['hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch','ankle_roll']]
  self.arms=[self.names.index(s+'_'+j+'_joint') for s in ['left','right'] for j in ['shoulder_pitch','elbow_pitch']]
  self.q0=np.array(self.path[0]['q']);self.origin=np.array(self.path[0]['root'][:3]);self.R0=Rotation.from_quat(self.path[0]['root'][3:]).as_matrix();self.end=np.array(self.base['endpoint']['root']);self.axis=Rotation.from_matrix(self.R0.T@Rotation.from_quat(self.end[3:]).as_matrix()).as_rotvec();self.lean=self.base['dense']['parameters'][2]
  lim={j['name']:np.deg2rad(j['limits']) for j in read(self.mp)['joints'] if j['type']=='PhysicsRevoluteJoint'}
  self.lower=np.r_[[-1.,-1.],[lim[self.names[j]][0]+.031 for j in self.legs]];self.upper=np.r_[[1.,1.],[lim[self.names[j]][1]-.031 for j in self.legs]]
  self.targets=[]
  for e in self.path:
   T=transform(Rotation.from_quat(e['root'][3:]).as_matrix(),np.array(e['root'][:3]));fk=self.m.fk(np.array(e['q']));self.targets.append(([T@fk[s+'_ankle_roll_link'] for s in ['left','right']],np.array(e['com'][:2])))
  self.variable_scale=np.r_[[.1,.1],np.ones(12)];self.residual_scale=np.r_[[.1]*3+[.2]*3,[.1]*3+[.2]*3,[.1]*2]
 def encode(self,e):return np.r_[e['root'][:2],np.array(e['q'])[self.legs]],np.array([e['approach_fraction'],e.get('orientation_fraction',e['approach_fraction'])])
 def state(self,k,z,p):
  e=self.path[k];u=e['u'];blend=smooth(u) if e['phase']=='shift' else 1. if e['phase']=='swing' else 1-smooth(u) if e['phase']=='approach' else 0.
  R=Rotation.from_rotvec([self.lean*blend,0,0]).as_matrix()@self.R0@Rotation.from_rotvec(p[1]*self.axis).as_matrix();root=np.r_[z[:2],self.origin[2]+p[0]*(self.end[2]-self.origin[2])];q=self.q0.copy();q[self.legs]=z[2:];q[self.arms]=self.q0[self.arms]-.35*p[0];T=transform(R,root);fk=self.m.fk(q);C=np.average([apply(T@fk[b],v) for b,v in zip(self.i['body_names'],self.i['local_com'])],axis=0,weights=self.i['mass']);return q,T,fk,C
 def residual(self,k,z,p):
  q,T,fk,C=self.state(k,z,p);feet,xy=self.targets[k];out=[]
  for s,F0 in zip(['left','right'],feet):
   F=T@fk[s+'_ankle_roll_link'];A=F0[:3,:3].T@F[:3,:3];out.extend(F[:3,3]-F0[:3,3]);out.extend(.1*np.array([A[2,1]-A[1,2],A[0,2]-A[2,0],A[1,0]-A[0,1]]))
  out.extend(C[:2]-xy);return np.array(out)
 def jacobian(self,k,z,p,h=1e-6):
  A=np.empty((14,14));B=np.empty((14,2))
  for j in range(14):
   zp=z.copy();zm=z.copy();zp[j]+=h;zm[j]-=h;A[:,j]=(self.residual(k,zp,p)-self.residual(k,zm,p))/(2*h)
  for j in range(2):
   pp=p.copy();pm=p.copy();pp[j]+=h;pm[j]-=h;B[:,j]=(self.residual(k,z,pp)-self.residual(k,z,pm))/(2*h)
  return A,B
 def project(self,k,z,p):
  x=z.copy();history=[];reason='iteration_limit';tol=1e-10
  for it in range(12):
   r=self.residual(k,x,p);history.append(float(np.max(abs(r))))
   if history[-1]<=tol:reason='converged';break
   A,_=self.jacobian(k,x,p);step=np.linalg.solve(A,-r);ok=False
   for j in range(9):
    scale=2.**(-j);v=x+scale*step
    if np.any(v<self.lower) or np.any(v>self.upper):continue
    rr=self.residual(k,v,p)
    if np.linalg.norm(rr/self.residual_scale)<np.linalg.norm(r/self.residual_scale):x=v;ok=True;break
   if not ok:reason='no_bounded_decreasing_newton_step';break
  r=self.residual(k,x,p);passed=bool(max(abs(r))<=tol)
  if passed:reason='converged'
  return x,dict(knot=k,converged=passed,reason=reason,iterations=len(history),initial_max_residual=history[0],final_max_residual=float(max(abs(r))),minimum_joint_planning_margin=float(np.min(np.minimum(x[2:]-self.lower[2:],self.upper[2:]-x[2:]))),root_shift_m=(x[:2]-z[:2]).tolist(),max_leg_change_rad=float(max(abs(x[2:]-z[2:]))),history=history)
def main():
 O=D/'constraint_coordinates_357';assert not O.exists();O.mkdir();M=Coordinates();sources=[D/'clearance_lift_path_330/results.json',D/'multiknot_path_344/results.json',D/'guarded_path_351/results.json'];rows=[];As=[];Bs=[];states=[];phases=[];residuals=[];tangents=[];svs=[];projected=[];proj=[];Jhalf=[]
 for src in sources:
  path=read(src)['dense']['path']
  for k in range(1,104):
   z,p=M.encode(path[k]);r=M.residual(k,z,p);A,B=M.jacobian(k,z,p);Ahalf,Bhalf=M.jacobian(k,z,p,5e-7);err=float(max(abs(A-Ahalf).max(),abs(B-Bhalf).max()));assert err<2e-6
   scaled=A*M.variable_scale[None,:]/M.residual_scale[:,None];sv=np.linalg.svd(scaled,compute_uv=False);rank=int((sv>1e-10).sum());tangent=np.linalg.solve(A,-B) if rank==14 else np.full((14,2),np.nan)
   rows.append(dict(source=src.parent.name,knot=k,rank=rank,dimensionless_condition=float(sv[0]/sv[-1]),minimum_singular_value=float(sv[-1]),max_residual=float(max(abs(r))),half_step_derivative_error=err,implicit_tangent_residual=float(abs(A@tangent+B).max()) if rank==14 else None))
   As.append(A);Bs.append(B);states.append(z);phases.append(p);residuals.append(r);tangents.append(tangent);svs.append(sv);Jhalf.append(err)
   if src.parent.name=='guarded_path_351':
    zz,rr=M.project(k,z,p);projected.append(zz);proj.append(rr)
 print('Jacobian and bounded projection records complete',flush=True)
 q=np.array(projected);projection_source=read(sources[-1])['dense']['path'];reconstruction=[]
 for src in sources[:2]:
  path=read(src)['dense']['path'];reconstruction.append(float(max(max(abs(M.residual(k,*M.encode(path[k])))) for k in range(1,104))))
 result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope=__doc__,sources=[str(p) for p in sources],nodes_per_source=103,total_nodes=len(rows),dependent_coordinates=['root_x_m','root_y_m']+[M.names[j] for j in M.legs],independent_coordinates=['descent_fraction','orientation_fraction'],rank_threshold=1e-10,rank_threshold_is_numerical=True,variable_scale=M.variable_scale.tolist(),residual_scale=M.residual_scale.tolist(),all_rank14=all(r['rank']==14 for r in rows),maximum_dimensionless_condition=max(r['dimensionless_condition'] for r in rows),maximum_half_step_derivative_error=max(Jhalf),feasible_source_reconstruction_max=reconstruction,rows=rows,projection=dict(scope='One bounded Newton projection per saved351 knot1..103; phases fixed, rootxy/legs vary; no force/time/path objective. At most12 iterations and9 fixed halvings each, original planning joint bounds. Numerical residual tolerance1e-10, not changed physical screens. Failures are local algorithm outcomes, not global infeasibility.',rows=proj,converged=sum(r['converged'] for r in proj),failed_knots=[r['knot'] for r in proj if not r['converged']],complete_geometry_screen=False,command_or_timing_screen=False),source_sha256={str(p.resolve()):sha(p) for p in sources+[M.mp,M.ip,Path(__file__)]},physics_ticks=0,policy_optimizer_updates=0,physical_admission=False,repair_complete=False)
 np.savez_compressed(O/'arrays.npz',A=As,B=Bs,z=states,p=phases,residual=residuals,tangent=tangents,singular_values=svs,projected_z=projected);(O/'results.json').write_text(json.dumps(result,indent=2));(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ['rows','projection','source_sha256']},indent=2));print(json.dumps({'projection_converged':result['projection']['converged'],'failed_knots':result['projection']['failed_knots']}),flush=True)
if __name__=='__main__':main()
