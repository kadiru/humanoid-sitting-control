"""Independent analytical body-twist audit of357 central-difference coordinates.
No optimization/projection rerun, physical criteria change or runtime admission.
"""
from pathlib import Path
import json,hashlib,datetime
import numpy as np
from scipy.spatial.transform import Rotation
from humanoid_sitting.shared_frame import BodyModel,apply,transform
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def cross(a,b):return np.array([a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]])
def skew(a):return np.array([[0,-a[2],a[1]],[a[2],0,-a[0]],[-a[1],a[0],0]])
def main():
 O=D/'constraint_coordinates_audit_358';assert not O.exists();O.mkdir();p=D/'constraint_coordinates_357';r=read(p/'results.json');z=np.load(p/'arrays.npz');arrays={k:z[k] for k in z.files}
 for f,h in r['source_sha256'].items():assert sha(f)==h
 m=BodyModel(D/'cooked_model_50/model.json');meta=read(D/'cooked_model_50/model.json');ine=read(D.parent/'crouch_rest_bridge_01/balanced_standing10_01/static_inertial_model.json');base=read(D/'clearance_lift_path_330/results.json');original=base['dense']['path'];mass=np.array(ine['mass']);sides=['left','right'];legs=[m.names.index(s+'_'+j+'_joint') for s in sides for j in ['hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch','ankle_roll']];arms=[m.names.index(s+'_'+j+'_joint') for s in sides for j in ['shoulder_pitch','elbow_pitch']];joints={j['name']:j for j in m.joints};parent={j['body1'][0]:j['body0'][0] for j in m.joints if j['body0']}
 def below(body,ancestor):
  while True:
   if body==ancestor:return True
   if body not in parent:return False
   body=parent[body]
 def state(e):
  q=np.array(e['q']);T=transform(Rotation.from_quat(e['root'][3:]).as_matrix(),np.array(e['root'][:3]));fk=m.fk(q);points={b:apply(T@fk[b],v) for b,v in zip(ine['body_names'],ine['local_com'])};C=sum(mm*points[b] for mm,b in zip(mass,ine['body_names']))/mass.sum();return q,T,fk,points,C
 targets=[]
 for e in original:
  _,T,fk,_,C=state(e);targets.append(([T@fk[s+'_ankle_roll_link'] for s in sides],np.array(e['com'][:2])))
 R0=Rotation.from_quat(original[0]['root'][3:]).as_matrix();end=base['endpoint']['root'];axis=Rotation.from_matrix(R0.T@Rotation.from_quat(end[3:]).as_matrix()).as_rotvec();dz=end[2]-original[0]['root'][2]
 def residual(e,k):
  _,T,fk,_,C=state(e);v=[]
  for s,F0 in zip(sides,targets[k][0]):
   F=T@fk[s+'_ankle_roll_link'];A=F0[:3,:3].T@F[:3,:3];v.extend(F[:3,3]-F0[:3,3]);v.extend([.1*(A[2,1]-A[1,2]),.1*(A[0,2]-A[2,0]),.1*(A[1,0]-A[0,1])])
  v.extend(C[:2]-targets[k][1]);return np.array(v)
 def analytic(e,k):
  q,T,fk,points,C=state(e);joint_geo={}
  for j in legs+arms:
   name=m.names[j];jj=joints[name];J=T@fk[jj['body0'][0]]@m.frames[name][0];joint_geo[j]=(J[:3,'XYZ'.index(jj['axis'])],J[:3,3],jj['body1'][0])
  A=np.zeros((14,16))
  for col in range(16):
   root_v=np.eye(3)[:,col] if col<2 else np.array([0.,0.,dz]) if col==14 else np.zeros(3);rot_v=T[:3,:3]@axis if col==15 else np.zeros(3);active=[(legs[col-2],1.)] if 2<=col<14 else [(j,-.35) for j in arms] if col==14 else []
   def velocity(body,P):
    v=root_v+cross(rot_v,P-T[:3,3]);w=rot_v.copy()
    for j,weight in active:
     ax,origin,child=joint_geo[j]
     if below(body,child):v+=weight*cross(ax,P-origin);w+=weight*ax
    return v,w
   for t,s in enumerate(sides):
    body=s+'_ankle_roll_link';F=T@fk[body];v,w=velocity(body,F[:3,3]);A[t*6:t*6+3,col]=v;dR=targets[k][0][t][:3,:3].T@skew(w)@F[:3,:3];A[t*6+3:t*6+6,col]=.1*np.array([dR[2,1]-dR[1,2],dR[0,2]-dR[2,0],dR[1,0]-dR[0,1]])
   dc=sum(mm*velocity(b,points[b])[0] for mm,b in zip(mass,ine['body_names']))/mass.sum();A[12:,col]=dc[:2]
  return A[:,:14],A[:,14:]
 mx=dict(residual=0.,dependent_jacobian=0.,phase_jacobian=0.,implicit_tangent=0.,dimensionless_condition=0.,projected_residual=0.);count=0
 for src in r['sources']:
  path=read(src)['dense']['path']
  for k in range(1,104):
   e=path[k];A,B=analytic(e,k);rr=residual(e,k);a=arrays['A'][count];b=arrays['B'][count];mx['dependent_jacobian']=max(mx['dependent_jacobian'],float(abs(A-a).max()));mx['phase_jacobian']=max(mx['phase_jacobian'],float(abs(B-b).max()));mx['residual']=max(mx['residual'],float(abs(rr-arrays['residual'][count]).max()));tan=np.linalg.lstsq(A,-B,rcond=None)[0];mx['implicit_tangent']=max(mx['implicit_tangent'],float(abs(tan-arrays['tangent'][count]).max()));scaled=A*np.array(r['variable_scale'])[None,:]/np.array(r['residual_scale'])[:,None];sv=np.linalg.svd(scaled,compute_uv=False);assert int(sum(sv>r['rank_threshold']))==r['rows'][count]['rank'];mx['dimensionless_condition']=max(mx['dimensionless_condition'],abs(float(sv[0]/sv[-1])-r['rows'][count]['dimensionless_condition']));assert r['rows'][count]['knot']==k and r['rows'][count]['source']==Path(src).parent.name;count+=1
 assert count==309;assert mx['residual']<1e-12;assert max(mx['dependent_jacobian'],mx['phase_jacobian'])<2e-6;assert mx['implicit_tangent']<2e-5;assert mx['dimensionless_condition']<.01
 lim={j['name']:np.deg2rad(j['limits']) for j in meta['joints'] if j['type']=='PhysicsRevoluteJoint'};failed=[];source=read(r['sources'][2])['dense']['path']
 for j,row in enumerate(r['projection']['rows']):
  k=row['knot'];e=json.loads(json.dumps(source[k]));zz=arrays['projected_z'][j];e['root'][:2]=zz[:2].tolist()
  for idx,value in zip(legs,zz[2:]):e['q'][idx]=float(value);assert lim[m.names[idx]][0]+.031<=value<=lim[m.names[idx]][1]-.031
  rr=residual(e,k);err=abs(float(max(abs(rr)))-row['final_max_residual']);mx['projected_residual']=max(mx['projected_residual'],err);assert bool(max(abs(rr))<=1e-10)==row['converged'];assert len(row['history'])<=12
  if not row['converged']:failed.append(k)
 assert failed==r['projection']['failed_knots'];assert mx['projected_residual']<1e-12
 out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,scope=__doc__,records=309,analytical_dependent_columns=309*14,analytical_phase_columns=309*2,projected_records=103,maximum_errors=mx,projection_failed_knots=failed,numerical_bounds=dict(residual=1e-12,jacobian=2e-6,tangent=2e-5,condition_absolute=.01),physical_admission=False,repair_complete=False,physics_ticks=0,policy_optimizer_updates=0,source_sha256={str(p/'results.json'):sha(p/'results.json'),str(p/'arrays.npz'):sha(p/'arrays.npz'),str(Path(__file__).resolve()):sha(__file__)})
 (O/'independent_audit.json').write_text(json.dumps(out,indent=2));(O/'SHA256.json').write_text(json.dumps({f.name:sha(f) for f in O.iterdir() if f.is_file()},indent=2));print(json.dumps(out,indent=2),flush=True)
if __name__=='__main__':main()
