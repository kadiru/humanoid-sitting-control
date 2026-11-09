import os,sys,time,json,hashlib,traceback
START=time.monotonic();os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
R=Path('/home/kadir/humanoid-sitting');E=R/'experiments';H=Path(__file__).resolve().parent;target=sys.argv[1]
sys.path[:0]=[str(H),str(E/'armchair_rest_ppo_20261105_v1'),str(E/'furniture_collision_repair_20261106_v7'),str(E/'furniture_collision_repair_20261106_v5'),str(E/'furniture_collision_repair_20261106_v4'),str(E/'furniture_perception_repair_20261029_v1'),str(R/'sitting_task/src')]
read=lambda p:json.loads(Path(p).read_text());out=H/'encoded_results'/target;out.mkdir(parents=True,exist_ok=False)
def write(n,v):
 with (out/n).open('x') as f:json.dump(v,f,indent=2,allow_nan=False,default=lambda x:x.tolist() if hasattr(x,'tolist') else x)
try:
 freeze=read(H/'encoded_freeze.json')
 for p,h in freeze['files'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
 import numpy as np
 from scipy.spatial import ConvexHull
 from scipy.spatial.transform import Rotation,Slerp
 from humanoid_sitting.shared_frame import BodyModel,apply,transform,level_rotation
 from encoded_full_body import FullRest
 from staged_lean import base
 from visible_surface import expand
 from contact_surface import associate
 from audit_full_mesh import audit as collision_audit
 model=BodyModel(R/'runs/g1_kinematics_geometry_04/model.json')
 inertia=read('/data/humanoid-sitting-perception/crouch_rest_bridge_01/balanced_standing10_01/static_inertial_model.json')
 shapes=read(E/'furniture_perception_repair_20261029_v1/render_colliders.json')['shapes']
 if target=='lean_center':
  p=R/'outputs/armchair_rest_ppo_20261105_v1/demo_rear_back_center';f=dict(np.load(p/'frame_0.npz'))
  with np.load(p/'trajectory.npz') as data:previous=data['previous'][0,0].astype(float)
  old=base.LeanGeometry(model,f['q'].astype(float),f['gravity'].astype(float),f['geometry'].astype(float),f['selected_points'],previous)
  old.clip_world[:-2,3]+=.002
  rt,fk=old.pose(old.start);_,point,_,_=old.geometry(old.start)
  owners=[]
  for shape in shapes:
   b=shape['body']
   if b not in ['torso_link','logo_link']:continue
   v=np.asarray(shape['vertices']);local=apply(np.linalg.inv(rt@fk[b]),point);tri=v[ConvexHull(v).simplices]
   u=tri[:,1]-tri[:,0];w=tri[:,2]-tri[:,0];d=local-tri[:,0]
   uu=(u*u).sum(1);ww=(w*w).sum(1);uw=(u*w).sum(1);ud=(u*d).sum(1);wd=(w*d).sum(1);den=uu*ww-uw*uw
   a=(ww*ud-uw*wd)/den;c=(uu*wd-uw*ud)/den;valid=(a>=-1e-8)&(c>=-1e-8)&(a+c<=1+1e-8)
   error=np.linalg.norm(d-a[:,None]*u-c[:,None]*w,axis=1);error[~valid]=np.inf
   if error.min()<1e-7:owners.append((float(error.min()),b,local))
  assert owners,'MATERIAL_CONTACT_OWNER_NOT_FOUND';_,body,marker=min(owners,key=lambda a:(a[0],a[1]))
  n=old.n;off=old.off;patch=old.cloud;goal=point-(point@n+off)*n
  q=f['q'];gravity=f['gravity'];initial=f['root_position'];basis=f['root_rotation']@old.level.T;task='leaning'
  case=read(p/'manifest.json')['case'];furniture=dict(np.load(case['mesh_npz']));allowed=read(Path(case['mesh_npz']).with_name('target_region.json'))['triangle_indices']
  association=dict(body=body,marker_local=marker.tolist(),source='Current standing camera finite back intersection',independent_material_triangle_error_m=min(x[0] for x in owners))
 elif target=='stool_center':
  case=next(c for c in read(E/'furniture_examples_20261105_v1/cases.json') if c['id']=='wooden_stool_01_top')
  meshpath=Path(case['mesh_npz']);furniture=dict(np.load(meshpath));rot=Rotation.from_euler('z',45,degrees=True).as_matrix();center=(furniture['vertices'].min(0)+furniture['vertices'].max(0))/2;center[2]=0.
  furniture['vertices']=(furniture['vertices']-center)@rot.T+center;allowed=read(meshpath.with_name('target_region.json'))['triangle_indices']
  p=R/'outputs/furniture_examples_20261105_v2/sit_wooden_stool_01_center';f=dict(np.load(p/'frame_0.npz'));previous=np.load(p/'startup.npz')['command'][0].astype(float)
  T=np.eye(4);T[:3,:3]=[[0,.707106781,-.707106781],[1,0,0],[0,-.707106781,-.707106781]];T[:3,3]=[-.07,0,.38]
  points,_,_=expand(model,f['K'],T,f['depth'],f['q'],f['gravity'],f['selected_pixels']);level=level_rotation(f['gravity'].astype(float));basis=f['root_rotation']@level.T
  placement=read(E/'furniture_rest_repair_20261107_v1/seat_models/sit_wooden_stool_01_center.placement.json');assert placement['independent_all_triangle_clip_audit']['passed'];shift=placement['closer_m'];translation=level.T@np.array([shift,0,0])
  world=points@f['root_rotation'].T+f['root_position'];world=(world-center)@rot.T+center;points=(world-f['root_position'])@f['root_rotation']+translation
  geometry=f['geometry'].astype(float).copy();geometry[5]-=geometry[2:5]@translation;geometry[6:9]+=translation
  value,local,report,cloud,height,goal=associate(model,f['q'],f['gravity'],geometry,points,'top')
  patch=cloud;n=level@value[2:5];off=value[5];q=f['q'];gravity=f['gravity'];initial=f['root_position']-basis@np.array([shift,0,0]);task='sitting'
  ref=R/'outputs/full_body_contact_imitation_v4/capture_known_old';assert read(ref/'demonstration_audit.json')['admitted'][8]
  data=np.load(ref/'demonstrations_50hz.npz');g=read(ref/'geometry.json')[8];refnormal=np.asarray(g['normal']);body_index=int(np.argmax(data['force'][999,8,:3]@refnormal));body=['pelvis','left_hip_yaw_link','right_hip_yaw_link'][body_index]
  refroot=data['root'][999,8];refrt=transform(Rotation.from_quat(refroot[3:]).as_matrix(),refroot[:3]);reffk=model.fk(data['q_post'][999,8]);choices=[]
  for shape in shapes:
   if shape['body']!=body:continue
   v=np.asarray(shape['vertices']);h=apply(refrt@reffk[body],v)@refnormal
   ids=np.flatnonzero(h<=h.min()+1e-8);choices.append((float(h.min()),v[ids].mean(0)))
  _,marker=min(choices,key=lambda x:x[0]);association=dict(body=body,marker_local=marker.tolist(),source='Admitted known_old/8/frame999; highest measured legal-body load, exact static hull support face',source_normal_force_N=float(data['force'][999,8,body_index]@refnormal),placement_inherited=True)
 else:raise ValueError('UNFROZEN_TARGET')
 write('association.json',association)
 plan=FullRest(model,inertia,shapes,q,gravity,previous,n,off,patch,body,marker,goal,task)
 result=plan.solve();write('plan.json',result)
 # Same physical contact classes; adapt audit's eligible set only by existing task.
 import audit_full_mesh
 audit_full_mesh.REST_BODIES={'torso_link','logo_link'} if task=='leaning' else {'pelvis','left_hip_yaw_link','right_hip_yaw_link'}
 nodes=[dict(z,phase='landing' if z['gap']<=.006 else 'approach') for z in result['nodes']]
 collision=collision_audit(model,shapes,furniture,allowed,nodes,basis,initial)
 foot_errors=[];normal_rates=[];loaded_drift=[]
 for a,b in zip(result['nodes'],result['nodes'][1:]):
  ra=np.asarray(a['root']);rb=np.asarray(b['root']);rotations=Slerp([0,1],Rotation.from_matrix(np.stack([ra[:3,:3],rb[:3,:3]])));last=None
  for u in np.linspace(0,1,9):
   q=(1-u)*np.asarray(a['q'])+u*np.asarray(b['q']);rt=transform(rotations(u).as_matrix(),(1-u)*ra[:3,3]+u*rb[:3,3]);fk=model.fk(q)
   for bb,anchor in zip(plan.foot_bodies,plan.feet):
    foot=rt@fk[bb];foot_errors.append(max(float(np.linalg.norm(foot[:3,3]-anchor[:3,3])),float(np.linalg.norm(Rotation.from_matrix(anchor[:3,:3].T@foot[:3,:3]).as_rotvec()))))
   point=apply(rt@fk[body],marker);gap=float(point@n+off)
   if b['phase']=='normal' and last is not None:normal_rates.append(abs(gap-last)/(float(b['seconds']-a['seconds'])/8))
   if b['phase']=='load':loaded_drift.append(float(np.linalg.norm(point-goal)))
   last=gap
 commands=np.array([z['command'] for z in result['nodes']]);times=np.array([z['seconds'] for z in result['nodes']]);prior=previous.copy();margin=1e6;increment=0.;finger=True
 for t in np.arange(.02,times[-1]+.019999,.02):
  targetcmd=np.array([np.interp(t,times,commands[:,j]) for j in range(37)]);desired=prior+(targetcmd-prior)/.1
  increment=max(increment,float(abs(targetcmd-prior).max()));margin=min(margin,float(np.min(np.r_[desired[:23]-plan.lower,plan.upper-desired[:23]])));finger &= np.array_equal(targetcmd[23:],previous[23:]);prior=targetcmd
 prior32=previous.astype(np.float32);max32=0.;margin32=1e6;finger32=True;encoding_error=0.
 for t in np.arange(.02,times[-1]+.02,.02):
  nominal=np.array([np.interp(t,times,commands[:,j]) for j in range(37)],dtype=np.float32)
  desired32=prior32+(nominal-prior32)/np.float32(.1)
  actual32=prior32+np.float32(.1)*(desired32-prior32)
  max32=max(max32,float(np.max(abs(actual32-prior32))))
  margin32=min(margin32,float(np.min(np.r_[desired32[:23].astype(float)-plan.lower,plan.upper-desired32[:23].astype(float)])))
  finger32 &= np.array_equal(actual32[23:],previous[23:].astype(np.float32))
  encoding_error=max(encoding_error,float(abs(actual32.astype(float)-nominal.astype(float)).max()));prior32=actual32
 tests=dict(float32_action_increment=max32<=.003,float32_desired_margin=margin32>=.03,float32_fingers=bool(finger32),encoded_settled_time=np.ceil(result['settled_seconds']/.02)*.02<14.4,original_action_increment=increment<=.003,original_desired_margin=margin>=.03,fingers_fixed=bool(finger),full_mesh=collision['passed'],hold_time_reserved=result['settled_seconds']<14.4,between_node_feet=max(foot_errors)<.003,normal_speed=max(normal_rates)<=.006,loaded_material_contact_drift=max(loaded_drift)<.002)
 row=dict(model_pass=all(tests.values()),float32_maximum_increment_rad=max32,float32_minimum_desired_margin_rad=margin32,float32_forward_filter_error_rad=encoding_error,tests=tests,collision=collision,maximum_increment_rad=increment,minimum_desired_margin_rad=margin,maximum_foot_error=max(foot_errors),maximum_normal_speed=max(normal_rates),maximum_loaded_material_point_drift=max(loaded_drift),model_effort_ratio=result['nodes'][-1]['model_effort_ratio'],return_model=result['return_model'])
except Exception as exc:
 row=dict(model_pass=False,error=repr(exc),traceback=traceback.format_exc())
 if 'plan' in locals() and hasattr(plan,'provisional'):write('partial_not_admitted.json',plan.provisional)
row.update(total_seconds=time.monotonic()-START,physics_launch_ready=False,physical_success=False,physics_steps=0,PPO_updates=0)
write('receipt.json',row);print(json.dumps({k:v for k,v in row.items() if k not in ['collision','traceback']}),flush=True)
