import os,sys,time,json,hashlib,traceback
START=time.monotonic();os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
R=Path('/home/kadir/humanoid-sitting');E=R/'experiments';H=Path(__file__).resolve().parent;target=sys.argv[1]
sys.path[:0]=[str(H),str(E/'rest_ready_20261108_v1'),str(E/'armchair_rest_ppo_20261105_v1'),str(E/'furniture_collision_repair_20261106_v7'),str(E/'furniture_collision_repair_20261106_v5'),str(E/'furniture_collision_repair_20261106_v4'),str(E/'furniture_perception_repair_20261029_v1'),str(R/'sitting_task/src')]
read=lambda p:json.loads(Path(p).read_text());out=H/'results'/target;out.mkdir(parents=True,exist_ok=False)
def write(n,v):
 with (out/n).open('x') as f:json.dump(v,f,indent=2,allow_nan=False,default=lambda x:x.tolist() if hasattr(x,'tolist') else x)
try:
 freeze=read(H/'freeze.json')
 for p,h in freeze['files'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
 import numpy as np
 from scipy.spatial import ConvexHull
 from scipy.spatial.transform import Rotation,Slerp
 from humanoid_sitting.shared_frame import BodyModel,apply,transform,level_rotation
 from terminal_plan import FullRest
 from camera_clearance import rasterize
 from compact_camera import CompactCamera as CameraClearance
 from contact_facet import select
 from observed_surface import ObservedPatch
 from paired_contact import BODIES,upper_vertices
 from seed import construct as original_seed
 from observed_clearance_v2 import observed_mesh
 from staged_lean import base
 from visible_surface import expand
 from contact_surface import associate
 from audit_full_mesh import audit as collision_audit
 T=np.eye(4);T[:3,:3]=[[0,.707106781,-.707106781],[1,0,0],[0,-.707106781,-.707106781]];T[:3,3]=[-.07,0,.38]
 model=BodyModel(R/'runs/g1_kinematics_geometry_04/model.json')
 inertia=read('/data/humanoid-sitting-perception/crouch_rest_bridge_01/balanced_standing10_01/static_inertial_model.json')
 shapes=read(E/'furniture_perception_repair_20261029_v1/render_colliders.json')['shapes']
 if target!='stool_edge':raise ValueError('UNFROZEN_FINITE_EDGE_TARGET')
 case=next(c for c in read(E/'furniture_examples_20261105_v1/cases.json') if c['id']=='wooden_stool_01_top')
 p=R/'outputs/furniture_examples_20261105_v2/sit_wooden_stool_01_center'
 f=dict(np.load(p/'frame_0.npz'));previous=np.load(p/'startup.npz')['command'][0].astype(float)
 q=f['q'];gravity=f['gravity'];level=level_rotation(gravity.astype(float).copy())
 n=level@f['geometry'][2:5].astype(float);off=float(f['geometry'][5]);n/=np.linalg.norm(n)
 if n[2]<.97:raise ValueError('STOOL_TOP_NORMAL_NOT_UPRIGHT')
 patch=np.asarray(f['selected_points'],float)@level.T
 initial=f['root_position'];basis=f['root_rotation']@level.T;task='leaning'
 furniture=dict(np.load(case['mesh_npz']));allowed=read(Path(case['mesh_npz']).with_name('target_region.json'))['triangle_indices']
 rt=transform(level);fk=model.fk(q);body=BODIES[0];markers=[];points=[]
 for bb in BODIES:
  vertices=upper_vertices(model,shapes,bb);world=apply(rt@fk[bb],vertices);height=world@n+off
  mm=vertices[height<=height.min()+1e-10].mean(0);markers.append(mm);points.append(apply(rt@fk[bb],mm))
 marker=np.array(markers);point=np.mean(points,axis=0);goal=patch.mean(0)
 association=dict(bodies=list(BODIES),native_object=case['id'],surface='Actually observed stool top/edge',source='Original untransformed stool frame and exact native furniture mesh; no height/scale/material change',intent='Proximal-thigh support; sitting vs upright lean remains unclassified until physical posture evaluated',physical_success=False)
 ref=R/'outputs/full_body_contact_imitation_v4/capture_known_old'
 demo_audit=read(ref/'demonstration_audit.json');assert demo_audit['admitted'][8]
 with np.load(ref/'demonstrations_50hz.npz') as demos:reference_q=demos['q_post'][999,8].astype(float).copy()
 reference_provenance=dict(trajectory=str(ref/'demonstrations_50hz.npz'),audit=str(ref/'demonstration_audit.json'),case=8,frame=999,admitted_complete_episode=True,physical_success_in_NEW_scene=False)
 if f['geometry'][11]<=.5 or f['geometry'][12]>.21:raise ValueError('STALE_GEOMETRY')
 depth=f['depth'];level=level_rotation(f['gravity'].astype(float).copy())
 write('association.json',association)
 camera=CameraClearance(model,shapes,depth,f['K'],T,q,level,n,off,body)
 camera.support_bodies=BODIES
 # Anchor the same currently observed native top using its measured cloud centroid.
 anchor=patch.mean(0);anchor-=n*(anchor@n+off)
 _,_,_,_,anchor_facet=select(camera,depth,anchor)
 observed_patch=ObservedPatch(camera,depth,anchor_facet['camera_pixel'])
 patch=np.unique(np.concatenate([observed_patch.facet(k)[3] for k in sorted(observed_patch.component)]),axis=0)
 write('observed_target_association.json',dict(anchor=anchor_facet,component=observed_patch.report(),
    measured_patch_vertices=len(patch),
    source='Original target-mask connected observed triangles; original22mm final inset retained',physical_success=False))
 provisional_plan=FullRest(model,inertia,shapes,q,gravity,previous,n,off,patch,body,marker,goal,task)
 # Retain every original camera sample/margin, then append the exact6mm
 # expanded motor-housing vertices with zero EXTRA margin (buffer already in geometry).
 from itertools import product
 camera.sample_clearance={b:np.full(len(v),.006) for b,v in camera.samples.items()}
 for bb,V in provisional_plan.housing_mesh.items():
  camera.samples[bb]=np.concatenate([camera.samples[bb],V])
  camera.sample_clearance[bb]=np.r_[camera.sample_clearance[bb],np.zeros(len(V))]
  bounds=camera.samples[bb];camera.boxes[bb]=np.array(list(product(*zip(bounds.min(0),bounds.max(0)))))
 camera.count=sum(len(v) for v in camera.samples.values())
 provisional_plan.reference_q=reference_q;provisional_plan.reference_provenance=reference_provenance
 provisional_plan.bind_camera(camera);provisional_plan.deadline=START+45.
 provisional_plan.record=lambda name,value:write('standing_'+name+'.json',value)
 _,standing_root,standing_fk=provisional_plan.pose(provisional_plan.x0)
 provisional_plan.marker=provisional_plan.exposed_local(standing_root,standing_fk)
 provisional_plan.contact_frozen=True;provisional_plan.states.clear();provisional_plan.derivative_cache.clear()
 provisional_plan.audit_derivatives(provisional_plan.x0)
 write('standing_virtual_work_audit.json',provisional_plan.independent_work_audit(provisional_plan.x0,.2))
 provisional_plan.contact_frozen=False;provisional_plan.states.clear();provisional_plan.derivative_cache.clear()
 seed_x,seed_source=original_seed(provisional_plan)
 _,seed_root,seed_fk=provisional_plan.pose(seed_x)
 seed_points=provisional_plan.support_points(seed_root,seed_fk)
 facets=[observed_patch.query(pt) for pt in seed_points]
 write('contact_facets.json',dict(original_seed=seed_source,facets=[z[3] for z in facets]))
 plan=FullRest(model,inertia,shapes,q,gravity,previous,n,off,patch,body,marker,goal,task)
 plan.reference_q=reference_q;plan.reference_provenance=reference_provenance
 plan.support_normals=np.array([z[0] for z in facets]);plan.support_offsets=np.array([z[1] for z in facets]);plan.contact_edges=np.array([z[2] for z in facets])
 plan.states.clear();plan.derivative_cache.clear();plan.reset_standing_gaps()
 plan.bind_camera(camera);plan.observed_patch=observed_patch
 camera.original_rows=camera.rows
 plan.record=lambda name,value:write(name+'.json',value)
 write('observed_patch.json',observed_patch.report())
 write('input_counts.json',dict(camera_samples=camera.count,observed_triangles=int(camera.valid.sum()),target_triangles=int(camera.target.sum()),ground_triangles=int(camera.ground.sum())))
 def endpoint_audit(node):
  import audit_full_mesh
  audit_full_mesh.REST_BODIES=set(BODIES)
  result=collision_audit(model,shapes,furniture,allowed,[dict(node,phase='landing')],basis,initial)
  write('endpoint_mesh_audit.json',result)
  return result
 plan.endpoint_audit=endpoint_audit
 result=plan.solve();goal=plan.target.copy();marker=plan.marker.copy();write('plan.json',result)
 # Same physical contact classes; adapt audit's eligible set only by existing task.
 import audit_full_mesh
 audit_full_mesh.REST_BODIES=set(BODIES)
 nodes=[dict(z,phase='landing' if z['phase'] in ('normal','load') else 'approach') for z in result['nodes']]
 collision=collision_audit(model,shapes,furniture,allowed,nodes,basis,initial)
 foot_errors=[];normal_rates=[];normal_joint_rates=[];loaded_drift=[]
 for a,b in zip(result['nodes'],result['nodes'][1:]):
  if b['phase']=='normal':
   normal_joint_rates.append(float(np.max(abs(np.asarray(b['q'])[:23]-np.asarray(a['q'])[:23])))/float(b['seconds']-a['seconds']))
  ra=np.asarray(a['root']);rb=np.asarray(b['root']);rotations=Slerp([0,1],Rotation.from_matrix(np.stack([ra[:3,:3],rb[:3,:3]])));last=None
  for u in np.linspace(0,1,9):
   q=(1-u)*np.asarray(a['q'])+u*np.asarray(b['q']);rt=transform(rotations(u).as_matrix(),(1-u)*ra[:3,3]+u*rb[:3,3]);fk=model.fk(q)
   for bb,anchor in zip(plan.foot_bodies,plan.feet):
    foot=rt@fk[bb];foot_errors.append(max(float(np.linalg.norm(foot[:3,3]-anchor[:3,3])),float(np.linalg.norm(Rotation.from_matrix(anchor[:3,:3].T@foot[:3,:3]).as_rotvec()))))
   point=plan.support_points(rt,fk)
   # Independent minimum over every body vertex, not the optimizer witness.
   gap=np.einsum('bi,bi->b',point,plan.support_normals)+plan.support_offsets
   if b['phase']=='normal' and last is not None:normal_rates.append(float(np.max(abs(gap-last)))/(float(b['seconds']-a['seconds'])/8))
   if b['phase']=='load':loaded_drift.append(float(np.max(np.linalg.norm(point-np.asarray(result['nodes'][-1]['support_points']),axis=1))))
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
 tests=dict(float32_action_increment=max32<=.003,float32_desired_margin=margin32>=.03,float32_fingers=bool(finger32),encoded_settled_time=np.ceil(result['settled_seconds']/.02)*.02<14.4,original_action_increment=increment<=.003,original_desired_margin=margin>=.03,fingers_fixed=bool(finger),full_mesh=collision['passed'],hold_time_reserved=result['settled_seconds']<14.4,between_node_feet=max(foot_errors)<.003,normal_speed=max(normal_rates)<=.006,normal_joint_speed=max(normal_joint_rates)<=.05,loaded_material_contact_drift=max(loaded_drift)<.002)
 row=dict(model_pass=all(tests.values()),float32_maximum_increment_rad=max32,float32_minimum_desired_margin_rad=margin32,float32_forward_filter_error_rad=encoding_error,tests=tests,collision=collision,maximum_increment_rad=increment,minimum_desired_margin_rad=margin,maximum_foot_error=max(foot_errors),maximum_normal_speed=max(normal_rates),maximum_normal_joint_speed=max(normal_joint_rates),maximum_loaded_material_point_drift=max(loaded_drift),model_effort_ratio=result['nodes'][-1]['model_effort_ratio'],return_model=result['return_model'])
except Exception as exc:
 row=dict(model_pass=False,error=repr(exc),traceback=traceback.format_exc())
 if 'plan' in locals():
  write('partial_not_admitted.json',dict(getattr(plan,'provisional',{}),optimizations=getattr(plan,'optimizations',[]),state_evaluations=plan.calls))
row.update(total_seconds=time.monotonic()-START,physics_launch_ready=False,physical_success=False,physics_steps=0,PPO_updates=0)
write('receipt.json',row);print((out/'receipt.json').read_text(),flush=True)
