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
 if target=='lean_center':
  p=R/'outputs/armchair_rest_ppo_20261105_v1/demo_rear_back_center';f=dict(np.load(p/'frame_0.npz'))
  with np.load(p/'trajectory.npz') as data:previous=data['previous'][0,0].astype(float)
  old=base.LeanGeometry(model,f['q'].astype(float),f['gravity'].astype(float),f['geometry'].astype(float),f['selected_points'],previous)
  old.clip_world[:-2,3]+=.002
  rt,fk=old.pose(old.start);_,point,_,_=old.geometry(old.start)
  body='pelvis';vertices=np.concatenate([np.asarray(z['vertices']) for z in shapes if z['body']==body])
  world=apply(rt@fk[body],vertices);height=world@old.n+old.off;lowest=height.min()
  # Current standing support face, not any prior failed endpoint.
  marker=vertices[height<=lowest+1e-10].mean(0);point=apply(rt@fk[body],marker)
  facet=np.c_[world@old.up,world@old.lateral];eq=ConvexHull(np.c_[old.cloud@old.up,old.cloud@old.lateral]).equations
  uv=np.array([point@old.up,point@old.lateral]);inset=float(np.min(-(eq[:,:2]@uv+eq[:,2])))
  if inset<.022 or not old.band[0]+.04<=point@old.up<=old.band[1]-.04:raise ValueError(('PELVIS_OUTSIDE_CURRENT_OBSERVED_BACK',inset,float(point@old.up),old.band.tolist()))
  n=old.n;off=old.off;patch=old.cloud;goal=point-(point@n+off)*n
  q=f['q'];gravity=f['gravity'];initial=f['root_position'];basis=f['root_rotation']@old.level.T;task='leaning'
  case=read(p/'manifest.json')['case'];furniture=dict(np.load(case['mesh_npz']));allowed=read(Path(case['mesh_npz']).with_name('target_region.json'))['triangle_indices']
  association=dict(body=body,marker_local=marker.tolist(),source='Current standing pelvis supporting convex face; current camera finite back inset and height',current_inset_m=inset,physical_contact_rule='Original run_supported_postures permits pelvis on the target patch')
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
 if f['geometry'][11]<=.5 or f['geometry'][12]>.21:raise ValueError('STALE_GEOMETRY')
 depth=f['depth'];level=level_rotation(f['gravity'].astype(float).copy())
 if target=='stool_center':
  # Transform only triangles actually observed in saved depth. Furniture truth
  # remains exclusive to the independent evaluator, never optimizer input.
  observed,_,obs_report=observed_mesh(model,depth,f['K'],T,f['q'],f['gravity'],f['geometry'])
  world_observed=observed['vertices']@basis.T+f['root_position']
  world_observed=(world_observed-center)@rot.T+center
  augmented=(world_observed-initial)@basis
  depth=rasterize(augmented,observed['faces'],f['K'],T,level,*f['depth'].shape)
  association.update(camera_augmentation='Only saved observed triangle vertices rigidly rotated and translated; unseen space remains unknown; fresh real reset required',observed_triangles=obs_report['triangles'])
 write('association.json',association)
 camera=CameraClearance(model,shapes,depth,f['K'],T,q,level,n,off,body)
 provisional_plan=FullRest(model,inertia,shapes,q,gravity,previous,n,off,patch,body,marker,goal,task)
 seed_x,seed_source=original_seed(provisional_plan)
 _,seed_root,seed_fk=provisional_plan.pose(seed_x)
 seed_contact=provisional_plan.material_point(seed_root,seed_fk,0.)
 seed_contact-=n*(seed_contact@n+off)
 standing_goal=goal.copy()
 n,off,goal,edges,facet_report=select(camera,depth,seed_contact)
 facet_report.update(standing_goal=standing_goal.tolist(),contact_selection='Exposed material point of ORIGINAL current-input analytic seed projected to original observed fitted plane; no failed-state file read',seed_source=seed_source,standing_to_seed_contact_m=float(np.linalg.norm(seed_contact-standing_goal)))
 write('contact_facet.json',facet_report)
 plan=FullRest(model,inertia,shapes,q,gravity,previous,n,off,patch,body,marker,goal,task)
 plan.contact_edges=edges
 plan.bind_camera(camera)
 camera.original_rows=camera.rows
 plan.record=lambda name,value:write(name+'.json',value)
 write('input_counts.json',dict(camera_samples=camera.count,observed_triangles=int(camera.valid.sum()),target_triangles=int(camera.target.sum()),ground_triangles=int(camera.ground.sum())))
 def endpoint_audit(node):
  import audit_full_mesh
  audit_full_mesh.REST_BODIES={'pelvis'}
  result=collision_audit(model,shapes,furniture,allowed,[dict(node,phase='landing')],basis,initial)
  write('endpoint_mesh_audit.json',result)
  return result
 plan.endpoint_audit=endpoint_audit
 # Named original-seed diagnostic; no optimizer or returned/rejected point.
 plan.loads=np.linspace(0.,.2,5)
 x,seed_report=original_seed(plan)
 _,rt,fk=plan.pose(x);plan.marker=plan.exposed_local(rt,fk).copy()
 plan.contact_frozen=True;plan.states.clear()
 x[29]=float(np.max(abs(plan.command(x,0.)-plan.previous[:23])))/(.15*14.4)
 g=plan.endpoint_rows(x);e=plan.equal(x,0.)
 labels=['tilt_reserve']+['patch_inset_'+str(i) for i in range(len(plan.patch))]
 labels+=['pelvis_plane_vertex_'+str(i) for i in range(len(plan.mesh[body]))]
 labels+=['camera_'+b for b in camera.samples]+['contact_facet_edge_'+str(i) for i in range(3)]
 for load in plan.loads:
  prefix='load_'+str(float(load))+':'
  for b,poly in zip(plan.foot_bodies,plan.foot_polygons):
   labels += [prefix+b+':cop_edge_'+str(i) for i in range(len(poly))]+[prefix+b+':normal_load',prefix+b+':friction']
  for group in ['command_lower','command_upper','readiness_positive','readiness_negative']:
   labels += [prefix+group+':'+n for n in plan.model.names[:23]]
 labels+=['total_time_reserve']
 assert len(labels)==len(g),(len(labels),len(g))
 negative=[dict(row=i,name=labels[i],residual=float(value)) for i,value in enumerate(g) if value < -1e-6]
 Je=plan.equal_jac(x,0.);Jg=plan.endpoint_jac(x)
 # Independent finite differences of the previously unaudited mechanical ROWS,
 # not a repeat of the inherited primitive derivative proof. Camera active
 # minima are deliberately excluded from differentiability claims.
 rows0=len(plan.geometry_rows(x,0.));h=1e-6;max_error=0.;worst=None
 for k in range(29):
  a=x.copy();b=x.copy();a[k]-=h;b[k]+=h
  ga=np.concatenate([plan.mechanical_rows(a,float(l)) for l in plan.loads])
  gb=np.concatenate([plan.mechanical_rows(b,float(l)) for l in plan.loads])
  err=abs((gb-ga)/(2*h)-Jg[rows0:-1,k]);index=int(np.argmax(err))
  if float(err[index])>max_error:max_error=float(err[index]);worst=dict(column=k,row=labels[rows0+index])
 singular=np.linalg.svd(Je,compute_uv=False)
 row=dict(diagnostic_complete=True,original_seed_only=True,optimizers_called=0,
  full_inequality_rows=len(g),negative_original_seed_rows=negative,
  original_seed_maximum_equality_residual=float(np.max(abs(e))),
  equality_jacobian_singular_values=singular.tolist(),
  mechanical_row_derivative_max_error=max_error,mechanical_row_derivative_worst=worst,
  interpretation='Named current-input seed associations and row-derivative audit only; no returned solver point reconstructed, no feasible motion or repair claimed',
  model_pass=False,physics_launch_ready=False,physical_success=False,physics_steps=0,PPO_updates=0)
except Exception as exc:
 row=dict(diagnostic_complete=False,error=repr(exc),traceback=traceback.format_exc(),model_pass=False,physical_success=False,physics_steps=0,PPO_updates=0)
row.update(total_seconds=time.monotonic()-START)
write('receipt.json',row);print((out/'receipt.json').read_text(),flush=True)
