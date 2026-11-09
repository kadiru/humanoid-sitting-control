"""Offline recorded-pose render diagnostic. No physics steps, policies or learning.

Truth poses are used only to construct controlled diagnostic scenes. All depth
and plane estimates are produced by the unchanged installed sensor pipeline.
"""
import sys,json,hashlib,datetime,shutil
from pathlib import Path
import numpy as np
import torch
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')
OUT=D/'render_isolation_v2_22'
original=json.loads((D/'joint_path_17/candidate.command.json').read_text())
argv=original[3:];k=argv.index('--landing-config');del argv[k:k+2]
argv[argv.index('--output')+1]=str(OUT)
sys.argv=[__file__]+argv
import run_low_positive_feedback_v2 as harness
from humanoid_sitting.supported_surface import observed_support
from run_supported_postures import REAR_CAMERA
from humanoid_sitting.shared_frame import BodyModel

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def main():
 OUT.mkdir(exist_ok=False);(OUT/'source').mkdir();shutil.copy2(__file__,OUT/'source'/Path(__file__).name)
 a=harness.parent_harness.arguments();cfg,_,_=harness.parent_harness.build(a,OUT)
 bt=D/'normal_landing3s_06/candidate/trace.npz';ct=D/'joint_path_17/candidate/trace.npz'
 b=dict(np.load(bt));c=dict(np.load(ct));selected=[0,1,6,7,10,11];unselected=[i for i in range(24) if i not in selected]
 inputs={str(p):sha(p) for p in [bt,ct,D/'banks/probe/geometry.json',Path(__file__)]}
 write(OUT/'protocol.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='Offline controlled render of recorded development poses, no physics integration/policies/optimizer/holdout. A baseline all24 poses versus B same baseline unselected poses plus six candidate selected poses. A repeated to test determinism and reversibility.',times_s=[8.,8.2,8.4],sequence=['A0','A1','B0','B1','A2'],selected=selected,unselected=unselected,hashes=inputs))
 records=[];arrays={}
 with harness.base.launch_simulation(cfg,a):
  raw=harness.base.gym.make('Humanoid-Sit-Learned-G1',cfg=cfg).unwrapped
  raw.reset();clock0=raw.sim.get_physics_step_count();robot=raw.scene['robot'];cam=raw.scene['rear_camera'];pelvis=robot.body_names.index('pelvis');model=BodyModel(harness.base.MODEL)
  write(OUT/'runtime.json',dict(physics_count_after_scene_initialization=clock0,renderer_types=raw.sim.render_context.renderer_types,env_origins=raw.scene.env_origins.cpu().tolist(),camera_type=str(type(cam)),physics_backend=str(type(raw.sim.physics_manager))))
  for t in [8.,8.2,8.4]:
   j=int(np.argmin(abs(b['t']-t)));assert abs(b['t'][j]-t)<1e-6 and b['q'].shape==c['q'].shape
   reference_state={};group={}
   for label in ['A0','A1','B0','B1','A2']:
    root=b['body'][j,:,pelvis].copy();q=b['q'][j].copy()
    if label.startswith('B'):root[selected]=c['body'][j,selected,pelvis];q[selected]=c['q'][j,selected]
    ids=torch.arange(24,device=raw.device);rt=torch.tensor(root,device=raw.device,dtype=torch.float32);qt=torch.tensor(q,device=raw.device,dtype=torch.float32)
    robot.write_root_pose_to_sim_index(root_pose=rt,env_ids=ids)
    robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros(24,6,device=raw.device),env_ids=ids)
    robot.write_joint_state_to_sim_index(position=qt,velocity=torch.zeros_like(qt),env_ids=ids)
    raw.scene.write_data_to_sim();raw.sim.forward();raw.scene.update(0.);raw.sim.render()
    raw.sim.render_context.update_scene_state(clock0);harness.base.acquire_current_camera(cam)
    assert raw.sim.get_physics_step_count()==clock0
    depth=cam.data.output['distance_to_image_plane'].torch.cpu().numpy().copy();K=cam.data.intrinsic_matrices.torch.cpu().numpy().copy()
    actual_q=robot.data.joint_pos.torch.cpu().numpy().copy();gravity=robot.data.projected_gravity_b.torch.cpu().numpy().copy();actual_body=robot.data.body_pose_w.torch.cpu().numpy().copy()
    estimates=[];diagnostics=[]
    for i in range(24):
     v,diag=observed_support(model,K[i],REAR_CAMERA,depth[i],actual_q[i],gravity[i],prior=b['geometry'][max(j-2,0),i]);estimates.append(np.full(13,np.nan) if v is None else v);diagnostics.append(diag)
    estimates=np.array(estimates)
    from scipy.spatial.transform import Rotation
    expected_camera=root[:,:3]+Rotation.from_quat(root[:,3:]).apply(np.broadcast_to(REAR_CAMERA[:3,3],(24,3)))
    camera_pos=cam.data.pos_w.torch.cpu().numpy().copy()
    assert abs(camera_pos-expected_camera).max()<2e-5,('camera_pose_stale',float(abs(camera_pos-expected_camera).max()))
    assert np.isfinite(estimates).all(1).sum()>=20,('insufficient_measured_support',diagnostics)
    group[label]=dict(depth=depth,q=actual_q,gravity=gravity,body=actual_body,K=K,geometry=estimates,camera_position=camera_pos)
    for key,v in group[label].items():arrays[f't{int(round(t*10))}_{label}_{key}']=v
    if label=='A0':reference_state=group[label]
    metrics={}
    for key,v in group[label].items():
     aa=reference_state[key][unselected];bb=v[unselected];finite=np.isfinite(aa)&np.isfinite(bb)
     metrics[key]=dict(exact=bool(np.array_equal(aa,bb,equal_nan=True)),changed_values=int(np.count_nonzero(~(aa==bb)&~(np.isnan(aa)&np.isnan(bb)))),max_finite_difference=float(abs(aa[finite]-bb[finite]).max()) if finite.any() else None)
    records.append(dict(time=t,label=label,unselected_vs_A0=metrics,valid_geometry_cases=int(np.isfinite(estimates).all(1).sum()),diagnostics=diagnostics))
    print('RENDER',t,label,metrics['depth'],metrics['geometry'],flush=True)
   assert all(group[x]['q'][unselected].tobytes()==group['A0']['q'][unselected].tobytes() for x in group)
  np.savez_compressed(OUT/'render_arrays.npz',**arrays)
  write(OUT/'results.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),completed=True,physics_steps_advanced=raw.sim.get_physics_step_count()-clock0,optimizer_updates=0,fresh_holdout=False,records=records,inputs_unchanged=all(sha(p)==h for p,h in inputs.items()),conclusion='Controlled rendering evidence only; inspect repeatability, scene-switch effects, geometry and exact unchanged own-body/encoder/IMU inputs before inferring a cause.'))
  raw.close()
if __name__=='__main__':main()
