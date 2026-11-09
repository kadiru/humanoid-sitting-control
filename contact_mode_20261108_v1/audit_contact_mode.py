"""Independent exposed-contact and fixed-material-work audit on named inputs."""
import os,time,json,sys,hashlib
from pathlib import Path
START=time.monotonic();os.environ['CUDA_VISIBLE_DEVICES']=''
H=Path(__file__).resolve().parent;R=Path('/home/kadir/humanoid-sitting');E=R/'experiments'
sys.path[:0]=[str(H),str(R/'sitting_task/src')]
for p,h in json.loads((H/'freeze.json').read_text())['files'].items():
    assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
import numpy as np
from scipy.spatial.transform import Rotation
from humanoid_sitting.shared_frame import BodyModel,apply,level_rotation
from contact_mode import FullRest
read=lambda p:json.loads(Path(p).read_text())
model=BodyModel(R/'runs/g1_kinematics_geometry_04/model.json')
inertia=read('/data/humanoid-sitting-perception/crouch_rest_bridge_01/balanced_standing10_01/static_inertial_model.json')
shapes=read(E/'furniture_perception_repair_20261029_v1/render_colliders.json')['shapes']
p=R/'outputs/armchair_rest_ppo_20261105_v1/demo_rear_back_center';f=dict(np.load(p/'frame_0.npz'))
with np.load(p/'trajectory.npz') as d:previous=d['previous'][0,0].astype(float)
R0=level_rotation(f['gravity'].astype(float).copy());n=R0@f['geometry'][2:5];off=float(f['geometry'][5]);patch=f['selected_points']@R0.T
vertices=np.concatenate([s['vertices'] for s in shapes if s['body']=='pelvis'])
heights=(vertices@R0.T)@n;old_marker=vertices[heights<=heights.min()+1e-10].mean(0)
old_point=R0@old_marker;goal=old_point-(old_point@n+off)*n
plan=FullRest(model,inertia,shapes,f['q'],f['gravity'],previous,n,off,patch,'pelvis',old_marker,goal,'leaning')
ref=R/'outputs/full_body_contact_imitation_v4/capture_known_old';assert read(ref/'demonstration_audit.json')['admitted'][8]
with np.load(ref/'demonstrations_50hz.npz') as d:
    refq=d['q_post'][999,8].copy();refR=Rotation.from_quat(d['root'][999,8,3:7]).as_matrix()
x=plan.x0.copy();x[:23]=refq[:23];x[26:29]=Rotation.from_matrix(R0.T@refR).as_rotvec()
s=plan.state(x,0.);world=apply(s['root']@s['fk']['pelvis'],vertices)
new_projection=float(s['point']@n);minimum=float(np.min(world@n))
old_projection=float(apply(s['root']@s['fk']['pelvis'],old_marker)@n)
torso=np.concatenate([apply(s['root']@s['fk'][b],plan.mesh[b]) for b in ['torso_link','logo_link']])
tests=dict(exposed_point_attains_true_support=abs(new_projection-minimum)<1e-9,
           no_pelvis_vertex_penetrates=np.min(world@n-new_projection)>=-1e-9,
           old_face_centroid_restricts_this_successful_orientation=old_projection-minimum>.001)
unfrozen_load_rejected=False
try:plan.state(x,.2)
except ValueError as e:unfrozen_load_rejected=str(e)=='MATERIAL_CONTACT_MUST_BE_FROZEN_BEFORE_LOAD'
tests['cannot_load_before_material_freeze']=unfrozen_load_rejected
plan.freeze_contact(s);fixed=plan.marker.copy();shift=x.copy();shift[27]+=.02
shifted=plan.state(shift,.2)
tests['loaded_material_stays_fixed']=bool(np.max(abs(plan.contact_local(shifted)-fixed))<1e-10)
work=plan.independent_work_audit(x,.2);tests['independent_loaded_work']=work['passed']
result=dict(passed=all(tests.values()),tests=tests,source='Admitted known_old/8/frame999 orientation and q; used only as geometry/work witness, never candidate initialization',
    old_marker_local=old_marker.tolist(),exposed_marker_local=fixed.tolist(),old_centroid_implied_penetration_m=old_projection-minimum,
    minimum_torso_clearance_for_exposed_pelvis_plane_m=float(np.min(torso@n)-new_projection),
    reference_root_pitch_deg=float(Rotation.from_matrix(refR).as_euler('xyz',degrees=True)[1]),
    work_audit=work,seconds=time.monotonic()-START,physics_steps=0,GPU_used=False,
    limitations='This proves contact-feature selection and material-work consistency, not a feasible whole trajectory, upright posture, achieved force or physical repair.')
with (H/'contact_mode_audit.json').open('x') as out:json.dump(result,out,indent=2,allow_nan=False)
print(json.dumps(result),flush=True)
