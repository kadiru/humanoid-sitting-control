"""One frozen CPU construction with an outer45s process watchdog."""
import os,sys,time,json,hashlib
START=time.monotonic()
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
ROOT=Path('/home/kadir/humanoid-sitting');HERE=Path(__file__).resolve().parent;EXP=ROOT/'experiments'
target=sys.argv[1]
paths=['furniture_collision_repair_20261106_v8','furniture_collision_repair_20261106_v7','furniture_collision_repair_20261106_v6','furniture_perception_repair_20261029_v1','furniture_collision_repair_20261106_v5','furniture_collision_repair_20261106_v4','furniture_load_repair_20261106_v3','furniture_repair_20261106_v1/model_v4','furniture_repair_20261106_v1']
sys.path[:0]=[str(HERE)]+[str(EXP/p) for p in paths]+[str(ROOT/'scripts'),str(ROOT/'sitting_task/src')]
read=lambda p:json.loads(Path(p).read_text())
out=HERE/'results_v3'/target;out.mkdir(parents=True,exist_ok=False)
def write(name,d):
    with (out/name).open('x') as f:json.dump(d,f,indent=2,allow_nan=False,default=lambda v:v.tolist() if hasattr(v,'tolist') else v)
try:
    freeze=read(HERE/'freeze_v3.json')
    for p,h in freeze['files'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
    import numpy as np
    from scipy.spatial.transform import Rotation
    from humanoid_sitting.shared_frame import BodyModel,level_rotation
    from branch_plan import SeatPlan
    from visible_surface import expand
    from audit_support_geometry import audit as audit_geometry
    from complete_support_path import complete
    from audit_support_commands import audit as audit_commands
    from audit_mesh_clearance_fast import audit as audit_mesh
    m=BodyModel(ROOT/'runs/g1_kinematics_geometry_04/model.json')
    inertia=read('/data/humanoid-sitting-perception/crouch_rest_bridge_01/balanced_standing10_01/static_inertial_model.json')
    shapes=read(EXP/'furniture_perception_repair_20261029_v1/render_colliders.json')['shapes']
    refdir=ROOT/'outputs/full_body_contact_imitation_v4/capture_known_old'
    audit=read(refdir/'demonstration_audit.json');assert audit['admitted'][8] and audit['strict_clean'][8]
    with np.load(refdir/'demonstrations_50hz.npz') as ref:
        root0=ref['root'][0,8];root1=ref['root'][999,8];qref=ref['q_post'][999,8]
    reference=dict(root_delta_xz=(root1[[0,2]]-root0[[0,2]]).tolist(),pitch=float(Rotation.from_quat(root1[3:]).as_euler('xyz')[1]),
        arms=[float(qref[[m.names.index(s+'_'+j+'_joint') for s in ['left','right']]].mean()) for j in ['shoulder_pitch','elbow_pitch']])
    if target=='stool_center':
        case=next(c for c in read(EXP/'furniture_examples_20261105_v1/cases.json') if c['id']=='wooden_stool_01_top')
        meshpath=Path(case['mesh_npz']);furniture=dict(np.load(meshpath));rot=Rotation.from_euler('z',45,degrees=True).as_matrix()
        center=(furniture['vertices'].min(0)+furniture['vertices'].max(0))/2;center[2]=0.
        furniture['vertices']=(furniture['vertices']-center)@rot.T+center
        p=ROOT/'outputs/furniture_examples_20261105_v2/sit_wooden_stool_01_center'
        f=dict(np.load(p/'frame_0.npz'));init=dict(np.load(p/'startup.npz'))
        camera=np.eye(4);camera[:3,:3]=[[0,.707106781,-.707106781],[1,0,0],[0,-.707106781,-.707106781]];camera[:3,3]=[-.07,0,.38]
        expanded,_,_=expand(m,f['K'],camera,f['depth'],f['q'],f['gravity'],f['selected_pixels'])
        level=level_rotation(f['gravity'].astype(float));basis=f['root_rotation']@level.T
        placement=read(EXP/'furniture_rest_repair_20261107_v1/seat_models/sit_wooden_stool_01_center.placement.json')
        # Reuse the already audited static placement, not its failed posture.
        assert placement['independent_all_triangle_clip_audit']['passed']
        shift=float(placement['closer_m'])
        assert abs(shift-.2689748497324567)<1e-12
        translation=level.T@np.array([shift,0.,0.])
        observed_world=expanded@f['root_rotation'].T+f['root_position']
        observed_world=(observed_world-center)@rot.T+center
        observed=(observed_world-f['root_position'])@f['root_rotation']+translation
        geometry=f['geometry'].astype(float).copy();geometry[5]-=geometry[2:5]@translation;geometry[6:9]+=translation
        q=f['q'];gravity=f['gravity'];previous=init['command'][0];intent='stool'
        initial=f['root_position']-basis@np.array([shift,0.,0.])
    elif target=='lcouch_center':
        p=ROOT/'outputs/furniture_camera_coverage_20261106_v1/calibration_demos/sit_1kwsjhpY84_center_local_patch_normal_landing'
        z=dict(np.load(p/'construction_inputs.npz'));f=dict(np.load(p/'frame_0.npz'));case=read(p/'manifest.json')['case']
        meshpath=Path(case['mesh_npz']);furniture=dict(np.load(meshpath))
        q=z['q'];gravity=z['gravity'];geometry=z['geometry'];observed=z['selected_points'];previous=z['previous'];intent='seat'
        basis=f['root_rotation']@level_rotation(gravity.astype(float)).T;initial=f['root_position']
    else:raise ValueError('UNFROZEN_TARGET')
    plan=SeatPlan(m,inertia,shapes,q,gravity,geometry,observed,previous,intent,reference=reference)
    initial_root=plan.state(plan.z0)[1]
    checkq=q.copy();checkq[plan.legids]=plan.pitch_ik(initial_root);fk=m.fk(checkq)
    closure=[]
    for side,anchor in zip(('left','right'),plan.feet):
        foot=initial_root@fk[side+'_ankle_roll_link']
        closure.append(dict(side=side,position_m=float(np.linalg.norm(foot[:3,3]-anchor[:3,3])),rotation_rad=float(np.linalg.norm(Rotation.from_matrix(anchor[:3,:3].T@foot[:3,:3]).as_rotvec()))))
    write('initial_ik_audit.json',dict(closure=closure,all_positive_knees=bool(np.all(checkq[[m.names.index(s+'_knee_joint') for s in ['left','right']]]>0))))
    assert all(max(x['position_m'],x['rotation_rad'])<.003 for x in closure),'INITIAL_GEOMETRIC_IK_CLOSURE_FAILED'
    measured_height=float(np.mean(plan.cloud[:,2])-plan.floor)
    reference['root_delta_xz'][1]+=measured_height-.44+.012
    write('reference.json',dict(reference=reference,source='known_old/8/frame999',admitted_source=True,measured_height=measured_height))
    construction=plan.solve();write('plan.json',construction)
    geometry_audit=audit_geometry(plan,construction)
    path=complete(plan,construction);write('complete.json',path)
    commands=audit_commands(plan,construction,path)
    allowed=read(meshpath.with_name('target_region.json'))['triangle_indices']
    nodes=construction['nodes']+[dict(n,phase='landing') for n in path['landing'][1:]]
    collision=audit_mesh(m,shapes,furniture,allowed,nodes,basis,initial)
    admitted=bool(all(v for k,v in geometry_audit['tests'].items() if k!='time_for_original_hold') and commands['passed'] and collision['passed'])
    row=dict(model_admitted=admitted,geometry=geometry_audit,commands=commands,collision=collision)
except Exception as e:
    import traceback
    row=dict(model_admitted=False,error=repr(e),traceback=traceback.format_exc())
row.update(seconds_inside_process=time.monotonic()-START,simulation_steps=0,PPO_updates=0,physical_success=False,physics_launch_ready=False)
write('receipt.json',row)
print(json.dumps({k:v for k,v in row.items() if k not in ['geometry','commands','collision','traceback']}),flush=True)
