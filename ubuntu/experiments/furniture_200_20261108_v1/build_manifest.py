"""Deterministic 200-scenario request manifest; never launches a simulator."""
import collections
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent
HERE=Path(__file__).resolve().parent
old=json.loads((ROOT/'furniture_examples_20261105_v1/protocol.json').read_text())
cases={c['id']:c for c in json.loads((ROOT/'furniture_examples_20261105_v1/cases.json').read_text())}
assets=['myd1WSucAz','6MoOyPtetL','1kwsjhpY84','wooden_stool_01','modern_coffee_table_01']
profiles=[('center',0.,0.),('near',0.,.015),('left',-.04,-.01),('right',.04,-.01)]
yaws=[0.,-3.,3.,-6.,6.]
variants=[]
for asset in assets:
    seat=next(v for v in old['variants'] if v['asset']==asset and v['task']=='sitting' and v['id'].endswith('_center'))
    for task in ('sitting','leaning'):
        rear=asset+'_rear_back'
        base=cases[seat['case'] if task=='sitting' or rear not in cases else rear]
        center=base['target_center'];pivot=[center[0],center[1],0.]
        for yaw in yaws:
            a=math.radians(yaw);co,si=math.cos(a),math.sin(a)
            for profile,lateral,toward_robot in profiles:
                # Rigid furniture transform around a floor-height pivot; the
                # same matrix must transform visual mesh, collider and labels.
                tx=pivot[0]-co*pivot[0]+si*pivot[1]+toward_robot
                ty=pivot[1]-si*pivot[0]-co*pivot[1]+lateral
                matrix=[[co,-si,0.,tx],[si,co,0.,ty],[0.,0.,1.,0.],[0.,0.,0.,1.]]
                ident=f'{task}_{asset}_{len([v for v in variants if v["asset"]==asset and v["task"]==task])+1:02d}'
                variants.append(dict(id=ident,asset=asset,title=base['title'],task=task,
                    base_scene=base['id'],base_mesh_npz=base['mesh_npz'],base_mesh_usd=base['mesh_usd'],
                    furniture_transform=matrix,furniture_yaw_delta_deg=yaw,
                    profile=profile,lateral_shift_m=lateral,toward_robot_shift_m=toward_robot,
                    original_robot_lateral_offset_m=seat['y'] if task=='sitting' else 0.,
                    robot_reset='Fresh matching standing reset required; no rejected path or reset state reused',
                    support_intent='pelvis_or_upper_thigh_on_seat_top' if task=='sitting' else 'posterior_pelvis_or_upper_thigh_supported_with_hip_hinge',
                    surface_intent='seat/top' if task=='sitting' else 'rear_back' if rear in cases else 'upper_rim_pending_hip_height_feasibility',
                    source_furniture_height_m=base['bounds'][1][2]-base['bounds'][0][2],
                    surface_selection_status='requires_current_depth_patch_and_collision_audit',
                    construction_status='not_admitted',rollout_status='not_run',admitted_demonstration=False))
counts=collections.Counter((v['asset'],v['task']) for v in variants)
assert len(variants)==200 and len({v['id'] for v in variants})==200
assert all(counts[a,t]==20 for a in assets for t in ('sitting','leaning'))
for a in assets:
    for t in ('sitting','leaning'):
        rows=[v for v in variants if v['asset']==a and v['task']==t]
        assert len({json.dumps(v['furniture_transform']) for v in rows})==20
        assert all(v['furniture_transform'][2]==[0.,0.,1.,0.] for v in rows)
protocol=dict(version=1,state='PREPARED_200_SCENARIOS_NO_MOTION_ADMITTED',
    authorization='2026-11-08 human:20sitting and20leaning per furniture,5objects; vary position/angle; lean using posterior hips/upper thighs against suitable rear surfaces.',
    target=dict(sitting_demonstrations=100,leaning_demonstrations=100,objects=5,per_object_per_task=20),
    variants=variants,criteria=old['criteria'],
    intended_support='Preserve original physical pass/fail score separately. A new leaning example counts toward this request only with measured intended posterior pelvis/upper-thigh support; shoulder-only contact or mere proximity does not fill a hip-lean slot.',
    posture_preference='Hip hinge and backward pelvis displacement, minimal necessary knee bend, lower sustained joint-effort proxy, stable feet, ready-to-stand. Report knee flexion and root drop; do not invent an energy measurement or silently relabel a squat.',
    variation_policy='Small upright yaw and horizontal translations only. Exact same rigid transform for visible mesh/collider/target labels. No height rescaling, artificial supports, friction or action changes.',
    counting='Target200 valid full rollouts, not200 files. Preserve rejected starts and failed episodes separately. Historical demonstrations remain preserved but do not become new varied examples by duplication.',
    admission_order=['repair and audit base contact/path','fresh current-depth/reset/command/quiet checks for each transformed scenario','complete20s physical rollout','unchanged independent physical and contact scoring plus intended-support audit','corpus verification then imitation/PPO4090','all72 repairs/regressions and video'],
    resources=dict(gpu='4090 only',gpu_4070='reserved for AIWorker',scientific_owners=1,
        first_pass_max_rollouts=200,retries_per_scenario=0,active_controls=200000,warmup_controls=5000,
        physics_substeps_per_control=4,max_control_substeps_including_warmup=820000,
        construction_and_audits_wall_seconds_per_start=45,max_wall_seconds_per_trial=600,
        total_trial_wall_seconds_cap=120000,min_free_disk_bytes=20000000000,
        proposed_max_evidence_bytes_per_trial=96*1024**2,max_collection_evidence_bytes=200*96*1024**2,
        collector_storage_profile_validated=False,recurring_automation=False,queued_gpu_jobs=0),
    blockers=['No current base endpoint/path admitted; latest analytic endpoint rejected.','Stool and coffee table lack a prepared rear-back patch; hip-level rim support requires separate validation.','Live ankle/current-camera checks and full transformed-scene reset checks pending.'],
    training_launch_ready=False,physics_launch_ready=False,physical_success=False)
for name,data in [('protocol.json',protocol),('manifest_check.json',dict(passed=True,scope='Scenario counts, uniqueness and rigid-transform specification only',objects=5,sitting=100,leaning=100,unique_scenarios=200,new_physics_steps=0,new_demonstrations=0))]:
    with (HERE/name).open('x') as f:json.dump(data,f,indent=2)
source={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'furniture_examples_20261105_v1/protocol.json',ROOT/'furniture_examples_20261105_v1/cases.json',HERE/'protocol.json']}
with (HERE/'manifest_sources.json').open('x') as f:json.dump(source,f,indent=2)
print(json.dumps(dict(planned_sitting=100,planned_leaning=100,objects=5,unique_scenarios=200,physical_rollouts=0)))
