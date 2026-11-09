"""Freeze one reviewed contact-path construction without evaluating robot data."""
import hashlib,importlib.metadata,json,sys
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda:f.read(1024*1024),b''):h.update(data)
    return h.hexdigest()


def main(project,review_log,destination,parallel_review=None):
    project=Path(project).resolve();exp=project/'experiments';source=Path(__file__).resolve().parent
    ref=project/'outputs/armchair_rest_ppo_20261105_v1/learned_sitting72'
    capture=project/'outputs/rest_cycle_narrow_support_20261109_v1/camera_and_wrench_capture'
    mesh=exp/'rest_cycle_narrow_support_20261109_v1/assets/scenes/narrow_support_050'
    inputs=dict(reference_trajectory=ref/'trajectory.npz',reference_results=ref/'results.json',
        reference_audit=ref/'independent_audit.json',
        reference_geometry=project/'outputs/full_body_contact_imitation_v4/dataset/sitting/geometry.json',
        frame=capture/'frame_0.npz',binding=capture/'current_snapshot_binding.json',capture_audit=capture/'capture_audit.json',
        region_audit=exp/'rest_cycle_patch_topology_20261109_v1/validation/result.json',
        patch=exp/'rest_cycle_patch_topology_20261109_v1/validation/patch_0.json',
        model=project/'runs/g1_kinematics_geometry_04/model.json',
        inertia=Path('/data/humanoid-sitting-perception/crouch_rest_bridge_01/balanced_standing10_01/static_inertial_model.json'),
        colliders=exp/'furniture_perception_repair_20261029_v1/render_colliders.json',
        furniture_mesh=mesh/'mesh.npz',target_region=mesh/'target_region.json',review_log=Path(review_log).resolve())
    review=inputs['review_log'].read_text()
    if 'Ran 12 tests' not in review or not review.rstrip().endswith('OK'):raise ValueError('INDEPENDENT_COMPONENT_REVIEW_REQUIRED')
    sources=[source/n for n in ['freeze.py','run_frozen.py','builder.py','continuation.py','control_path.py','interior_audit.py','test_continuation.py']]
    sources += [exp/p for p in ['upper_thigh_support_20261108_v4/static_model.py',
        'rest_cycle_rolling_perch_20261109_v2/rolling.py',
        'rest_cycle_measured_support_20261109_v2/plan_contract.py','rest_cycle_measured_support_20261109_v2/interpolation_audit.py',
        ]]
    for name,folder in [('corner_audit.py','rest_cycle_measured_support_20261109_v2'),('audit_mesh_clearance.py','furniture_repair_20261106_v1')]:
        sources.append(source/name if (source/name).exists() else exp/folder/name)
    if parallel_review is not None:
        inputs['parallel_review']=Path(parallel_review).resolve();log=inputs['parallel_review'].read_text()
        if 'Ran 7 tests' not in log or not log.rstrip().endswith('OK'):raise ValueError('SERIAL_EQUIVALENCE_REVIEW_REQUIRED')
        sources.extend(source/n for n in ['parallel_mesh.py','test_parallel_mesh.py'])
    sources += [project/'sitting_task/src/humanoid_sitting/shared_frame.py',Path(sys.executable).resolve()]
    freeze=dict(decision='ONE_CPU_CONSTRUCTION_NO_AUTOMATIC_DOWNSTREAM',project=str(project),
        inputs={k:str(p) for k,p in inputs.items()},sha256={str(p):sha(p) for p in list(inputs.values())+sources},
        runtime=dict(python=sys.version,numpy=importlib.metadata.version('numpy'),scipy=importlib.metadata.version('scipy'),shapely=importlib.metadata.version('shapely')),
        authorization='Human go after the CPU continuation result; both GPUs remain reserved. No recurring task or new 8-hour allocation is inferred.',
        initialization='Captured physical q and command from bound frame_0, with its own two foot anchors. No old plan/endpoint/failed iterate.',
        witness='Accepted known_old/16: first25consecutive >=0.05BW samples selects contact pitch; final held orientation selects rest roll/pitch/yaw. Actual source clean/hold audit required.',
        equations=['Fixed-foot FK plus COM-x equality solved by at most8 deterministic Newton corrections per sample, analytical geometric Jacobian, original independent FK validation.',
          'Approach root rotation by Slerp, minimum pelvis plane gap to12mm, root x closes prescribed COM balance; root z follows the measured plane equation.',
          '100 normal steps at0.006m/s reach contact with unchanged orientation; feet remain closed.',
          '160 exact material rolling steps transfer load; load is solved algebraically from full force/moment equilibrium. Support must reduce torque-squared effort.',
          'Each command is q plus same-state torque/stiffness and original fading standing bias; original0.1filter,0.003increment,0.03margin. No interpolation of commands between solved poses.'],
        test='All640controls and reverse commands must pass original gates, four geometry samples per control, full original model/mesh audit. Any failure rejects; no count/method/parameter retry.',
        independent_review='12 synthetic tests passed: analytic derivatives vs central differences, fixed-foot FK, COM closure, rolling material-point invariance, interior counterexample, stage bindings, both-direction original actuator and time limits. Full physical admission remains separate.',
        resources=dict(phase_controls=[380,100,160],total_controls_per_direction=640,maximum_closure_steps_per_control=8,
          maximum_linear_system_solves=5120,largest_linear_system=[13,13],terminal_optimizer_calls=0,
          total_wall_cap_s=45,work_stop_s=42,CPU_cores=1,AS_GiB=8,output_MiB=32,GPU_contexts=0,
          new_model_nodes_max=641,quarter_control_audit_samples=2560,full_original_mesh_samples_max=5121,
          simulation_steps=0,PPO_updates=0),
        finite_return='Preserve accepted intermediate model nodes only as rejected evidence if construction fails. A successful model result cannot auto-launch motion: current camera, loaded observer, native quiet, three physical cycles and all72 remain required.',
        unchanged='Action boxes, physical criteria, timing, clearance, raw velocity, load/release and protected72-case gates. Historical one-time waivers and closed diagnostic results are not reused.',
        retry_authorized=False,physical_success=False)
    if parallel_review is not None:
        freeze['resources'].update(mesh_backend='exact_partition',CPU_cores=9,parent_AS_GiB=8,
            mesh_workers=8,worker_AS_GiB=2,maximum_aggregate_AS_GiB=24)
        freeze['independent_review']+=' Seven additional partition tests passed, including exact complete serial-report equality, all8workers, failed and stalled worker cleanup.'
        freeze['substantive_resource_change']='The unfinished5121-pose mesh check is partitioned by original segment indices; all original clipping operations/sample order/first50/counts are preserved. This changes the execution algorithm, not physical tolerances, references, phase counts or the45second total cap.'
        freeze['prior_attempt']='Serial construction reached all640controls and passed2560interior checks, then timed out in triangle clipping. That execution stays closed. This build starts again from bound physical inputs; it does not load its plan or partial outputs.'
    with Path(destination).open('x') as f:json.dump(freeze,f,indent=2);f.write('\n')
    print('Frozen',len(freeze['sha256']),'source/input hashes; no model arithmetic or simulation executed.')


if __name__=='__main__':main(*sys.argv[1:])
