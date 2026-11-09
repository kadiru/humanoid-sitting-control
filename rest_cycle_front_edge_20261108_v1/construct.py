"""Cold construction bound to the actual current camera patch.

No translated old cloud, old endpoint, or saved path is a solve input. The
native saved root pose is used only by the independent mesh clearance audit.
"""
import os, sys, time, json, signal, resource, hashlib, traceback
from pathlib import Path
START = time.monotonic()
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
resource.setrlimit(resource.RLIMIT_AS, (40 * 1024**3, 40 * 1024**3))
def timeout(*args): raise TimeoutError('45_SECOND_TOTAL_CONSTRUCTION_CAP')
signal.signal(signal.SIGALRM, timeout); signal.setitimer(signal.ITIMER_REAL, 45.)
ROOT = Path('/home/kadir/humanoid-sitting'); HERE = Path(__file__).resolve().parent
SOURCE = ROOT/'outputs/rest_cycle_edge_visible_20261108_v1/sit_edge_visible/frame_0.npz'
ASSETS = HERE.parent/'rest_cycle_high_support_20261108_v1/assets'
NAME = 'sit_high_support_050_center'
sys.path[1:1] = [str(HERE),str(ROOT/'experiments/furniture_load_repair_20261106_v3'),
                str(ROOT/'experiments/furniture_repair_20261106_v1/model_v4'),
                str(ROOT/'experiments/furniture_repair_20261106_v1'),
                str(ROOT/'scripts'), str(ROOT/'sitting_task/src'),
                str(ROOT/'experiments/rest_cycle_contact_20261108_v1')]


def main():
    out = HERE/'model_construction'; out.mkdir(exist_ok=False)
    try:
        frozen = json.loads((HERE/'freeze.json').read_text())
        for name, digest in frozen['files'].items():
            assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest, name
        import numpy as np
        from humanoid_sitting.shared_frame import BodyModel, level_rotation
        from seat_plan import SeatPlan
        from audit_seat_plan import audit as audit_geometry
        from complete_seat_path import complete
        from audit_complete_path import audit as audit_commands
        from audit_mesh_clearance import audit as audit_clearance
        from recording import write
        sys.path.insert(1,str(ROOT/'experiments/furniture_perception_repair_20261029_v1'))
        import audit_depth
        audit_depth.ASSETS=ASSETS
        model = BodyModel(ROOT/'runs/g1_kinematics_geometry_04/model.json')
        inertia = json.loads(Path('/data/humanoid-sitting-perception/crouch_rest_bridge_01/balanced_standing10_01/static_inertial_model.json').read_text())
        shapes = json.loads((ROOT/'experiments/furniture_perception_repair_20261029_v1/render_colliders.json').read_text())['shapes']
        variant = next(v for v in json.loads((ASSETS/'protocol.json').read_text())['variants'] if v['id']==NAME)
        case = next(c for c in json.loads((ASSETS/'cases.json').read_text()) if c['id']==variant['case'])
        depth_check=json.loads((SOURCE.parent/'model_construction/current_depth_audit.json').read_text())
        assert depth_check['input_sha256']==hashlib.sha256(SOURCE.read_bytes()).hexdigest()
        write(out/'current_depth_audit.json',depth_check)
        assert depth_check['geometry_pass'] and depth_check['target_observable'] and depth_check['original_plane_gate'],'CURRENT_DEPTH_MESH_REJECTION'
        mesh = Path(case['mesh_npz'])
        with np.load(mesh) as data: furniture = dict(data)
        allowed = json.loads(mesh.with_name('target_region.json').read_text())['triangle_indices']
        with np.load(SOURCE) as z, np.load(ASSETS/'banks'/NAME/'reset_candidates.npz') as bank:
            assert z['geometry'][11]>.5 and z['geometry'][12]==0 and int(z['physics_step'])==100
            snapshot=json.loads((SOURCE.parent/'current_snapshot_binding.json').read_text())
            assert snapshot['frame_sha256']==hashlib.sha256(SOURCE.read_bytes()).hexdigest()
            previous=np.asarray(snapshot['previous_command'])
            assert np.array_equal(previous,bank['command'][0,0])
            q = z['q'].copy(); gravity = z['gravity'].copy()
            geometry = z['geometry'].copy(); points = z['selected_points'].copy()
            plan = SeatPlan(model, inertia, shapes, q, gravity, geometry, points, previous)
            # These are used only by the saved-state collision auditor below.
            basis = z['root_rotation'] @ level_rotation(gravity.astype(float).copy()).T
            initial = z['root_position'].copy()
            binding = dict(source=str(SOURCE), sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                source_physics_step=int(z['physics_step']), source_clock_s=float(z['clock_s']),
                q=q.tolist(), geometry=geometry.tolist(), current_points_count=len(points),
                frame='current startup pelvis', synthetic_geometry_translation_m=0.,
                map_localization_required=False, copied_old_plan_or_endpoint=False,
                camera_intrinsics=z['K'].tolist(), camera_resolution=list(z['depth'].shape),
                independent_reset_command_source=str(ASSETS/'banks'/NAME/'reset_candidates.npz'))
        write(out/'input_binding.json', binding)
        from front_edge import qualify
        with np.load(SOURCE) as z:
            edge_check=qualify(z['selected_points'],z['selected_pixels'],z['gravity'],z['depth'].shape)
            write(out/'edge_sensor_association.json',edge_check)
            assert edge_check['passed'],'CURRENT_BILATERAL_FRONT_EDGE_REJECTED'
            # Evaluation-only check of the measured line against the real tabletop.
            point_level=np.array([edge_check['edge_x_at_center_m'],0.,0.])
            point_level[2]=-(plan.d+plan.n[0]*point_level[0])/plan.n[2]
            predicted_world=z['root_position']+basis@point_level
            actual_edge=float(furniture['vertices'][furniture['faces'][allowed]].reshape(-1,3)[:,0].max())
            actual_check=dict(passed=abs(float(predicted_world[0])-actual_edge)<=.006,
                observed_front_world_x=float(predicted_world[0]),actual_front_world_x=actual_edge,
                error_m=float(predicted_world[0])-actual_edge,truth_used_for='independent admission only')
            write(out/'edge_geometry_audit.json',actual_check)
            assert actual_check['passed'],'BILATERAL_EDGE_DOES_NOT_MATCH_FIXTURE'
        construction = plan.solve(); write(out/'plan.json', construction)
        association=construction['contact_association']
        assert np.array_equal(np.asarray(association['local_point']),np.asarray(shapes[association['shape_index']]['vertices'][association['vertex_index']]))
        assert shapes[association['shape_index']]['body']=='pelvis'
        write(out/'material_association_audit.json',dict(passed=True,association=association))
        geometry_check = audit_geometry(plan, construction)
        path = complete(plan, construction); write(out/'complete.json', path)
        command_check = audit_commands(plan, construction, path)
        nodes = construction['nodes'] + [dict(n, phase='landing') for n in path['landing'][1:]]
        clearance = audit_clearance(model, shapes, furniture, allowed, nodes, basis, initial)
        from audit_witness import audit as audit_witness
        witness_check=audit_witness(plan,construction)
        write(out/'actuator_witness_audit.json',witness_check)
        geometry_pass = all(v for k,v in geometry_check['tests'].items() if k!='time_for_original_hold')
        write(out/'audit.json', dict(model_admitted=bool(geometry_pass and command_check['passed'] and clearance['passed'] and witness_check['passed']),
            geometry_audit=geometry_check, command_audit=command_check, clearance_audit=clearance,
            current_patch_binding=str(out/'input_binding.json'), elapsed_seconds=time.monotonic()-START,
            physical_success=False, complete_cycle=False, physics_steps=0, PPO_updates=0))
        print(json.dumps(dict(model_admitted=bool(geometry_pass and command_check['passed'] and clearance['passed'] and witness_check['passed']),
              minimum_patch_inset_m=geometry_check['min_patch_inset'],
              failed_geometry=[k for k,v in geometry_check['tests'].items() if not v],
              failed_commands=[k for k,v in command_check['tests'].items() if not v],
              clearance_passed=bool(clearance['passed']), elapsed_seconds=time.monotonic()-START)))
    except BaseException as e:
        if 'plan' in locals():
            (out/'stage_evidence.json').write_text(json.dumps(getattr(plan,'stage_evidence',{'endpoint_solver':getattr(plan,'endpoint_solver',None)}),indent=2)+'\n')
        with (out/'failure.json').open('x') as f:
            json.dump(dict(error=repr(e),traceback=traceback.format_exc(),elapsed_seconds=time.monotonic()-START,
                           physics_steps=0,PPO_updates=0,automatic_retry=False), f, indent=2)
        raise
    finally: signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == '__main__': main()
