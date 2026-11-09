"""Recorded-model diagnosis only: unused sagittal stance support in paths91/95."""
from pathlib import Path
import datetime, hashlib, json
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from humanoid_sitting.shared_frame import BodyModel, apply, transform

D = Path('/data/humanoid-sitting-perception/low_seat_contact_01')
O = D / 'stance_support_diagnosis_97'
def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    assert read(D/'landing_ankle_path_audit_96/independent_audit.json')['passed']
    O.mkdir(exist_ok=False)
    mp = D/'cooked_model_50/model.json'
    model = BodyModel(mp)
    limits={j['name']:np.deg2rad(j['limits']) for j in model.joints
            if j['type']=='PhysicsRevoluteJoint'}
    moving=[i for i,n in enumerate(model.names) if any(t in n for t in
            ['hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch','ankle_roll',
             'shoulder_pitch','elbow_pitch'])]
    records = []
    files = [Path(__file__).resolve(), mp,
             D/'landing_ankle_path_audit_96/independent_audit.json']
    for name in ['contact_path_screen_v2_91', 'landing_ankle_path_95']:
        source = D/name/'results.json'; files.append(source)
        data = read(source); z = data['dense']; start = z['path'][0]
        root = np.asarray(start['root']); q = np.asarray(start['q'])
        T = transform(Rotation.from_quat(root[3:]).as_matrix(), root[:3])
        v = apply(T @ model.fk(q)['left_ankle_roll_link'],
                  model.vertices['left_ankle_roll_link'])
        sole = v[v[:,2] < v[:,2].min()+.003]
        H = ConvexHull(sole[:,:2]).equations
        target = np.asarray(z['path'][16]['com_target_xy'])
        # Halfspace normals returned by Qhull have unit length. Apply the same
        # existing 5.5mm target reserve; keep physical/model limits separate.
        xmin, xmax = -np.inf, np.inf
        for hx, hy, hb in H:
            rhs = -.0055 - hy*target[1] - hb
            if abs(hx) < 1e-12:
                assert rhs >= -1e-12
            elif hx > 0: xmax = min(xmax, rhs/hx)
            else: xmin = max(xmin, rhs/hx)
        assert xmin <= target[0] <= xmax
        candidates = []
        for dx in [0., -.02, -.04, -.06, -.08]:
            xy = target + [dx, 0.]
            candidates.append(dict(rearward_offset_m=-dx, target_xy=xy.tolist(),
                target_margin_m=float(np.min(-(H[:,:2]@xy+H[:,2]))),
                inside_existing_target_reserve=bool(xmin <= xy[0] <= xmax)))
        ankle = model.names.index('right_ankle_pitch_joint')
        failure_records = []
        for k in z['failed_knots']:
            e = z['path'][k]
            failure_records.append(dict(knot=k, phase=e['phase'], u=e['u'],
                com_target_xy=e['com_target_xy'], root=e['root'],
                right_ankle_pitch_rad=e['q'][ankle],
                foot_position_error_m=e['foot_position_error_m'],
                foot_orientation_error_rad=e['foot_orientation_error_rad']))
        command_limits=[]; phase_arcs={}
        for k,node in enumerate(z['command_nodes']):
            cmd=np.asarray(node['command'])
            margin,joint=min((min(cmd[i]-limits[model.names[i]][0],
                                 limits[model.names[i]][1]-cmd[i]),model.names[i])
                             for i in moving)
            if margin<.03001:
                command_limits.append(dict(knot=k,phase=z['path'][k]['phase'],
                    joint=joint,margin_rad=float(margin),passed=bool(margin>=.03)))
            if k:
                phase=z['path'][k]['phase']
                arc=float(np.max(abs(cmd-np.asarray(z['command_nodes'][k-1]['command']))))
                phase_arcs[phase]=phase_arcs.get(phase,0.)+arc
        records.append(dict(source=str(source), fixed_stance_target_xy=target.tolist(),
            left_sole_xy=sole[:,:2].tolist(), normalized_halfspaces=H.tolist(),
            sagittal_interval_at_fixed_y_with_5_5mm_reserve=[xmin,xmax],
            available_rearward_shift_m=float(target[0]-xmin),
            target_only_checks=candidates, failed_knots=failure_records,
            command_limit_records=command_limits,command_arc_by_phase_rad=phase_arcs,
            model_geometry_passed=z['model_geometry_passed'],
            model_command_passed=z['model_command_passed'],
            command_schedule_passed=z['command_schedule_passed']))
    result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        scope='OFFLINE diagnosis of fixed sagittal COM target in recorded paths91/95. '
              'The target was the closest original-COM point inside the left stance sole. '
              'This computes unused rearward support range using the same5.5mm target reserve. '
              'No new IK, optimized trajectory, commands, force allocation or simulation. '
              'An inset COM point alone does not prove feasible ankle motion, effort, '
              'contact transfer, friction, sensed loading or dynamic safety.',
        records=records, source_sha256={str(p):sha(p) for p in files},
        physics_ticks=0, optimizer_updates=0, holdout_used=False, physical_admission=False)
    (O/'diagnosis.json').write_text(json.dumps(result,indent=2))
    (O/'SHA256.json').write_text(json.dumps({'diagnosis.json':sha(O/'diagnosis.json')},indent=2))
    print(json.dumps([dict(source=r['source'],available_rearward_shift_m=r['available_rearward_shift_m'],
        checks=r['target_only_checks']) for r in records]),flush=True)

if __name__=='__main__': main()
