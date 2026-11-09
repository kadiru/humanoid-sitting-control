"""Independent finite-work force-authority diagnosis; no simulation or motion search."""
from pathlib import Path
import datetime
import hashlib
import json

import numpy as np
from scipy.optimize import linprog
from scipy.spatial.transform import Rotation

from humanoid_sitting.shared_frame import BodyModel, apply, transform
from low_seat_joint_path_controller_17 import measured

D = Path('/data/humanoid-sitting-perception/low_seat_contact_01')
O = D / 'contact_force_authority_104'


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    O.mkdir(exist_ok=False)
    source = D / 'stance_com_path_v2_98/results.json'
    prior = D / 'contact_force_schedule_102/results.json'
    assert not read(prior)['summary']['success']
    z = read(source)['dense']
    assert z['model_geometry_passed'] and len(z['path']) == 115
    mp = D / 'cooked_model_50/model.json'
    ip = D.parent / 'crouch_rest_bridge_01/balanced_standing10_01/static_inertial_model.json'
    cp = D.parent / 'contact_effort_repair_01/plane_calibration.json'
    m, inertia = BodyModel(mp), read(ip)
    names = m.names
    moving = [names.index(s+'_'+j+'_joint') for s in ['left', 'right']
              for j in ['hip_pitch', 'hip_roll', 'hip_yaw', 'knee', 'ankle_pitch', 'ankle_roll']]
    moving += [names.index(s+'_'+j+'_joint') for s in ['left', 'right']
               for j in ['shoulder_pitch', 'elbow_pitch']]
    kp = np.array([200, 150, 150, 200, 160, 160]*2 + [40]*4)
    effort = np.array([300, 300, 300, 300, 20, 20]*2 + [300]*4)
    limits = {j['name']: np.deg2rad(j['limits']) for j in m.joints
              if j['type'] == 'PhysicsRevoluteJoint'}
    lo = np.array([limits[names[j]][0] for j in moving])
    hi = np.array([limits[names[j]][1] for j in moving])
    mass = np.array(inertia['mass'])
    W = mass.sum()*9.81
    src = D / 'normal_landing3s_06/candidate'
    tr = np.load(src/'trace.npz')
    sensors = np.load(src/'sensor_inputs.npz')['sensors']
    at, case = int(np.argmin(abs(tr['t']-1.))), 2
    startq = tr['q'][at, case].astype(float)
    assert np.array_equal(startq[:23].astype(np.float32), sensors[0, 10, case, 270:293])
    n, d, *_ = measured(m, startq, sensors[0, 10, case, 339:342],
                        tr['geometry'][at, case], [tr['estimated_root_x'][at, case], 0, 0], read(cp))
    command0 = np.array(z['command_nodes'][0]['command'])
    q0 = np.array(z['path'][0]['q'])
    rows, work_errors, matrices = [], [], {}
    options = dict(primal_feasibility_tolerance=1e-10, dual_feasibility_tolerance=1e-10)
    for k, (e, v) in enumerate(zip(z['path'], z['command_nodes'])):
        q, root = np.array(e['q']), np.array(e['root'])
        T = transform(Rotation.from_quat(root[3:]).as_matrix(), root[:3])
        fk = m.fk(q)
        C = np.average([apply(T@fk[b], cc) for b, cc in zip(inertia['body_names'], inertia['local_com'])], axis=0, weights=mass)
        P, sides = np.array(v['corner_world_points']), v['corner_sides']
        local = [np.linalg.inv(T@fk[s+'_ankle_roll_link'])@np.r_[p, 1.] for p, s in zip(P, sides)]
        grav, B = [], []
        for j in moving:
            energies, corner_work = [], []
            for sign in [-1, 1]:
                qq = q.copy()
                qq[j] += sign*1e-5
                ff = m.fk(qq)
                energies.append(sum(mm*9.81*apply(T@ff[b], cc)[2] for b, mm, cc in zip(inertia['body_names'], mass, inertia['local_com'])))
                corner_work.append([W*(T@ff[s+'_ankle_roll_link']@ll)[2] for s, ll in zip(sides, local)])
            grav.append((energies[1]-energies[0])/2e-5)
            B.append(-(np.array(corner_work[1])-corner_work[0])/2e-5)
        grav, B = np.array(grav), np.array(B)
        wold = np.array(v['corner_fractions'])
        tau = grav+B@wold
        err = float(max(abs(tau-v['planned_static_torque_Nm'])))
        assert err < 1e-6
        work_errors.append(err)
        if k == 0:
            tau0 = tau.copy()
        base = command0[moving]+q[moving]-q0[moving]+(grav-tau0)/kp
        M = B/kp[:, None]
        pv = apply(T, m.vertices['pelvis'])
        pd = pv@n+d
        seat = pv[pd <= pd.min()+1e-7].mean(0)
        seat -= n*(seat@n+d)
        normal = v['assumed_seat_normal_fraction']
        assert normal == 0 or abs(pd.min()) < 1e-8
        sv = normal/n[2]
        A = np.vstack([np.ones(len(P)), P[:, :2].T, [s == 'right' for s in sides]])
        b = np.r_[1-sv, C[:2]-sv*seat[:2], v['planned_right_load_fraction']]
        bounds = [(float(w), float(w)) for w in wold] if k in [0, 114] else [(0, 1)]*len(P)
        # Maximize the smallest distance of any command to its hard joint limit.
        U = np.vstack([np.c_[M, np.ones(16)], np.c_[-M, np.ones(16)],
                       np.c_[B, np.zeros(16)], np.c_[-B, np.zeros(16)]])
        rhs = np.r_[hi-base, base-lo, effort-grav, effort+grav]
        objective = np.r_[np.zeros(len(P)), -1.]
        fit = linprog(objective, A_ub=U, b_ub=rhs, A_eq=np.c_[A, np.zeros(len(b))], b_eq=b,
                      bounds=bounds+[(-1., 1.)], method='highs', options=options)
        assert fit.success, (k, fit.message)
        w = fit.x[:-1]
        cmd = base+M@w
        margin = float(min(np.r_[cmd-lo, hi-cmd]))
        eq = float(max(abs(A@w-b)))
        assert eq < 1e-8 and abs(margin-fit.x[-1]) < 1e-8
        assert min(w) >= -1e-10 and max(abs(grav+B@w)/effort) <= 1+1e-8
        oldcmd = base+M@wold
        assert max(abs(oldcmd-np.array(v['command'])[moving])) < 2e-8
        limiting = [names[moving[i]] for i in range(16) if min(cmd[i]-lo[i], hi[i]-cmd[i]) <= margin+1e-8]
        rows.append(dict(knot=k, phase=e['phase'], u=e['u'], best_margin_rad=margin,
                         original_margin_rad=v['min_command_margin_rad'], limiting_joints=limiting,
                         actual_command_limit_satisfied=bool(margin >= .03),
                         tightened_lp_limit_satisfied=bool(margin >= .03001),
                         forces=w.tolist(), max_equilibrium_residual=eq,
                         force_work_error_Nm=err, assumed_seat_normal_fraction=normal,
                         right_load_fraction=v['planned_right_load_fraction'],
                         solver_inequality_duals=fit.ineqlin.marginals.tolist(),
                         solver_equality_duals=fit.eqlin.marginals.tolist(),
                         solver_lower_duals=fit.lower.marginals.tolist(),
                         solver_upper_duals=fit.upper.marginals.tolist(), objective=float(fit.fun)))
        matrices.update({f'{k}_U': U, f'{k}_rhs': rhs, f'{k}_A': np.c_[A, np.zeros(len(b))],
                         f'{k}_b': b, f'{k}_bounds': np.array(bounds+[(-1., 1.)]),
                         f'{k}_x': fit.x, f'{k}_objective': objective})
    np.savez_compressed(O/'lp_witnesses.npz', **matrices)
    failures = [x['knot'] for x in rows if not x['actual_command_limit_satisfied']]
    files = [Path(__file__).resolve(), source, prior, mp, ip, cp, src/'trace.npz', src/'sensor_inputs.npz']
    out = dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
               scope='Independent finite-difference virtual-work construction and one margin-maximizing LP per fixed path98 knot. 115 LPs, no geometry search. Preserve each knot total right-foot and assumed seat load, exact endpoint forces and effort limits. Tests actual .03 command bound separately from102 internal .03001 reserve. Saved primal/dual witnesses allow independent arithmetic checks; local numerical optimization is not a dynamic or global impossibility proof.',
               summary=dict(knots=115, failing_actual_limit_knots=failures,
                            min_best_margin_rad=min(x['best_margin_rad'] for x in rows),
                            max_virtual_work_error_Nm=max(work_errors),
                            path_force_distribution_can_meet_actual_command_limit=not failures),
               nodes=rows, source_sha256={str(p.resolve()): sha(p) for p in files},
               physics_ticks=0, optimizer_updates=0, holdout_used=False, physical_admission=False)
    (O/'results.json').write_text(json.dumps(out, indent=2))
    (O/'SHA256.json').write_text(json.dumps({p.name: sha(p) for p in O.iterdir() if p.is_file()}, indent=2))
    print(json.dumps(out['summary']), flush=True)


if __name__ == '__main__':
    main()
