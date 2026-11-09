"""One frozen saved-data validation, no tuning, simulation or deployment."""
import os, time, signal, resource, sys, json, hashlib, traceback
from pathlib import Path
START = time.monotonic()
os.environ.update(CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
resource.setrlimit(resource.RLIMIT_AS, (40*1024**3, 40*1024**3))
def deadline(*_): raise TimeoutError('45_SECOND_TOTAL_LOAD_VALIDATION')
signal.signal(signal.SIGALRM, deadline); signal.setitimer(signal.ITIMER_REAL, 45)
R = Path('/home/kadir/humanoid-sitting'); H = Path(__file__).resolve().parent
sys.path.insert(0, str(R/'sitting_task/src'))

def main():
    out = H/'validation'; out.mkdir(exist_ok=False)
    def write(name, x): (out/name).write_text(json.dumps(x, indent=2, allow_nan=False)+'\n')
    try:
        for p, h in json.loads((H/'freeze.json').read_text())['files'].items():
            assert hashlib.sha256(Path(p).read_bytes()).hexdigest() == h, p
        import numpy as np
        from observer import SpatialLoadObserver
        from humanoid_sitting.shared_frame import BodyModel, level_rotation
        model = BodyModel(R/'runs/g1_kinematics_geometry_04/model.json')
        inertia = json.loads(Path('/data/humanoid-sitting-perception/crouch_rest_bridge_01/balanced_standing10_01/static_inertial_model.json').read_text())
        obs = SpatialLoadObserver(model, inertia)
        folder = R/'outputs/rest_cycle_wide_camera_20261108_v1/sit_coffee_wide'
        with np.load(folder/'trajectory.npz') as f: z = dict(f)
        with np.load(folder/'substeps.npz') as f: sub = dict(f)
        with np.load(folder/'frame_0.npz') as f: frame = dict(f)
        assert len(z['q']) == 1000 and len(sub['physics_step']) == 4000
        assert np.array_equal(np.diff(sub['physics_step']), np.ones(3999))
        # Native motor-axis measurement at the last substep matches q[t].
        # Sensor frame obs[t+1] supplies gravity/geometry for that same state.
        # No controller approximate-PD torque is substituted as a measurement.
        tau = sub['projected_effort'][3::4, 0]
        frame_sensor = z['obs'][1:, 0, 222:333]
        q = z['q'][:-1, 0]; gravity = frame_sensor[:, 69:72]
        assert np.max(abs(q[:, :23]-frame_sensor[:, :23])) < 2e-6
        basis = frame['root_rotation'] @ level_rotation(frame['gravity'].astype(float).copy()).T
        # Independent labels only; these values are never passed to estimate().
        truth = z['physical_force_w'][:-1, 0, :5].sum(1) @ basis
        foot_truth = z['physical_force_w'][:-1, 0, -2:] @ basis
        predicted = []; cond = []; jacs = []; rhs_all = []; solved = []
        for i in range(999):
            value, condition, jac, rhs = obs.estimate(q[i], tau[i], gravity[i])
            solved.append(value is not None)
            predicted.append(np.zeros(15) if value is None else value)
            cond.append(condition); jacs.append(jac); rhs_all.append(rhs)
        pred = np.asarray(predicted); cond = np.asarray(cond); solved = np.asarray(solved)
        normal = np.stack([level_rotation(g.copy()) @ f[100:103]
                           for g, f in zip(gravity, frame_sensor)])
        geometry_ok = (frame_sensor[:,109]>.5) & (frame_sensor[:,110]>=0) & (frame_sensor[:,110]<=.21)
        normal_load = (pred[:,:3]*normal).sum(1)/obs.weight
        label = z['support_bw'][:-1,0]
        # Retain existing gates. A velocity mismatch cannot be repaired by
        # substituting a finite-difference velocity or by broadening this test.
        raw_qd = frame_sensor[:,23:46]*10.
        quiet = (abs(raw_qd).max(1)<=.05) & (abs(frame_sensor[:,73]*3.)<=.02)
        admissible = solved & geometry_ok & quiet
        radius = .02  # Existing force-validation tolerance, not fitted to data.
        lower, upper = normal_load-radius, normal_load+radius
        loaded = admissible & (lower>=.05)
        unloaded = admissible & (upper<.01)
        # Negative/no-contact classification also requires a physically possible
        # interval. Strongly negative model loads are invalid, never 'unloaded'.
        admissible &= upper>=0
        loaded &= admissible; unloaded &= admissible
        errors = abs(normal_load-label)
        loaded_truth = label>=.05; absent = label<.01
        summary = dict(samples=999,method='six spatial joint equations per leg with native motor-axis effort',
          shape='2x6 equations, Fx/Fy/Fz/Mx/My/Mz at each foot',
          maximum_condition=float(cond.max()),solved=int(solved.sum()),
          geometry_valid=int(geometry_ok.sum()),raw_quiet=int(quiet.sum()),
          mean_abs_normal_error_bw=float(errors.mean()),final5s_mean_abs_normal_error_bw=float(errors[-250:].mean()),
          final5s_max_abs_normal_error_bw=float(errors[-250:].max()),
          final5s_mean_support_label_bw=float(label[-250:].mean()),
          final5s_mean_support_estimate_bw=float(normal_load[-250:].mean()),
          final5s_mean_foot_force_error_bw=(np.mean(np.linalg.norm(pred[:,3:].reshape(-1,2,6)[-250:,:,:3]-foot_truth[-250:],axis=-1),axis=0)/obs.weight).tolist(),
          declared_loaded=int(loaded.sum()),declared_unloaded=int(unloaded.sum()),
          false_loaded=int((loaded & ~loaded_truth).sum()),false_unloaded=int((unloaded & ~absent).sum()),
          interval_misses=int((admissible & ((label<lower)|(label>upper))).sum()),
          loaded_recall=float((loaded & loaded_truth).sum()/max(1,loaded_truth.sum())),
          unloaded_recall=float((unloaded & absent).sum()/max(1,absent.sum())),
          last5s_max_raw_joint_speed_rad_s=float(abs(raw_qd[-250:]).max()),
          leg_contact_label_samples=int((np.linalg.norm(z['physical_force_w'][:-1,0,1:3],axis=-1).sum(1)>.01*obs.weight).sum()),
          independent_physical_success_unchanged=False,
          simulation_launches=0,PPO_updates=0,actor_admitted=False)
        summary['saved_screen_passed'] = bool(summary['false_loaded']==summary['false_unloaded']==summary['interval_misses']==0 and
           summary['loaded_recall']>=.9 and summary['unloaded_recall']>=.9 and errors[-250:].mean()<=.02)
        summary['elapsed_seconds'] = time.monotonic()-START
        np.savez_compressed(out/'predictions.npz',prediction=pred,condition=cond,jacobian=jacs,rhs=rhs_all,
          q=q,gravity=gravity,torque=tau[:-1],normal=normal,truth=truth,foot_truth=foot_truth,
          load_label=label,load_prediction=normal_load,admissible=admissible,loaded=loaded,unloaded=unloaded,
          lower=lower,upper=upper,raw_qd=raw_qd,weight=np.array(obs.weight))
        write('result.json',summary); print(json.dumps(summary),flush=True)
    except BaseException as e:
        write('failure.json',dict(error=repr(e),traceback=traceback.format_exc(),elapsed_seconds=time.monotonic()-START));raise
    finally: signal.setitimer(signal.ITIMER_REAL,0)

if __name__=='__main__': main()
