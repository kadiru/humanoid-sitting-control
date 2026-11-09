"""Fixed saved-data source attribution; no predictor tuning or new simulation."""
import os,time,signal,resource,json,sys,hashlib
from pathlib import Path
START=time.monotonic()
os.environ.update(CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
resource.setrlimit(resource.RLIMIT_AS,(40*1024**3,40*1024**3))
def deadline(*_):raise TimeoutError('45_SECOND_TOTAL_MEASUREMENT_ATTRIBUTION')
signal.signal(signal.SIGALRM,deadline);signal.setitimer(signal.ITIMER_REAL,45)
R=Path('/home/kadir/humanoid-sitting');H=Path(__file__).resolve().parent;O=H/'validation'

def main():
    import numpy as np
    assert not (O/'measurement_attribution.json').exists()
    for p,h in json.loads((H/'audit_v2_freeze.json').read_text())['files'].items():
        assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
    z=dict(np.load(O/'predictions.npz'))
    model=json.loads((R/'runs/g1_kinematics_geometry_04/model.json').read_text())
    names=model['joint_names']
    indices=[[names.index(s+'_'+j+'_joint') for j in ('hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch','ankle_roll')] for s in ('left','right')]
    # Exact same already-produced spatial operator, same q and IMU, same labels;
    # change only measured solver torque to the previous approximatePD telemetry.
    with np.load(R/'outputs/rest_cycle_wide_camera_20261108_v1/sit_coffee_wide/trajectory.npz') as f:
        old_torque=f['physical_torque'][:-1,0]
    grav=z['rhs']+z['torque'][:,indices]
    old_foot=np.linalg.solve(z['jacobian'],(grav-old_torque[:,indices])[...,None])[...,0]
    old_rest=np.array([0.,0.,z['weight']])-old_foot[:,:,:3].sum(1)
    old_load=(old_rest*z['normal']).sum(1)/z['weight']
    label=z['load_label']
    old_error=abs(old_load-label);new_error=abs(z['load_prediction']-label)
    narrow=R/'outputs/rest_cycle_narrow_support_20261109_v1/camera_and_wrench_capture'
    with np.load(narrow/'standing_calibration.npz') as f:
        q=f['q'][:,0,:23];qd=f['physical_joint_velocity'][:,0,:23]
    quiet=(abs(qd).max(1)<=.05)
    longest=current=0
    for v in quiet:current=current+1 if v else 0;longest=max(current,longest)
    rows=[]
    for i in np.argsort(abs(qd).max(0))[-6:][::-1]:
        rows.append(dict(joint=names[int(i)],maximum_raw_speed_rad_s=float(abs(qd[:,i]).max()),
          measured_position_change_rad=float(q[-1,i]-q[0,i]),raw_velocity_integral_rad=float(qd[1:,i].sum()*.02)))
    value=dict(comparison='Only torque source changes; spatial Jacobians, q, IMU, labels unchanged',
      final5s_old_approximate_PD_mean_error_bw=float(old_error[-250:].mean()),
      final5s_native_motor_axis_mean_error_bw=float(new_error[-250:].mean()),
      whole_record_old_mean_error_bw=float(old_error.mean()),whole_record_native_mean_error_bw=float(new_error.mean()),
      narrow_standing=dict(samples=50,quiet_samples=int(quiet.sum()),longest_raw_quiet_controls=longest,
          required_consecutive_quiet_controls=25,passes_raw_joint_quiet_prerequisite=longest>=25,largest_velocity_channels=rows),
      velocity_values_not_replaced=True,no_threshold_changes=True,
      attribution_scope='Supports a torque-source correction on this saved sit; does not explain underlying native velocity inconsistency or validate loaded narrow-fixture control.',
      simulation_steps=0,PPO_updates=0,elapsed_seconds=time.monotonic()-START)
    (O/'measurement_attribution.json').write_text(json.dumps(value,indent=2)+'\n')
    np.savez_compressed(O/'measurement_attribution.npz',old_load=old_load,new_load=z['load_prediction'],label=label,
      narrow_q=q,narrow_qd=qd,narrow_quiet=quiet)
    print(json.dumps(value),flush=True)

if __name__=='__main__':
    try:main()
    finally:signal.setitimer(signal.ITIMER_REAL,0)
