"""Independent original physical verdict and loaded-sensing admission."""
import os,time,signal,resource,sys,json,hashlib
from pathlib import Path
START=time.monotonic();os.environ.update(CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});resource.setrlimit(resource.RLIMIT_AS,(40*1024**3,40*1024**3))
def timeout(*_):raise TimeoutError('45_SECOND_CALIBRATION_AUDIT')
signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,45)
R=Path('/home/kadir/humanoid-sitting');H=Path(__file__).resolve().parent;O=R/'outputs'/H.name/'narrow_load_calibration'
sys.path[:0]=[str(R/'experiments/furniture_load_repair_20261106_v3'),str(R/'sitting_task/src')]

def main():
    import numpy as np
    from humanoid_sitting.shared_frame import level_rotation
    from audit_rollout import audit
    for p,h in json.loads((H/'calibration_freeze.json').read_text())['files'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
    physical=audit(O)
    with np.load(O/'trajectory.npz') as f:z=dict(f)
    frame=z['obs'][1:,0,222:333];g=z['gravity_pre'][1:,0]
    normal=np.stack([level_rotation(x.astype(float).copy())@v[100:103] for x,v in zip(g,frame)])
    estimate=(z['estimated_rest_force_pre'][1:,0]*normal).sum(1)/z['weight'][:-1,0]
    label=z['support_bw'][:-1,0]
    quiet=(abs(z['qd_pre'][1:,0,:23]).max(1)<=.05)&(abs(z['gyro_pre'][1:,0,1])<=.02)
    valid=(frame[:,109]>.5)&(frame[:,110]>=0)&(frame[:,110]<=.21)&z['estimate_available_pre'][1:,0]
    eligible=quiet&valid;lo=estimate-.02;hi=estimate+.02
    loaded=eligible&(lo>=.05);errors=abs(estimate-label)
    false=int((loaded&(label<.05)).sum());miss=int((eligible&((label<lo)|(label>hi))).sum())
    tests=dict(original_physical_pass=physical['admitted'],qualified_final_hold=bool(loaded[-250:].all()),
      final5s_load_accuracy=bool(errors[-250:].max()<=.02),no_false_contact=false==0,no_interval_misses=miss==0,
      complete_path=json.loads((O/'capture_receipt.json').read_text())['path_samples_executed']==json.loads((O/'capture_receipt.json').read_text())['path_samples_expected'])
    result=dict(passed=bool(all(tests.values())),tests={k:bool(v) for k,v in tests.items()},
      physical=physical,loaded_qualified_final250=int(loaded[-250:].sum()),raw_quiet_final250=int(quiet[-250:].sum()),
      final5s_load_mean_error_bw=float(errors[-250:].mean()),final5s_load_max_error_bw=float(errors[-250:].max()),
      mean_final5s_sensor_load_bw=float(estimate[-250:].mean()),mean_final5s_label_load_bw=float(label[-250:].mean()),
      false_loaded=false,interval_misses=miss,elapsed_seconds=time.monotonic()-START,
      complete_cycle=False,stand_up_validated=False,PPO_updates=0,
      meaning='Only a passing result permits proceeding to current sensor-driven return validation; never a three-start completion claim.')
    (O/'load_admission.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':
    try:main()
    finally:signal.setitimer(signal.ITIMER_REAL,0)
