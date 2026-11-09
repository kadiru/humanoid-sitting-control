"""Independent saved-fixture result and fixed reproduction predicates; no physics."""
import os,time,signal,resource,json,hashlib
from pathlib import Path
START=time.monotonic();os.environ.update(CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});resource.setrlimit(resource.RLIMIT_AS,(40*1024**3,40*1024**3))
def deadline(*_):raise TimeoutError('45_SECOND_FIXED_FIXTURE_AUDIT')
signal.signal(signal.SIGALRM,deadline);signal.setitimer(signal.ITIMER_REAL,45)
import numpy as np
H=Path(__file__).resolve().parent;O=H.parent.parent/'outputs'/H.name/'fixture'
with np.load(O/'native_states.npz') as z:q=z['q'];v=z['qd'];target=z['target'];dt=float(z['dt'])
assert q.shape==v.shape==target.shape==(2000,2)
assert all(np.isfinite(a).all() for a in (q,v,target)) and dt==.005
initial=json.loads((O/'initialization.json').read_text())
p=np.asarray(initial['q'])[:,0];maximum=0.;filter_error=0.
for c in range(500):
    actual=target[4*c];pred=p+np.clip(-.1*p,-.003,.003)
    filter_error=max(filter_error,float(abs(pred-actual).max()));maximum=max(maximum,float(abs(actual-p).max()))
    assert np.array_equal(target[4*c:4*c+4],np.broadcast_to(actual,(4,2)))
    p=actual
assert filter_error<=2e-7 and maximum<=.003+2e-7
x=q[-1000:,1].astype(float);speed=v[-1000:,1].astype(float)
criteria=dict(loaded_angle_range_under_1e_4_rad=float(np.ptp(x))<=1e-4,
 loaded_mean_native_speed_exceeds_0_05_rad_s=abs(float(speed.mean()))>.05,
 loaded_mean_speed_disagrees_with_position_change_by_over_0_05_rad_s=abs(float(speed.mean())-float(x[-1]-x[0])/(999*dt))>.05)
result=dict(reproduction_passed=all(criteria.values()),criteria=criteria,
 setup_and_data_checks_passed=True,maximum_target_increment_rad=maximum,maximum_filter_error_rad=filter_error,
 initial_physics_steps=initial['physics_steps'],active_physics_steps=2000,fixture_transitions=4000,
 physics_configuration_changed=False,readings_corrected=False,robot_repair=False,complete_robot_cycles=0,
 trajectory_sha256=hashlib.sha256((O/'native_states.npz').read_bytes()).hexdigest(),wall_s=time.monotonic()-START)
with (O/'independent_audit.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result));signal.setitimer(signal.ITIMER_REAL,0)
