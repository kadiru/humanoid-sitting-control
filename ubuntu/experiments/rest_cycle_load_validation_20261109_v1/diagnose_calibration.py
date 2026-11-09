"""One bounded saved-record attribution; no simulator, optimizer, or tuning."""
import os,time,resource,signal,json,hashlib
from pathlib import Path
START=time.monotonic()
os.environ.update(CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
resource.setrlimit(resource.RLIMIT_AS,(40*1024**3,40*1024**3))
def deadline(*_):raise TimeoutError('45_SECOND_SAVED_DIAGNOSIS')
signal.signal(signal.SIGALRM,deadline);signal.setitimer(signal.ITIMER_REAL,45)
import numpy as np
H=Path(__file__).resolve().parent;O=Path('/home/kadir/humanoid-sitting/outputs')/H.name/'narrow_load_calibration'
CONTACTS=['pelvis','left_hip_yaw_link','right_hip_yaw_link','torso_link','logo_link','left_ankle_roll_link','right_ankle_roll_link']
def main():
    frozen=json.loads((H/'diagnosis_freeze.json').read_text())
    for p,h in frozen['files'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
    with np.load(O/'trajectory.npz') as f:z=dict(f)
    meta=json.loads((O/'results.json').read_text());sg=json.loads((O/'controller_signals.json').read_text())
    bw=z['weight'][:,0];force=z['physical_force_w'][:,0];speed=z['physical_slip_m_s'][:,0]
    active=np.linalg.norm(force,axis=-1)>.01*bw[:,None];slip=active&(speed>.02)
    assert np.array_equal(slip.any(1),~z['no_slip'][:,0])
    events=[]
    for t,b in zip(*np.where(slip)):
        events.append(dict(control=int(t+1),time_s=float((t+1)*.02),body=CONTACTS[b],speed_m_s=float(speed[t,b]),force_bw=float(np.linalg.norm(force[t,b])/bw[t]),path_index=sg[t]['path_index'],support_bw=float(z['support_bw'][t,0]),
          neighbors=[dict(control=int(k+1),speed=float(speed[k,b]),force_bw=float(np.linalg.norm(force[k,b])/bw[k]),support_bw=float(z['support_bw'][k,0])) for k in range(max(0,t-3),min(1000,t+4))]))
    q=z['q_pre'][:,0,:23];qd=z['qd_pre'][:,0,:23];tau=z['native_motor_tau_pre'][:,0,:23]
    rankings=np.argsort(np.max(abs(qd[-250:]),axis=0))[::-1]
    rows=[]
    for j in rankings[:6]:
        rows.append(dict(joint=meta['joint_names'][j],final5s_raw_speed_max_rad_s=float(abs(qd[-250:,j]).max()),final5s_raw_speed_mean_rad_s=float(qd[-250:,j].mean()),final5s_angle_range_rad=float(np.ptp(q[-250:,j])),final5s_angle_change_rad=float(q[-1,j]-q[-250,j]),sampled_velocity_integral_rad=float(qd[-250:,j].sum()*.02),final5s_tracking_error_max_rad=float(abs(q[-250:,j]-z['previous'][-250:,0,j]).max())))
    start=slice(0,24);end=slice(-250,None)
    torquesq=np.sum(tau*tau,axis=1);ratio=float(torquesq[end].mean()/torquesq[start].mean())
    result=dict(slip_events=events,velocity_rows=rows,raw_quiet_final250=int((np.max(abs(qd[end]),axis=1)<=.05).sum()),native_motor_effort=dict(active_joints=23,standing_first24_samples_mean_tau_squared_Nm2=float(torquesq[start].mean()),final250_mean_tau_squared_Nm2=float(torquesq[end].mean()),hold_over_standing_ratio=ratio,proxy_only=True,electrical_energy_measured=False),
      final5s_feet_vertical_load_bw=float((force[end,5:,2].sum(1)/bw[end]).mean()),final5s_support_bw=float(z['support_bw'][end,0].mean()),camera_valid_controls=int(sum(v['geometry_valid'] for v in sg)),maximum_camera_age_s=max(v['geometry_age_s'] for v in sg),
      readout_cause_proven=False,criteria_changed=False,physics_steps=0,PPO_updates=0,wall_seconds=time.monotonic()-START)
    with (O/'saved_diagnosis.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result))
if __name__=='__main__':
    try:main()
    finally:signal.setitimer(signal.ITIMER_REAL,0)
