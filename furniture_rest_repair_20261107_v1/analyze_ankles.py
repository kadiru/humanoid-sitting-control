"""Saved-data joint/contact analysis; no controller, optimizer or simulator launch."""
import sys,json
from pathlib import Path
import numpy as np
ROOT=Path('/home/kadir/humanoid-sitting');HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'sitting_task/src'))
from humanoid_sitting.shared_frame import BodyModel
from scipy.spatial.transform import Rotation
m=BodyModel(ROOT/'runs/g1_kinematics_geometry_04/model.json')
rows=[]
folders=[p for p in (ROOT/'outputs/furniture_camera_coverage_20261106_v1/calibration_demos').iterdir()]
folders+=[ROOT/'outputs/furniture_repair_20261106_v1/motor_axis_readout']
for folder in folders:
 d=dict(np.load(folder/'trajectory.npz'));names=json.loads((folder/'results.json').read_text())['body_names'];end=1000
 if d['failure'].any():end=int(np.flatnonzero(d['failure'][:,0])[0])
 start=400 if end>500 else 0
 report=dict(source=str(folder),window=[start*.02,end*.02],joints=[])
 for name in ['left_ankle_roll_joint','right_ankle_roll_joint','left_hip_roll_joint','right_hip_roll_joint']:
  j=m.names.index(name);q=d['q'][start:end,0,j];qd=d['physical_joint_velocity'][start:end,0,j];cmd=d['command'][start:end,0,j]
  fft=abs(np.fft.rfft(qd-qd.mean()));freq=np.fft.rfftfreq(len(qd),.02);peak=int(np.argmax(fft[1:])+1)
  fit=np.column_stack([q[:-1]-cmd[:-1],qd[:-1],np.ones(len(q)-1)])
  coef=np.linalg.lstsq(fit,qd[1:],rcond=None)[0];res=qd[1:]-fit@coef
  report['joints'].append(dict(name=name,q_range_rad=[float(q.min()),float(q.max())],command_range_rad=[float(cmd.min()),float(cmd.max())],mean_abs_qd=float(abs(qd).mean()),max_abs_qd=float(abs(qd).max()),dominant_velocity_Hz=float(freq[peak]),lag1_correlation=float(np.corrcoef(qd[:-1],qd[1:])[0,1]),velocity_recurrence_fit=coef.tolist(),velocity_fit_rmse=float(np.sqrt(np.mean(res**2))),mean_applied_torque=float(d['physical_torque'][start:end,0,j].mean())))
 for side in ['left','right']:
  pose=d['body'][start:end,0,names.index(side+'_ankle_roll_link')];e=Rotation.from_quat(pose[:,3:7]).as_euler('xyz')
  report[side+'_foot']=dict(roll_range_deg=np.rad2deg([e[:,0].min(),e[:,0].max()]).tolist(),pitch_range_deg=np.rad2deg([e[:,1].min(),e[:,1].max()]).tolist(),world_position_range_m=np.ptp(pose[:,:3],axis=0).tolist())
 rows.append(report)
(HERE/'ankle_diagnosis.json').write_text(json.dumps(dict(rows=rows,physics_steps=0,effort_is_implicit_PD_telemetry_not_energy=True),indent=2));print(json.dumps(rows,indent=2))
