"""Validate a current-sensor force-balance observer on two saved lean rollouts.

All predictions use encoders, IMU and recorded motor torques. Physical forces
are read only after prediction to measure error; never input to the observer.
"""
import os,sys,json,time,hashlib
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path('/home/kadir/humanoid-sitting');HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'sitting_task/src')]
import numpy as np
from humanoid_sitting.shared_frame import BodyModel,level_rotation
from foot_wrench_observer import FootWrenchObserver
read=lambda p:json.loads(Path(p).read_text())
freeze=read(HERE/'lean_observer_freeze.json')
for n,h in freeze['files'].items():assert hashlib.sha256(Path(n).read_bytes()).hexdigest()==h,n
model=BodyModel(ROOT/'runs/g1_kinematics_geometry_04/model.json')
inertia=read('/data/humanoid-sitting-perception/crouch_rest_bridge_01/balanced_standing10_01/static_inertial_model.json')
observer=FootWrenchObserver(model,inertia);rows=[]
for name in freeze['cases']:
    start=time.monotonic();p=ROOT/'outputs/furniture_examples_20261105_v2'/name
    d=dict(np.load(p/'trajectory.npz'));pred=[];conditions=[];normals=[]
    for t in range(1000):
        frame=d['obs'][t,0,222:333];q=d['q_pre'][t,0];gravity=frame[69:72]
        torque=np.zeros(37);torque[:23]=frame[46:69]*d['physical_torque_scale'][t,0,:23]
        wrench,cond=observer.estimate(q,torque,gravity)
        force=np.array([-wrench[:,0].sum(),0.,observer.weight-wrench[:,1].sum()])
        pred.append(force);conditions.append(cond)
        normal=level_rotation(gravity.astype(float))@frame[100:103]
        normals.append(normal/np.linalg.norm(normal))
    pred=np.array(pred);normals=np.array(normals);conditions=np.array(conditions)
    # Startup in-air bias uses only the initial sensor-derived prediction.
    # It is frozen once and never calibrated from contact or force truth.
    corrected=pred-pred[0]
    actual=d['physical_force_w'][:-1,0,:5].sum(1)
    # The gravity frame retains robot yaw; rotate evaluation forces into that frame.
    frame0=np.load(p/'frame_0.npz');world_basis=frame0['root_rotation']@level_rotation(frame0['gravity'].astype(float)).T
    actual=actual@world_basis
    evaluate=corrected[1:];n=normals[1:];up=np.array([0.,0.,1.])-n*n[:,2,None];up/=np.linalg.norm(up,axis=1)[:,None]
    estimated_nt=np.c_[(evaluate*n).sum(1),(evaluate*up).sum(1)]
    actual_nt=np.c_[(actual*n).sum(1),(actual*up).sum(1)]
    error=estimated_nt[-250:]-actual_nt[-250:]
    row=dict(case=name,seconds=time.monotonic()-start,maximum_condition=float(conditions.max()),
             final_predicted_normal_shear_N=estimated_nt[-1].tolist(),final_actual_normal_shear_N=actual_nt[-1].tolist(),
             final250_mean_abs_error_N=np.mean(abs(error),axis=0).tolist(),
             startup_sensor_bias_N=pred[0].tolist(),
             passed=bool(conditions.max()<200 and np.max(np.mean(abs(error),axis=0))<.02*observer.weight),
             physical_success=False)
    rows.append(row);print(json.dumps(row),flush=True)
with (HERE/'lean_observer_check.json').open('x') as f:json.dump(dict(cases=rows,simulation_steps=0,ppo_updates=0),f,indent=2)
