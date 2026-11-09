"""Offline sensor-observer validation against preserved contact truth labels."""
import json,hashlib
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from humanoid_sitting.shared_frame import BodyModel
from foot_wrench_observer import FootWrenchObserver

ROOT=Path('/home/kadir/humanoid-sitting');OLD=Path('/data/humanoid-sitting-perception/crouch_rest_bridge_01');OUT=OLD.parent/'traction_transfer_01'
OUT.mkdir(exist_ok=True);dest=OUT/'observer_validation_01';dest.mkdir(exist_ok=False)
model=BodyModel(ROOT/'runs/g1_kinematics_geometry_04/model.json');inertial=json.loads((OLD/'balanced_standing10_01/static_inertial_model.json').read_text());observer=FootWrenchObserver(model,inertial)
folder=OLD/'foot_balance3_01';data=json.loads((folder/'results.json').read_text());t=np.load(folder/'trace.npz');contact=np.load(folder/'every_tick.npz');pelvis=data['body_names'].index('pelvis');est=[];cond=[];times=[];truth=[];valid=[]
for k,tm in enumerate(t['t']):
    if tm<9.9:continue
    ti=int(round(tm/.02))-1;assert abs(contact['t'][ti]-tm)<1e-8
    e=[];c=[]
    for i in range(18):
        # In an offline replay the saved orientation reconstructs the IMU's
        # gravity reading; it is not supplied to a runtime controller.
        gravity=Rotation.from_quat(t['body'][k,i,pelvis,3:]).inv().apply([0,0,-1])
        ee,cc=observer.estimate(t['q'][k,i],t['torque'][k,i],gravity);e.append(ee);c.append(cc)
    est.append(e);cond.append(c);times.append(tm);truth.append(contact['force'][ti,:,5:,:][:,:,[0,2]]);valid.append(~contact['failure'][ti])
est=np.asarray(est);truth=np.asarray(truth);valid=np.asarray(valid);times=np.asarray(times);err=est[:,:,:,:2]-truth
records={}
for label,mask in [('all_before_physical_failure',valid),('late_before_physical_failure',valid&(times[:,None]>=15))]:
    x=err[mask];records[label]=dict(samples=int(mask.sum()),rms_error_N=np.sqrt((x*x).mean((0,1))).tolist(),max_error_N=abs(x).max((0,1)).tolist(),bias_N=x.mean((0,1)).tolist())
records['per_case_final']=[]
for i in range(18):
    mask=(times>=15)&valid[:,i];x=err[mask,i]
    records['per_case_final'].append(dict(case=i,samples=int(mask.sum()),estimated_mean_N=est[mask,i,:2,:2].mean(0).tolist() if len(x) else None,truth_mean_N=truth[mask,i].mean(0).tolist() if len(x) else None,rms_error_N=np.sqrt((x*x).mean((0,1))).tolist() if len(x) else None))
records['max_condition']=float(np.max(cond));records['runtime_inputs']='Joint angles, applied joint torque, IMU gravity, static kinematic/inertial robot model only';records['scope']='Quasi-static estimate. Offline simulator orientation reconstructs saved IMU reading; contact truth evaluates accuracy only.'
(dest/'validation.json').write_text(json.dumps(records,indent=2));np.savez_compressed(dest/'observer_trace.npz',t=times,estimated=est,truth=truth,before_failure=valid,condition=cond)
for name in ['foot_wrench_observer.py','audit_foot_wrench_observer.py']:(dest/name).write_text((ROOT/'scripts'/name).read_text())
print(json.dumps(records,indent=2))
