"""Independent saved CPU readout audit of the unchanged spatial observer.

Native contact forces supply labels only. This does not fit an uncertainty
band, modify controller inputs, or admit a sensor from two static postures.
"""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
import hashlib
import json
import signal
import sys
import time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation


def main():
    start=time.monotonic()
    def timeout(*_): raise TimeoutError('45_SECOND_CPU_LOAD_AUDIT')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(45)
    c=json.loads(Path(sys.argv[1]).read_text());out=Path(sys.argv[2]);out.mkdir(exist_ok=False)
    for p,h in c['sha256'].items():
        assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
    sys.path[:0]=[c['observer_directory'],c['package_directory']]
    from observer import SpatialLoadObserver
    from humanoid_sitting.shared_frame import BodyModel,level_rotation
    model=BodyModel(c['model'])
    observer=SpatialLoadObserver(model,json.loads(Path(c['inertia']).read_text()))
    rows=[]
    for case in c['cases']:
        z=np.load(case['trajectory']);n=np.asarray(case['normal'])
        assert list(z['joint_names'])==model.names
        body=list(z['link_names']).index('pelvis')
        selected=np.arange(3,len(z['q']),4)
        predictions=[];labels=[];quiet=[];conditions=[];foot_errors=[]
        for index in selected:
            q=z['q'][index];qd=z['qd'][index];tau=z['tau'][index]
            root=Rotation.from_quat(z['pose'][index,body,3:7]).as_matrix()
            gravity=root.T@np.array([0.,0.,-1.]);level=level_rotation(gravity.copy())
            gyro=root.T@z['velocity'][index,body,3:6]
            value,condition,_,_=observer.estimate(q,tau,gravity)
            normal_level=level@root.T@n
            pred=np.nan if value is None else float(value[:3]@normal_level/observer.weight)
            force=z['normal'][index]+z['friction'][index]
            if force.ndim==2:
                assert case['kind']=='standing'
                label=0.;floor=force
            else:
                label=float(force[:,1].sum(0)@n/observer.weight);floor=force[:,0]
            feet=[list(z['link_names']).index(s+'_ankle_roll_link') for s in ['left','right']]
            foot_true=floor[feet]@(root@level.T)
            foot_error=np.nan if value is None else float(np.linalg.norm(
                value[3:].reshape(2,6)[:,:3]-foot_true,axis=1).max()/observer.weight)
            quiet.append(bool(abs(qd[:23]).max()<=.05 and abs(gyro[1])<=.02 and not z['sleeping'][index]))
            predictions.append(pred);labels.append(label);conditions.append(condition);foot_errors.append(foot_error)
        pred=np.asarray(predictions);label=np.asarray(labels);mask=np.asarray(quiet)&np.isfinite(pred)
        err=abs(pred-label)
        row=dict(case=case['name'],controls=len(selected),quiet_controls=int(mask.sum()),
            max_abs_error_bw=float(err.max()),mean_abs_error_bw=float(err.mean()),
            last_second_max_abs_error_bw=float(err[-50:].max()),
            qualified_max_abs_error_bw=float(err[mask].max()) if mask.any() else None,
            mean_load_label_bw=float(label.mean()),mean_load_estimate_bw=float(pred.mean()),
            max_condition=float(np.max(conditions)),
            existing_radius_bw=.02,release_upper_threshold_bw=.01,
            release_sensor_admitted=False)
        rows.append(row)
        np.savez_compressed(out/(case['name']+'.npz'),prediction=pred,label=label,
            raw_quiet=mask,condition=conditions,foot_error=foot_errors)
    result=dict(cases=rows,observer_changed=False,uncertainty_fitted=False,
        release_sensor_admitted=False,complete_cycles=0,GPU_contexts=0,
        wall_s=time.monotonic()-start)
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
    signal.alarm(0)


if __name__=='__main__':main()
