"""Single frozen CPU check of the inertial-load hypothesis on saved readouts."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
import time
START=time.monotonic()
import signal
def deadline(*_):raise TimeoutError('45_SECOND_DYNAMIC_OBSERVER_SCREEN')
signal.signal(signal.SIGALRM,deadline);signal.setitimer(signal.ITIMER_REAL,45)
import json
import hashlib
import sys
import resource
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation


def main():
    c=json.loads(Path(sys.argv[1]).read_text());out=Path(sys.argv[2]);out.mkdir(exist_ok=False)
    for p,h in c['sha256'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
    sys.path[:0]=[c['observer_directory'],c['package_directory']]
    from humanoid_sitting.shared_frame import BodyModel,level_rotation
    from dynamic_load import DynamicLoadObserver,shift_acceleration
    from observer import SpatialLoadObserver
    model=BodyModel(c['model']);inertia=json.loads(Path(c['inertia']).read_text())
    parameters=json.loads(Path(c['parameters']).read_text())
    new=DynamicLoadObserver(model,inertia,parameters['inertia_body'],parameters['armature'])
    old=SpatialLoadObserver(model,inertia);rows=[]
    for case in c['cases']:
        with np.load(case['trajectory']) as archive:
            z={key:archive[key] for key in archive.files}
        names=list(z['link_names']);root_i=names.index('pelvis')
        assert list(z['joint_names'])==model.names
        assert np.allclose(z['mass'].astype(float).sum(),sum(inertia['mass']),rtol=0,atol=1e-8)
        q0=z['q'][0];g0=np.array([0.,0.,-1.]);zero=np.zeros(len(q0));zero3=np.zeros(3)
        static,_,_,_=old.estimate(q0,z['tau'][0],g0)
        reduced,_=new.estimate_dynamic(q0,z['tau'][0],g0,zero,zero,zero3,zero3,zero3)
        assert np.max(abs(static-reduced))<1e-10,'STATIC_REDUCTION_FAILED'
        records=[]
        for k in range(3,len(z['q']),4):
            q,qd,tau=z['q'][k],z['qd'][k],z['tau'][k]
            qdd=(z['qd'][k].astype(float)-z['qd'][k-1])/.005
            pose=z['pose'][k];vel=z['velocity'][k].astype(float)
            acceleration=(vel-z['velocity'][k-1])/.005
            root=Rotation.from_quat(pose[root_i,3:7]).as_matrix()
            gravity=root.T@np.array([0.,0.,-1.]);level=level_rotation(gravity.copy())
            omega=vel[root_i,3:];alpha=acceleration[root_i,3:]
            r=root@z['com'][0,root_i,:3]
            a_origin=shift_acceleration(acceleration[root_i,:3],omega,alpha,-r)
            # Offline ideal-IMU synthesis. Root velocities/contact truth never
            # enter the candidate observer API. This does not certify a live
            # accelerometer implementation or replace native joint speeds.
            dynamic,detail=new.estimate_dynamic(q,tau,gravity,qd,qdd,root.T@omega,root.T@alpha,root.T@a_origin)
            baseline,_,_,_=old.estimate(q,tau,gravity)
            assert dynamic is not None and baseline is not None
            normal=level@root.T@np.asarray(case['normal'])
            force=z['normal'][k]+z['friction'][k]
            label=0. if case['kind']=='standing' else float(force[:,1].sum(0)@case['normal']/new.weight)
            a_pred=np.stack([detail['com_acceleration'][name] for name in names])
            a_true=acceleration[:,:3]@(root@level.T)
            accel_error=a_pred-a_true
            total_accel_error=np.sum(z['mass'][0,:,None]*accel_error,axis=0)
            records.append([baseline[:3]@normal/new.weight,dynamic[:3]@normal/new.weight,label,
                np.linalg.norm(total_accel_error)/new.weight,np.linalg.norm(accel_error,axis=1).max(),
                abs(qd[:23]).max(),abs((root.T@omega)[1])])
        a=np.asarray(records);mask=(a[:,5]<=.05)&(a[:,6]<=.02)
        old_err=abs(a[:,0]-a[:,2]);new_err=abs(a[:,1]-a[:,2])
        row=dict(case=case['name'],samples=len(a),raw_quiet=int(mask.sum()),static_reduction_passed=True,
            old_max_error_bw=float(old_err.max()),dynamic_max_error_bw=float(new_err.max()),
            old_mean_error_bw=float(old_err.mean()),dynamic_mean_error_bw=float(new_err.mean()),
            momentum_acceleration_error_max_bw=float(a[:,3].max()),
            individual_COM_acceleration_error_max_m_s2=float(a[:,4].max()),
            acceleration_relation_passed=bool(a[:,3].max()<=.001))
        rows.append(row);np.savez_compressed(out/(case['name']+'.npz'),rows=a)
    # Frozen mechanism screen, not a change to any physical success criterion.
    useful=bool(rows[0]['dynamic_max_error_bw']<=.7*rows[0]['old_max_error_bw']
                and rows[1]['dynamic_max_error_bw']<=.001
                and all(r['acceleration_relation_passed'] for r in rows))
    result=dict(cases=rows,mechanism_screen_passed=useful,uncertainty_admitted=False,
        physical_success=False,release_sensor_admitted=False,native_speeds_unchanged=True,
        simulation_steps=0,GPU_contexts=0,complete_cycles=0,
        input_scope='Saved native data synthesizes ideal pelvis IMU for offline mechanism test only',
        wall_s=time.monotonic()-START,process_CPU_s=time.process_time(),
        peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':
    try:main()
    finally:signal.setitimer(signal.ITIMER_REAL,0)
