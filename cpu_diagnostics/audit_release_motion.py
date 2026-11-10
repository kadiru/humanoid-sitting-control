"""Frozen, independent hold/unloading sensor validation from CPU recordings.

No simulator, controller, path fitting, or online contact oracle is imported.
An empirical error interval is tested, not asserted to be universally valid.
"""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
import time
START=time.monotonic()
import signal
def timeout(*_):raise TimeoutError('45_SECOND_RELEASE_AUDIT')
signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,45)
import json,hashlib,sys,resource
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation


def longest(mask):
    best=run=0
    for value in mask:
        run=run+1 if value else 0;best=max(best,run)
    return best


def main():
    c=json.loads(Path(sys.argv[1]).read_text());out=Path(sys.argv[2]);out.mkdir(exist_ok=False)
    for p,h in c['sha256'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
    sys.path[:0]=[c['observer_directory'],c['package_directory']]
    from humanoid_sitting.shared_frame import BodyModel,level_rotation
    from dynamic_load import DynamicLoadObserver,shift_acceleration
    from observer import SpatialLoadObserver
    model=BodyModel(c['model']);inertia=json.loads(Path(c['inertia']).read_text())
    params=json.loads(Path(c['parameters']).read_text())
    candidate=DynamicLoadObserver(model,inertia,params['inertia_body'],params['armature'])
    old=SpatialLoadObserver(model,inertia)
    # NpzFile indexing decompresses the entire named array on every access.
    # Materialize each channel once before the per-sample arithmetic.
    with np.load(c['trajectory']) as archive:
        z={key:archive[key] for key in archive.files}
    names=list(z['link_names']);root_i=names.index('pelvis')
    assert list(z['joint_names'])==model.names
    support_n=np.asarray(c['normal']);rows=[]
    leg_bodies=set().union(*(candidate.desc[side[0]] for side in candidate.sides))
    above_feet=[names.index(n) for n in leg_bodies if 'ankle_roll_link' not in n]
    for k in range(3,len(z['q']),4):
        q,qd,tau=z['q'][k],z['qd'][k],z['tau'][k]
        qdd=(z['qd'][k].astype(float)-z['qd'][k-1])/.005
        vel=z['velocity'][k].astype(float);acc=(vel-z['velocity'][k-1])/.005
        root=Rotation.from_quat(z['pose'][k,root_i,3:7]).as_matrix()
        gravity=root.T@np.array([0.,0.,-1.]);level=level_rotation(gravity.copy())
        omega=vel[root_i,3:];alpha=acc[root_i,3:]
        r=root@z['com'][0,root_i,:3]
        origin_a=shift_acceleration(acc[root_i,:3],omega,alpha,-r)
        new,detail=candidate.estimate_dynamic(q,tau,gravity,qd,qdd,root.T@omega,root.T@alpha,root.T@origin_a)
        baseline,_,_,_=old.estimate(q,tau,gravity)
        assert new is not None and baseline is not None,'OBSERVER_UNAVAILABLE'
        normal=level@root.T@support_n
        force=z['normal'][k]+z['friction'][k]
        truth=float(force[:,1].sum(0)@support_n/candidate.weight)
        a_pred=np.stack([detail['com_acceleration'][name] for name in names])
        a_true=acc[:,:3]@(root@level.T)
        residual=np.sum(z['mass'][0,:,None]*(a_pred-a_true),axis=0)
        extra_leg_force=np.linalg.norm(force[above_feet,1],axis=-1).sum()
        rows.append([baseline[:3]@normal/candidate.weight,new[:3]@normal/candidate.weight,
            truth,np.linalg.norm(residual)/candidate.weight,abs(qd[:23]).max(),
            abs((root.T@omega)[1]),extra_leg_force,z['pose'][k,root_i,2]])
        if len(rows)%50==0:
            np.save(out/'completed_rows.npy',np.asarray(rows))
            (out/'progress.json').write_text(json.dumps(dict(completed_samples=len(rows),
                last_physics_index=k,wall_s=time.monotonic()-START)))
    a=np.asarray(rows);quiet=(a[:,4]<=.05)&(a[:,5]<=.02)
    error=abs(a[:,1]-a[:,2]);old_error=abs(a[:,0]-a[:,2]);radius=c['frozen_release_radius_bw']
    # Clipping the lower endpoint is the known unilateral normal-load domain;
    # it must not turn a fully negative interval into an admitted release.
    lo=np.maximum(0,a[:,1]-radius);hi=a[:,1]+radius
    released=quiet&(0<=lo)&(lo<=hi)&(hi<.01)
    loaded=quiet&(a[:,1]-.02>=.05)
    misses=quiet&((a[:,2]<lo)|(a[:,2]>hi))
    false_release=released&(a[:,2]>=.01)
    results={}
    for phase,mask in [('all',np.ones(len(a),bool)),('quiet',quiet),
                       ('unloading',np.arange(len(a))>=250),
                       ('unloaded_quiet',quiet&(a[:,2]<.01))]:
        results[phase]=dict(samples=int(mask.sum()),dynamic_max_error_bw=float(error[mask].max()) if mask.any() else None,
            static_max_error_bw=float(old_error[mask].max()) if mask.any() else None)
    progress=json.loads(Path(c['progress']).read_text())
    tests=dict(full_stimulus=len(a)==1250 and 'abort' not in progress,
        quiet_accuracy=bool(quiet.any() and error[quiet].max()<=.001),
        acceleration_relation=bool(a[:,3].max()<=.001),
        interval_coverage=not misses.any(),no_false_release=not false_release.any(),
        released_25_consecutive=longest(released)>=25,
        loaded_25_consecutive=longest(loaded)>=25,
        no_unmodeled_leg_contact=bool(a[:,6].max()<=1.))
    result=dict(passed=bool(all(tests.values())),tests={k:bool(v) for k,v in tests.items()},errors=results,
        frozen_release_radius_bw=radius,interval_misses=int(misses.sum()),false_releases=int(false_release.sum()),
        longest_released_quiet_s=.02*longest(released),longest_loaded_quiet_s=.02*longest(loaded),
        maximum_momentum_acceleration_error_bw=float(a[:,3].max()),
        maximum_unmodeled_leg_contact_N=float(a[:,6].max()),
        final_root_height_m=float(a[-1,7]),
        scope='Offline ideal-IMU empirical sensor validation; no live IMU/current-camera admission',
        release_sensor_admitted=False,physical_success=False,complete_cycles=0,
        physics_steps=0,GPU_contexts=0,wall_s=time.monotonic()-START,
        process_CPU_s=time.process_time(),peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    np.savez_compressed(out/'predictions.npz',rows=a,quiet=quiet,released=released,interval_misses=misses)
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':
    try:main()
    finally:signal.setitimer(signal.ITIMER_REAL,0)
