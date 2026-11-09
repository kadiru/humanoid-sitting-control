"""Independent virtual-work/decision audit; no producer or optimizer imports."""
import os,time,signal,resource,json,sys,hashlib
from pathlib import Path
START=time.monotonic()
os.environ.update(CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
resource.setrlimit(resource.RLIMIT_AS,(40*1024**3,40*1024**3))
def deadline(*_):raise TimeoutError('45_SECOND_TOTAL_LOAD_AUDIT')
signal.signal(signal.SIGALRM,deadline);signal.setitimer(signal.ITIMER_REAL,45)
R=Path('/home/kadir/humanoid-sitting');H=Path(__file__).resolve().parent;O=H/'validation'
sys.path.insert(0,str(R/'sitting_task/src'))

def main():
    import numpy as np
    from scipy.spatial.transform import Rotation
    from humanoid_sitting.shared_frame import BodyModel,level_rotation,apply
    for p,h in json.loads((H/'audit_v2_freeze.json').read_text())['files'].items():
        assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
    assert not (O/'audit_v2.json').exists()
    z=dict(np.load(O/'predictions.npz'));r=json.loads((O/'result.json').read_text())
    model=BodyModel(R/'runs/g1_kinematics_geometry_04/model.json')
    inertia=json.loads(Path('/data/humanoid-sitting-perception/crouch_rest_bridge_01/balanced_standing10_01/static_inertial_model.json').read_text())
    grav_errors=[];jac_errors=[]
    for t in (0,500,998):
        q=z['q'][t].astype(float);rot=level_rotation(z['gravity'][t].astype(float).copy());fk=model.fk(q)
        for side_idx,side in enumerate(('left','right')):
            body=side+'_ankle_roll_link'
            for k,j in enumerate(('hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch','ankle_roll')):
                idx=model.names.index(side+'_'+j+'_joint');poses=[];potential=[]
                for sign in (-1,1):
                    v=q.copy();v[idx]+=sign*1e-5;f=model.fk(v)
                    potential.append(sum(m*9.81*(rot@apply(f[b],c))[2]
                      for b,m,c in zip(inertia['body_names'],inertia['mass'],inertia['local_com'])))
                    poses.append(f[body])
                translation=rot@(poses[1][:3,3]-poses[0][:3,3])/2e-5
                angular=rot@Rotation.from_matrix(poses[1][:3,:3]@poses[0][:3,:3].T).as_rotvec()/2e-5
                jac_errors.append(float(abs(np.r_[translation,angular]-z['jacobian'][t,side_idx,k]).max()))
                g=(potential[1]-potential[0])/2e-5
                grav_errors.append(abs(g-z['rhs'][t,side_idx,k]-z['torque'][t,idx]))
    a=z['admissible'];y=z['load_label'];lo=z['lower'];hi=z['upper']
    loaded=a&(lo>=.05);unloaded=a&(hi<.01)
    checks=dict(virtual_work_gravity=max(grav_errors)<2e-6,virtual_work_jacobian=max(jac_errors)<2e-6,
      loaded_bits=bool(np.array_equal(loaded,z['loaded'])),unloaded_bits=bool(np.array_equal(unloaded,z['unloaded'])),
      false_loaded=int((loaded&(y<.05)).sum())==r['false_loaded'],
      false_unloaded=int((unloaded&(y>=.01)).sum())==r['false_unloaded'],
      interval_misses=int((a&((y<lo)|(y>hi))).sum())==r['interval_misses'],
      not_deployed=r['actor_admitted'] is False,finite=all(np.isfinite(v).all() for v in z.values()))
    value=dict(passed=all(checks.values()),checks=checks,max_gravity_error_Nm=max(grav_errors),max_jacobian_error=max(jac_errors),
       scope='Accounting and equation audit only; does not override rejected load sensing or physical trial labels',elapsed_seconds=time.monotonic()-START)
    value['checks']={k:bool(v) for k,v in value['checks'].items()}
    value['max_gravity_error_Nm']=float(value['max_gravity_error_Nm'])
    (O/'audit_v2.json').write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value));assert value['passed']

if __name__=='__main__':
    try:main()
    finally:signal.setitimer(signal.ITIMER_REAL,0)
