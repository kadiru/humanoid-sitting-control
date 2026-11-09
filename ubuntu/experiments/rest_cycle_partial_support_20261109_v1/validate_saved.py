"""One bounded saved-camera association and release test. No simulator imports."""
import os,time,signal,resource,sys,json,hashlib,unittest
from pathlib import Path
START=time.monotonic()
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
resource.setrlimit(resource.RLIMIT_AS,(40*1024**3,40*1024**3))
def timeout(*args):raise TimeoutError('45_SECOND_TOTAL_SAVED_VALIDATION_CAP')
signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,45)
R=Path('/home/kadir/humanoid-sitting');H=Path(__file__).resolve().parent
sys.path.insert(0,str(R/'sitting_task/src'))

def main():
    O=H/'saved_validation';O.mkdir(exist_ok=False)
    try:
        freeze=json.loads((H/'saved_freeze.json').read_text())
        for p,h in freeze['files'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
        import numpy as np
        from edge import detect
        from release import separation
        from humanoid_sitting.shared_frame import BodyModel,level_rotation,apply
        suite=unittest.defaultTestLoader.discover(str(H),pattern='test_components.py')
        with (O/'software_tests.txt').open('w') as f:
            result=unittest.TextTestRunner(stream=f).run(suite)
        assert result.wasSuccessful(),'SOFTWARE_TESTS_FAILED'
        src=Path(freeze['frame'])
        with np.load(src) as data:z=dict(data)
        T=np.asarray(freeze['camera_in_body']);g=z['geometry'];level=level_rotation(z['gravity'].astype(float).copy())
        # Current camera extrinsic is a calibration constant. Saved world poses
        # enter only the independent post-detection check, never detect().
        sensor,arrays=detect(z['depth'],z['K'],T,g[2:5],float(g[5]),level,z['selected_pixels'])
        (O/'edge_sensor.json').write_text(json.dumps(sensor,indent=2)+'\n')
        np.savez_compressed(O/'edge_arrays.npz',**arrays)
        audit=dict(passed=False,reason='sensor_edge_not_qualified')
        release=dict(state='unknown',load_upper_bw=None,reason='edge_not_qualified')
        if sensor['passed']:
            with np.load(freeze['fixture_mesh']) as f:fv=f['vertices']
            basis=z['root_rotation']@level.T
            verts=(fv-z['root_position'])@basis
            bound=sensor['front_x_interval_m']
            actual=float(verts[:,0].max())
            n=level@g[2:5]
            audit=dict(passed=bool(bound[0]-.006<=actual<=bound[1]+.006 and np.max(verts@n+g[5])<=.003),
                actual_front_x_m=actual,predicted_front_interval_m=bound,
                full_fixture_below_plane_max_m=float(np.max(verts@n+g[5])),truth_used_by_detector=False,
                scope='New finite-surface envelope/edge audit only; not a rerun of depth correspondence.')
            if audit['passed']:
                model=BodyModel(R/'runs/g1_kinematics_geometry_04/model.json');fk=model.fk(z['q'])
                shape=json.loads((R/'experiments/furniture_perception_repair_20261029_v1/render_colliders.json').read_text())['shapes']
                grouped={}
                for s in shape:grouped.setdefault(s['body'],[]).extend((apply(fk[s['body']],s['vertices'])@level.T).tolist())
                release=separation(vertices=grouped,plane_normal=n,plane_offset=float(g[5]),
                    front_upper=bound[1],front_error=.006,normal_error=.003,plane_center=level@g[6:9],
                    qualified_edge=True,geometry_age=float(g[12]))
        (O/'edge_independent_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
        (O/'release_standing.json').write_text(json.dumps(release,indent=2)+'\n')
        output=dict(software_tests=result.testsRun,software_passed=True,edge_passed=sensor['passed'],
            edge_audit_passed=audit['passed'],release_state=release['state'],
            source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
            loaded_release_validation=False,runtime_control_admitted=False,physical_success=False,
            new_physics_steps=0,new_solver_calls=0,PPO_updates=0,elapsed_seconds=time.monotonic()-START)
        (O/'summary.json').write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(output))
    except BaseException as e:
        (O/'failure.json').write_text(json.dumps(dict(error=repr(e),elapsed_seconds=time.monotonic()-START),indent=2)+'\n')
        raise
    finally:signal.setitimer(signal.ITIMER_REAL,0)

if __name__=='__main__':main()
