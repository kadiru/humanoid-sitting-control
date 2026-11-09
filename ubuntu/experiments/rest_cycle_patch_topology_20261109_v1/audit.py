"""Saved-image topology correction; preserve every camera/physical threshold."""
import os,time,signal,resource,sys,json,hashlib,traceback
from pathlib import Path
START=time.monotonic();os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});resource.setrlimit(resource.RLIMIT_AS,(40*1024**3,40*1024**3))
def timeout(*a):raise TimeoutError('45_SECOND_TOPOLOGY_VALIDATION_CAP')
signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,45)
R=Path('/home/kadir/humanoid-sitting');H=Path(__file__).resolve().parent;O=H/'validation';S=R/'outputs/rest_cycle_narrow_support_20261109_v1/camera_and_wrench_capture'
sys.path.insert(1,str(R/'sitting_task/src'))
def write(n,d):(O/n).write_text(json.dumps(d,indent=2)+'\n')
def main():
    O.mkdir(exist_ok=False)
    for p,h in json.loads((H/'freeze.json').read_text())['files'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
    import numpy as np,unittest
    from scipy.ndimage import label
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    from humanoid_sitting.shared_frame import level_rotation
    from observed_patch_v3 import build
    from test_topology import TopologyTests
    tests=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(TopologyTests));assert tests.wasSuccessful()
    T=np.asarray(json.loads((R/'experiments/rest_cycle_partial_support_20261109_v1/camera_preflight.json').read_text())['camera_in_body'])
    case=json.loads((S/'manifest.json').read_text())['case']
    with np.load(case['mesh_npz']) as z:fv,faces=z['vertices'],z['faces']
    allowed=json.loads(Path(case['mesh_npz']).with_name('target_region.json').read_text())['triangle_indices'];rows=[]
    for tick in [0,50]:
        with np.load(S/f'frame_{tick}.npz') as z:f=dict(z)
        with np.load(S/f'patch_mask_v2_{tick}.npz') as z:oldmask=z['mask']
        old=json.loads((S/f'observed_patch_v2_{tick}.json').read_text());level=level_rotation(f['gravity'].astype(float).copy())
        patch,mask=build(f,T,level);assert np.array_equal(mask,oldmask),'SENSOR_CLASSIFICATION_CHANGED'
        write(f'patch_{tick}.json',patch)
        basis=f['root_rotation']@level.T;verts=(fv-f['root_position'])@basis;e,b=np.asarray(patch['e']),np.asarray(patch['b']);uv=np.c_[verts@e,verts@b]
        support=unary_union([Polygon(uv[faces[i]]) for i in allowed]);poly=Polygon(patch['vertices_uv'],patch['holes_uv']);outside=float(poly.difference(support).area)
        count=label(mask)[1]
        passed=bool(outside<=1e-10 and poly.is_valid and poly.area>=.002 and count==patch['pixel_topology']['pixel_components'])
        rows.append(dict(tick=tick,passed=passed,identical_sensor_mask=True,independent_mask_connected_components=count,
            old_eligible_components=old['eligible_components'],new_eligible_components=patch['eligible_components'],
            old_selected_area_m2=old['area_m2'],new_selected_area_m2=patch['area_m2'],outside_allowed_triangles_m2=outside,
            frame_sha256=hashlib.sha256((S/f'frame_{tick}.npz').read_bytes()).hexdigest()))
    d=dict(passed=all(r['passed'] for r in rows),frames=rows,tests_passed=tests.testsRun,elapsed_seconds=time.monotonic()-START,new_physics_steps=0,PPO_updates=0,
        explanation='Samepixelclassification; integerpixelunion avoids separatelyprojectedrun roundoff cracks. Originalnormal/plane/erosion/meshcriteria unchanged. Newpolygoncoverage checkedindependently.')
    write('result.json',d);print(json.dumps(d))
if __name__=='__main__':
    try:main()
    except BaseException as e:write('failure.json',dict(error=repr(e),traceback=traceback.format_exc(),elapsed_seconds=time.monotonic()-START));raise
    finally:signal.setitimer(signal.ITIMER_REAL,0)
