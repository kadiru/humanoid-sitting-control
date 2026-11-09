"""Saved-only new clipped-corner representation; retain rejected line evidence."""
import os,time,signal,resource,sys,json,hashlib,traceback
from pathlib import Path
START=time.monotonic();os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});resource.setrlimit(resource.RLIMIT_AS,(40*1024**3,40*1024**3))
def timeout(*a):raise TimeoutError('45_SECOND_TOTAL_OBSERVED_POLYGON_AUDIT_CAP')
signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,45)
R=Path('/home/kadir/humanoid-sitting');H=Path(__file__).resolve().parent;O=R/'outputs'/H.name/'camera_and_wrench_capture'
sys.path.insert(1,str(R/'sitting_task/src'))
def write(name,x):(O/name).write_text(json.dumps(x,indent=2)+'\n')
def main():
    import numpy as np
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    from humanoid_sitting.shared_frame import level_rotation
    from observed_patch_v2 import build
    for p,h in json.loads((H/'patch_v2_freeze.json').read_text())['files'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
    prior=json.loads((O/'capture_audit.json').read_text());case=json.loads((O/'manifest.json').read_text())['case']
    with np.load(case['mesh_npz']) as z:fv,faces=z['vertices'],z['faces']
    allowed=json.loads((H/'assets/scenes/corner_support_table_050/target_region.json').read_text())['triangle_indices']
    rows=[]
    for tick in [0,50]:
        source=O/f'frame_{tick}.npz';dep=next(r['depth'] for r in prior['frames'] if r['tick']==tick)
        assert hashlib.sha256(source.read_bytes()).hexdigest()==dep['input_sha256']
        assert dep['geometry_pass'] and dep['original_plane_gate'] and dep['target_observable']
        with np.load(source) as z:frame=dict(z)
        level=level_rotation(frame['gravity'].astype(float).copy());n=level@frame['geometry'][2:5];points=frame['selected_points']@level.T
        T=np.asarray(json.loads((H.parent/'rest_cycle_partial_support_20261109_v1/camera_preflight.json').read_text())['camera_in_body'])
        patch,mask=build(frame,T,level);write(f'observed_patch_v2_{tick}.json',patch);np.savez_compressed(O/f'patch_mask_v2_{tick}.npz',mask=mask)
        # Independent polygon/actual allowed-triangle union coverage. No use of
        # producer halfspaces or estimated corner lines in this audit.
        basis=frame['root_rotation']@level.T;verts=(fv-frame['root_position'])@basis
        e,b=np.asarray(patch['e']),np.asarray(patch['b']);uv=np.c_[verts@e,verts@b]
        parts=[Polygon(uv[faces[i]]) for i in allowed if Polygon(uv[faces[i]]).area>1e-14]
        support=unary_union(parts);candidate=Polygon(patch['vertices_uv'],patch['holes_uv']);outside=candidate.difference(support)
        normal_distance=float(np.max(abs(points@n+frame['geometry'][5])))
        passed=bool(outside.area<=1e-10 and candidate.is_valid and candidate.area>=.002)
        rows.append(dict(tick=tick,passed=passed,outside_allowed_triangles_m2=float(outside.area),
            observed_safe_area_m2=float(candidate.area),measured_plane_max_residual_m=normal_distance,
            vertices=len(patch['vertices_uv']),depth_audit_inherited_by_hash=True))
    result=dict(passed=all(r['passed'] for r in rows),frames=rows,elapsed_seconds=time.monotonic()-START,
        correction='Fixed observed-region representation: original selected cloud includes5mm band and convex hull bridges grooves. Current3mmplane and5deg local normals produce a nonconvex polygon with holes.6mmerosion and full independent allowedtriangle coverage retained. No new camera/raycast.',
        numerical_geometry='GEOS/float64 triangle-union coverage; not an exact rational certificate',new_physics_steps=0,PPO_updates=0)
    write('patch_audit_v2.json',result);print(json.dumps(result))
if __name__=='__main__':
    try:main()
    except BaseException as e:write('patch_failure_v2.json',dict(error=repr(e),traceback=traceback.format_exc(),elapsed_seconds=time.monotonic()-START));raise
    finally:signal.setitimer(signal.ITIMER_REAL,0)
