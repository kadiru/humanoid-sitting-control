"""Frozen preflight: serializer regression and conditional camera separation."""
import json,sys,time,os,resource
from pathlib import Path
import numpy as np
ROOT=Path('/home/kadir/humanoid-sitting');HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'sitting_task/src'))
from humanoid_sitting.shared_frame import BodyModel
from separation import shadow_separation,REST_BODIES
from recording import write

def main():
    os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
    resource.setrlimit(resource.RLIMIT_AS,(40*1024**3,40*1024**3))
    start=time.monotonic();model=BodyModel(ROOT/'runs/g1_kinematics_geometry_04/model.json')
    case=next(c for c in json.loads((ROOT/'experiments/furniture_examples_20261105_v1/cases.json').read_text()) if c['id']=='modern_coffee_table_01_top')
    with np.load(case['mesh_npz']) as z:vertices=z['vertices'].copy()
    rows=[]
    for suffix in ['center','angled']:
        p=ROOT/f'outputs/furniture_load_repair_20261106_v3/calibration_demos/sit_modern_coffee_table_01_{suffix}_normal_landing/frame_0.npz'
        with np.load(p) as z:
            q=z['q'];g=z['geometry'];r=shadow_separation(model,q,g)
            # True pose / full mesh used only in this independent domain audit.
            normal=z['root_rotation']@g[2:5];offset=float(g[5]-normal@z['root_position'])
            protrusion=float(np.max(vertices@normal+offset))
            missing=g.copy();missing[11]=0
            stale=g.copy();stale[12]=.22
            # Lift the plane until it intersects every legal rest body's domain.
            shifted=g.copy();shifted[5]-=.5
            negatives=dict(missing=not shadow_separation(model,q,missing)['candidate'],
                           stale=not shadow_separation(model,q,stale)['candidate'],
                           overlap=not shadow_separation(model,q,shifted)['candidate'])
            rows.append(dict(start=suffix,shadow=r,complete_fixture_halfspace_protrusion_m=protrusion,
                             fixture_halfspace_pass=protrusion<=.003,negative_checks=negatives))
    result=dict(rows=rows,passed=all(r['fixture_halfspace_pass'] and all(r['negative_checks'].values()) for r in rows),
                separation_ready_for_shadow_collection=True,control_admitted=False,
                loaded_signal_validated=False,standup_validated=False,new_physics_steps=0,
                seconds=time.monotonic()-start)
    write(HERE/'preflight.json',result)
    print(json.dumps(result))
    assert result['passed']

if __name__=='__main__':main()
