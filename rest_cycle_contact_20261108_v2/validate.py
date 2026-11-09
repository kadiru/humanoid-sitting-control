"""New top/front witness on the two preserved fresh standing resets."""
import json,sys,time,os,resource
from pathlib import Path
import numpy as np
ROOT=Path('/home/kadir/humanoid-sitting');HERE=Path(__file__).resolve().parent
sys.path[1:1]=[str(ROOT/'sitting_task/src'),str(ROOT/'experiments/furniture_perception_repair_20261029_v1'),str(ROOT/'experiments/rest_cycle_contact_20261108_v1')]
from humanoid_sitting.shared_frame import BodyModel
from separation import witness
from recording import write

def main():
    start=time.monotonic();os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
    resource.setrlimit(resource.RLIMIT_AS,(40*1024**3,40*1024**3))
    model=BodyModel(ROOT/'runs/g1_kinematics_geometry_04/model.json')
    case=next(c for c in json.loads((ROOT/'experiments/furniture_examples_20261105_v1/cases.json').read_text()) if c['id']=='modern_coffee_table_01_top')
    with np.load(case['mesh_npz']) as z:vertices=z['vertices'].copy()
    rows=[]
    for suffix in ['center','angled']:
        path=ROOT/f'outputs/furniture_load_repair_20261106_v3/calibration_demos/sit_modern_coffee_table_01_{suffix}_normal_landing/frame_0.npz'
        with np.load(path) as z:
            q=z['q'];g=z['geometry'];p=z['selected_points'];K=z['K']
            # The recorded selected patch has exact source pixel correspondences.
            pixels=z['selected_pixels'].astype(int);depth=z['depth'][pixels[:,0],pixels[:,1]]
            r=witness(model,q,g,p,K,float(depth.max()))
            N=np.array(r['envelope_normals']);D=np.array(r['envelope_offsets'])
            Nw=N@z['root_rotation'].T;Dw=D-Nw@z['root_position']
            protrusion=np.max(vertices@Nw.T+Dw,axis=0)
            stale=g.copy();stale[12]=.22
            negative=dict(stale=not witness(model,q,stale,p,K,float(depth.max()))['candidate'],
                          missing=not witness(model,q,g,[],K,float(depth.max()))['candidate'])
            rows.append(dict(start=suffix,shadow=r,complete_fixture_protrusion_m=protrusion.tolist(),
                             fixture_enclosure_pass=bool((protrusion<=.003).all()),negative_checks=negative))
    result=dict(rows=rows,passed=all(r['shadow']['candidate'] and r['fixture_enclosure_pass'] and all(r['negative_checks'].values()) for r in rows),
                control_admitted=False,live_shape_envelope_certified=False,physics_steps=0,seconds=time.monotonic()-start)
    write(HERE/'preflight.json',result);print(json.dumps(result));assert result['passed']

if __name__=='__main__':main()
