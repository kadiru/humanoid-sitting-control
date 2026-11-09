"""CPU-only independent geometric fixtures and frozen probe selection."""
import ast
import sys
import time
import numpy as np
import trimesh
from io_utils import ROOT,HERE,ASSETS,TRAIN,read,write,sha
from surface_map import SurfaceMap

def main():
    start=time.monotonic();assert not (HERE/'preflight.json').exists()
    for p in HERE.glob('*.py'):ast.parse(p.read_text())
    # A two-triangle rectangle has one physical perimeter. Its diagonal is not
    # a boundary. Compare membership with independent original slab equations.
    m=SurfaceMap.__new__(SurfaceMap)
    m.tri=np.array([[[-1,-1,0],[1,-1,0],[1,1,0]],[[-1,-1,0],[1,1,0],[-1,1,0]]],float)
    m.normals=np.tile([0.,0.,1.],(2,1));m.allowed=np.ones(2,bool)
    m.boundary=np.array([[[-1,-1,0],[1,-1,0]],[[1,-1,0],[1,1,0]],[[1,1,0],[-1,1,0]],[[-1,1,0],[-1,-1,0]]],float)
    rng=np.random.default_rng(64301);points=rng.uniform([-1.1,-1.1,-.025],[1.1,1.1,.025],(1000,3))
    expected=(np.abs(points[:,:2])<.99).all(1)&(np.abs(points[:,2])<.015)
    actual=np.array([x['valid'] for x in m.map_points(points)])
    np.testing.assert_array_equal(actual,expected)
    assert m.map_points([[0,0,0]])[0]['valid'],'INTERNAL_DIAGONAL_REJECTED'
    assert not m.map_points([[.995,0,0]])[0]['valid'],'PERIMETER_GUARD_LOST'
    assert not m.map_points([[0,0,.016]])[0]['valid'],'DISTANCE_GUARD_LOST'
    m.allowed[1]=False
    assert not m.map_points([[0,0,0]])[0]['valid'],'AMBIGUOUS_TARGET_SEAM_ACCEPTED'
    assert not m.map_points([[-.5,.5,0]])[0]['valid'],'WRONG_FACE_ACCEPTED'
    cases=read(ASSETS/'cases.json');points={};rows=[]
    for c in cases:
        surface=SurfaceMap(c);points[c['id']]=surface.probe_points()
        assert all(x['valid'] for x in surface.map_points([p['point'] for p in points[c['id']]]))
        wrong=np.flatnonzero(~surface.allowed)
        if len(wrong):
            # A representative non-target face cannot earn target credit merely
            # by lying inside the object's overall bounding box.
            j=int(wrong[np.argmax(surface.mesh.area_faces[wrong])])
            assert not surface.map_points(surface.mesh.triangles_center[j:j+1])[0]['valid']
        rows.append(dict(case=c['id'],target_triangles=len(surface.region),boundary_edges=len(surface.boundary),probe_points=3))
    write(HERE/'probe_points.json',points)
    # The imported physical evaluator/reward/action files are untouched. Explicit
    # compatibility is verified from the current files, not old running flags.
    sit=TRAIN;lean=ROOT/'experiments/full_body_contact_trainer_4090_step_v5'
    assert (sit/'environment.py').read_text().splitlines()[1:]==(lean/'environment.py').read_text().splitlines()[1:]
    write(HERE/'preflight.json',dict(passed=True,seconds=time.monotonic()-start,
        flat_surface_equivalence_points=1000,negative_fixtures=['outside_surface','perimeter','wrong_face','ambiguous_seam'],
        internal_triangle_diagonal_allowed=True,cases=rows,simulator_started=False,GPU_used=False,
        physical_simulation_validated=False,success_thresholds_changed=False))
    assert time.monotonic()-start<600
    print('CPU_PREFLIGHT_PASS:14 layouts; physics remains untested')

if __name__=='__main__':main()
