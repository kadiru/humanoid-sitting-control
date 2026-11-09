"""Independent correctness fixtures for the exact collision-audit accelerator."""
import json,sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent.parent
sys.path[1:1]=[str(ROOT/'experiments/furniture_repair_20261106_v1'),str(ROOT/'sitting_task/src')]
import audit_mesh_clearance as reference
import audit_mesh_clearance_fast as candidate


def main():
    rng=np.random.default_rng(66380)
    vertices=np.array([[x,y,z] for x in [-.1,.1] for y in [-.1,.1] for z in [-.1,.1]])
    shapes=[dict(body='pelvis',vertices=vertices.tolist())]
    class Model:
        def fk(self,q):return {'pelvis':np.eye(4)}
    model=Model();compared=0;collisions=0
    for phase in ['approach','landing']:
        for sample in range(12):
            triangles=rng.uniform(-.25,.25,(30,3,3))
            mesh={'vertices':triangles.reshape(-1,3),'faces':np.arange(90).reshape(-1,3)}
            rt=np.eye(4);rt[:3,3]=rng.uniform(-.03,.03,3)
            nodes=[dict(q=[0],root=np.eye(4).tolist(),phase=phase),dict(q=[0],root=rt.tolist(),phase=phase)]
            args=(model,shapes,mesh,list(range(15)),nodes,np.eye(3),np.zeros(3))
            a=reference.audit(*args);b=candidate.audit(*args)
            for key in a:
                assert a[key]==b[key],(key,a[key],b[key])
            collisions+=a['forbidden_or_early_clearance_intersection_count'];compared+=1
    assert collisions>0
    with np.load(ROOT/'experiments/furniture_examples_20261105_v1/scenes/wooden_stool_01_top/mesh.npz') as z:
        tri=z['vertices'][z['faces']]
    lo,hi=tri.min(1),tri.max(1);centers=(lo+hi)/2
    radius=np.linalg.norm((hi-lo)/2,axis=1).max();tree=cKDTree(centers)
    for _ in range(200):
        center=rng.uniform(lo.min(0)-.1,hi.max(0)+.1);half=rng.uniform(.001,.2,3)
        a,b=center-half,center+half
        expected=np.flatnonzero(np.all(hi>=a,axis=1)&np.all(lo<=b,axis=1))
        ids=np.array(tree.query_ball_point(center,np.linalg.norm(half)+radius+1e-10),int)
        actual=np.sort(ids[np.all(hi[ids]>=a,axis=1)&np.all(lo[ids]<=b,axis=1)])
        np.testing.assert_array_equal(actual,expected)
    result=dict(passed=True,exact_full_auditor_fixture_comparisons=compared,
                dense_mesh_AABB_comparisons=200,physics_steps=0,PPO_updates=0,
                physical_success_claim=False,clearance_changed=False)
    (HERE/'fast_audit_software_validation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result))


if __name__=='__main__':main()
