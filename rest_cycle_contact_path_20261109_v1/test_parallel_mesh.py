"""Compare the whole serial and partitioned reports on synthetic geometry."""
import itertools,os,unittest,time,multiprocessing
import numpy as np
from scipy.spatial.transform import Rotation
from audit_mesh_clearance import audit as serial_audit
from parallel_mesh import audit


class Model:
    def fk(self,q):
        torso=np.eye(4);torso[0,3]=.04+q[0]
        return dict(pelvis=np.eye(4),torso_link=torso)


class BrokenModel:
    def fk(self,q):raise ValueError('SYNTHETIC_WORKER_FAILURE')


class StalledModel:
    def fk(self,q):time.sleep(10);return Model().fk(q)


def example(clear=False,n=17):
    cube=list(itertools.product([-.1,.1],repeat=3))
    shapes=[dict(body=b,vertices=cube) for b in ('pelvis','torso_link')]
    furniture=dict(vertices=np.array([[-1,-1,0],[1,-1,0],[1,1,0],[-1,1,0]]),faces=np.array([[0,1,2],[0,2,3]]))
    nodes=[]
    for i in range(n):
        root=np.eye(4);root[:3,:3]=Rotation.from_euler('xyz',[i*.002,i*.003,-i*.001]).as_matrix()
        root[:3,3]=[i*.002,0.,1. if clear else .05]
        nodes.append(dict(q=[i*.0001]*37,root=root.tolist(),phase='landing' if i%3 else 'approach'))
    return (shapes,furniture,[0],nodes,np.eye(3),np.zeros(3))


class ParallelTests(unittest.TestCase):
    def setUp(self):self.before=os.environ.get('CUDA_VISIBLE_DEVICES');os.environ['CUDA_VISIBLE_DEVICES']=''
    def tearDown(self):
        if self.before is None:os.environ.pop('CUDA_VISIBLE_DEVICES',None)
        else:os.environ['CUDA_VISIBLE_DEVICES']=self.before
    def test_exact_report_for_mixed_contacts_and_truncated_examples(self):
        args=example(n=65)
        expected=serial_audit(Model(),*args)
        actual=audit(Model(),*args,workers=2,max_wall_s=10)
        self.assertGreater(expected['forbidden_or_early_clearance_intersection_count'],50)
        self.assertEqual(actual,expected)
    def test_exact_report_for_clear_geometry_and_uneven_partition(self):
        args=example(clear=True,n=18)
        self.assertEqual(audit(Model(),*args,workers=3,max_wall_s=10),serial_audit(Model(),*args))
    def test_single_segment_is_not_duplicated(self):
        args=example(n=2)
        self.assertEqual(audit(Model(),*args,workers=8,max_wall_s=10),serial_audit(Model(),*args))
    def test_full_eight_worker_partition_preserves_the_report(self):
        args=example(n=19)
        self.assertEqual(audit(Model(),*args,workers=8,max_wall_s=10),serial_audit(Model(),*args))
    def test_worker_failure_is_not_a_pass(self):
        with self.assertRaisesRegex(RuntimeError,'SYNTHETIC_WORKER_FAILURE'):
            audit(BrokenModel(),*example(),workers=2,max_wall_s=10)
    def test_no_unbounded_worker_or_time_request(self):
        for kwargs in (dict(workers=0),dict(workers=9),dict(max_wall_s=0)):
            with self.assertRaisesRegex(ValueError,'FINITE_PARALLEL'):
                audit(Model(),*example(),**kwargs)
    def test_stalled_workers_are_stopped_within_the_finite_return_budget(self):
        before={p.pid for p in multiprocessing.active_children()};start=time.monotonic()
        with self.assertRaisesRegex(TimeoutError,'PARALLEL_MESH_WORK_BUDGET'):
            audit(StalledModel(),*example(),workers=2,max_wall_s=1.2)
        self.assertLess(time.monotonic()-start,1.5)
        self.assertFalse({p.pid for p in multiprocessing.active_children()}-before)


if __name__=='__main__':unittest.main()
