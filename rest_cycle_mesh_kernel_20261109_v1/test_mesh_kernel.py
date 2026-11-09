"""Independent comparisons against the unchanged NumPy reference clipper."""
import itertools,json,os,time,unittest,multiprocessing
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from audit_mesh_clearance import audit as original_audit,clip_triangle
from mesh_audit import Clipper,audit
from parallel_audit import audit as parallel_audit


class Model:
    def fk(self,q):
        torso=np.eye(4);torso[0,3]=.04+q[0]
        return dict(pelvis=np.eye(4),torso_link=torso)


class BrokenModel:
    def fk(self,q):raise ValueError('SYNTHETIC_MODEL_FAILURE')


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


class KernelTests(unittest.TestCase):
    def setUp(self):
        self.library=os.environ['MESH_NUMPY_LIBRARY'];self.clipper=Clipper(self.library)
        self.box=ConvexHull(list(itertools.product([-1.,1.],repeat=3))).equations

    def compare(self,triangle,equations=None,expansion=.006):
        eq=self.box if equations is None else equations
        expected=clip_triangle(triangle,eq,expansion)
        actual=self.clipper.polygon(triangle,eq,expansion)
        np.testing.assert_array_equal(actual.view(np.uint64),expected.view(np.uint64))

    def test_clear_inside_crossing_and_degenerate(self):
        for t in [np.zeros((3,3)),np.full((3,3),4.),
                  np.array([[-3,0,0],[3,0,0],[0,3,0.]]),
                  np.array([[1,1,1],[1,1,1],[1,1,1.]])]:self.compare(t)
        self.assertGreater(self.clipper.compiled,0)

    def test_touching_and_adjacent_float_values_match_bitwise(self):
        for x in [1.,np.nextafter(1.,-np.inf),np.nextafter(1.,np.inf)]:
            self.compare(np.array([[x,-.5,0],[x,.5,0],[x,0,.5]]),expansion=0.)
        self.assertGreater(self.clipper.compiled,0)

    def test_near_parallel_and_repeated_planes(self):
        for epsilon in [0.,1e-300,1e-16,1e-9]:
            t=np.array([[-2.,0,1+epsilon],[2.,0,1-epsilon],[0,2,1.]])
            self.compare(t,np.repeat(self.box,3,axis=0),expansion=0.)

    def test_fixed_randomized_polyhedra_match_reference(self):
        rng=np.random.default_rng(721091)
        for _ in range(600):
            vertices=rng.normal(size=(20,3));eq=ConvexHull(vertices).equations
            self.compare(rng.normal(size=(3,3))*2,eq)
        self.assertGreater(self.clipper.compiled,400)

    def test_dot_cancellation_and_scale_extremes(self):
        for scale in [1e-150,1e-20,1.,1e20,1e150]:
            eq=np.array([[1.,-1.,1.,0.],[-1.,1.,-1.,-scale]])
            self.compare(np.array([[scale,scale,0],[scale,scale,scale],[-scale,-scale,scale]]),eq,0.)

    def test_nonfinite_is_rejected_not_classified_clear(self):
        for value in [np.nan,np.inf,-np.inf]:
            t=np.zeros((3,3));t[0,0]=value
            with self.assertRaisesRegex(ValueError,'FINITE'):self.clipper.intersects(t,self.box)

    def test_full_report_matches_all_samples_contacts_and_order(self):
        args=example(n=65);expected=original_audit(Model(),*args)
        actual,stats=audit(Model(),*args,library=self.library)
        self.assertEqual(actual,expected)
        self.assertGreater(expected['forbidden_or_early_clearance_intersection_count'],50)
        self.assertEqual(stats['compiled'],actual['triangle_tests'])

    def test_full_clear_report_and_terminal_sample(self):
        for n in [2,18]:
            args=example(clear=True,n=n)
            actual,_=audit(Model(),*args,library=self.library)
            self.assertEqual(actual,original_audit(Model(),*args))
            self.assertEqual(actual['sampled_poses'],4*(n-1)+1)

    def test_timeout_cannot_return_a_partial_pass(self):
        with self.assertRaisesRegex(TimeoutError,'WORK_BUDGET'):
            audit(Model(),*example(n=3),library=self.library,max_wall_s=1e-9)

    def test_progress_is_explicitly_partial(self):
        rows=[];report,_=audit(Model(),*example(n=34),library=self.library,progress=rows.append)
        self.assertEqual(rows[-1]['samples_completed'],report['sampled_poses'])
        self.assertTrue(all(r['complete'] is False for r in rows))

    def test_parallel_partition_matches_complete_serial_report(self):
        for n,workers in [(2,8),(19,8),(18,3)]:
            args=example(n=n);report,stats=parallel_audit(Model(),*args,library=self.library,workers=workers,max_wall_s=10)
            self.assertEqual(report,original_audit(Model(),*args))
            self.assertEqual(stats['compiled'],report['triangle_tests'])

    def test_worker_failure_is_not_a_pass(self):
        with self.assertRaisesRegex(RuntimeError,'SYNTHETIC_MODEL_FAILURE'):
            parallel_audit(BrokenModel(),*example(),library=self.library,workers=2,max_wall_s=10)

    def test_timeout_stops_only_own_workers(self):
        before={p.pid for p in multiprocessing.active_children()};start=time.monotonic()
        with self.assertRaisesRegex(TimeoutError,'WORK_BUDGET'):
            parallel_audit(StalledModel(),*example(),library=self.library,workers=2,max_wall_s=1.2)
        self.assertLess(time.monotonic()-start,1.5)
        self.assertFalse({p.pid for p in multiprocessing.active_children()}-before)


def benchmark(path):
    rng=np.random.default_rng(439123)
    vertices=rng.normal(size=(400,3));vertices/=np.linalg.norm(vertices,axis=1)[:,None]
    equations=ConvexHull(vertices).equations
    triangles=rng.normal(size=(200,3,3))*1.2
    start=time.monotonic();reference=[bool(len(clip_triangle(t,equations,.006))) for t in triangles]
    serial=time.monotonic()-start;clipper=Clipper(os.environ['MESH_NUMPY_LIBRARY'])
    prepared=clipper.prepare(equations)
    start=time.monotonic();actual=[clipper.intersects(t,prepared) for t in triangles]
    native=time.monotonic()-start
    assert actual==reference
    Path(path).write_text(json.dumps(dict(scope='synthetic geometry only',triangles=len(triangles),planes=len(equations),
        exact_classifications_equal=True,reference_s=serial,compiled_s=native,speedup=serial/native,
        compiled=clipper.compiled,GPU_contexts=0),indent=2)+'\n')


if __name__=='__main__':
    if 'MESH_BENCHMARK_OUTPUT' in os.environ:benchmark(os.environ['MESH_BENCHMARK_OUTPUT'])
    unittest.main()
