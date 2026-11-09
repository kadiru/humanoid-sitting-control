"""Invented arrays only. Reject any attempt to open study data during these tests."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def prohibit_study_data(event,args):
    if event=='open' and isinstance(args[0],(str,bytes)):
        name=str(args[0])
        if '/data/humanoid-sitting-perception/' in name or '/validated_low_seat_contact_01/' in name:
            raise RuntimeError('Synthetic test attempted study-data access')
sys.addaudithook(prohibit_study_data)

import unittest
import importlib.util
import tempfile
import json
import numpy as np
from scipy import sparse
from policy import validate_approval,advance,PROPOSAL_SHA256

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path)
    obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj

sys.path.insert(0,str(ROOT/'A'))
producer=module('candidate_check','A/numerical_checks.py')
auditor=module('candidate_audit','A/audit_numerical.py')
builder=module('interior_builder','B/builder.py')
progress=module('progress_gate','C/progress_gate.py')
progress_audit=module('progress_audit','C/audit_progress.py')

class ApprovalTests(unittest.TestCase):
    def test_default_package_cannot_authorize(self):
        import policy
        with self.assertRaises(PermissionError):policy.authorize()
        self.assertFalse((ROOT/'owner_claim.json').exists())
    def test_missing_approval_rejected(self):
        with self.assertRaises(PermissionError):validate_approval({'execution_approved':False},'x')
    def test_wrong_proposal_rejected(self):
        with self.assertRaises(AssertionError):validate_approval({'execution_approved':True,'proposal_sha256':'wrong'},'x')
    def test_bound_complete_receipt(self):
        validate_approval(dict(execution_approved=True,proposal_sha256=PROPOSAL_SHA256,
            freeze_sha256='test',explicit_human_message='invented fixture',human_message_reference='synthetic',
            exceptions=['single_cold_targeted_gap_retry','600_600_45_seconds_including_audits','strengthened_problem_optimality'],
            maximum_cycle_executions=1),'test')
    def test_no_downstream_after_failure(self):
        for stage in 'ABC':
            for exit_ok,audit_ok,accepted in [(False,True,True),(True,False,True),(True,True,False)]:
                self.assertIsNone(advance(stage,dict(execution_and_audit_passed=audit_ok,candidate_accepted=accepted),dict(passed=exit_ok)))
        self.assertEqual(advance('A',dict(execution_and_audit_passed=True,candidate_accepted=True),dict(passed=True)),'B')
        self.assertEqual(advance('B',dict(execution_and_audit_passed=True,candidate_accepted=True),dict(passed=True)),'C')
        self.assertIsNone(advance('C',dict(execution_and_audit_passed=True,candidate_accepted=True),dict(passed=True)))

class NumericalTests(unittest.TestCase):
    def setUp(self):
        # Original min x, x>=0; strengthened min x, x>=1. Added dual must be nonzero.
        self.M=sparse.csr_matrix([[-1.],[-1.]])
        self.E=sparse.csr_matrix((0,1));self.x=np.array([1.])
        self.a=dict(cost=np.array([1.]),rhs=np.array([0.,-1.]),equality_rhs=np.zeros(0),bounds=np.array([[-np.inf,np.inf]]))
        self.m=dict(inequality=np.array([0.,-1.]),equality=np.zeros(0),upper=np.zeros(1),lower=np.zeros(1))
    def test_strengthened_optimum_with_nonzero_added_dual(self):
        metrics,res=producer.check_point(self.M,self.E,self.a,self.x,self.m)
        self.assertTrue(metrics['passed']);self.assertEqual(metrics['absolute_gap'],0.)
        self.assertLessEqual(res['inequality'][0],0.)
        original,_=producer.check_point(self.M[:1],self.E,{**self.a,'rhs':self.a['rhs'][:1]},self.x,
            {**self.m,'inequality':self.m['inequality'][:1]})
        self.assertFalse(original['passed'])
        with tempfile.TemporaryDirectory() as folder:
            auditor.O=Path(folder);np.savez(auditor.O/'residuals.npz',**res)
            (auditor.O/'numerical_checks.json').write_text(json.dumps(metrics))
            self.assertEqual(auditor.audit_kkt(self.M,self.E,self.a,self.x,self.m),metrics)
            res['stationarity'][0]=.01;np.savez(auditor.O/'residuals.npz',**res)
            with self.assertRaises(AssertionError):auditor.audit_kkt(self.M,self.E,self.a,self.x,self.m)
    def test_exact_sign_violation_not_tolerated(self):
        self.m['inequality'][0]=3.25e-15
        self.assertFalse(producer.check_point(self.M,self.E,self.a,self.x,self.m)[0]['passed'])
    def test_infinite_bound_multiplier_rejected(self):
        self.m['upper'][0]=-1e-15
        self.assertFalse(producer.check_point(self.M,self.E,self.a,self.x,self.m)[0]['passed'])
    def test_primal_violation_rejected(self):
        self.x[0]=.5
        self.assertFalse(producer.check_point(self.M,self.E,self.a,self.x,self.m)[0]['passed'])

class AssemblyTests(unittest.TestCase):
    def test_original_interior_and_gap_tail(self):
        # One ordinary row, one two-sided consistency equation, two closed intervals.
        base=sparse.csr_matrix([[2,3,-1],[4,5,-1],[-4,-5,-1],
            [1,0,0],[0,1,0],[-1,0,0],[0,-1,0]],dtype=float)
        arrays=dict(rhs=np.array([0.,0.,0.,1.,1.,1.,1.]),equality_rhs=np.zeros(0))
        state=dict(r=np.array([.2]),guard=np.array([.01]),h=np.array([.3]))
        row=dict(keep=np.array([True]),soft=np.array([True]))
        interval=dict(lower=[[-1,1],[-1,1]],upper=[[1,1],[1,1]],widths=[[2,1],[2,1]])
        M,a=builder.construct(base,arrays,state,row,.5,interval,1,1)
        self.assertEqual(M.shape,(11,3));np.testing.assert_array_equal(M[:7,-1:].toarray(),np.zeros((7,1)))
        np.testing.assert_array_equal(M[7:].toarray(),[[1,0,1],[0,1,1],[-1,0,1],[0,-1,1]])
        np.testing.assert_array_equal(a['rhs'][3:7],arrays['rhs'][3:7])
        np.testing.assert_array_equal(a['rhs'][7:],np.ones(4)*.5)
        G=sparse.csr_matrix([[2,-3,0],[-1,4,0]],dtype=float)
        augmented=sparse.vstack([M,G],format='csr')
        np.testing.assert_array_equal(augmented[-2:].toarray(),G.toarray())
        self.assertEqual(augmented[-2:,-1:].nnz,0)

class ExactMarginTests(unittest.TestCase):
    def test_unscaled_margin_and_independent_trace(self):
        import time
        p=module('exact_producer_fixture','B/producer.py')
        a=module('exact_auditor_fixture','B/independent_return.py')
        H=sparse.csr_matrix([[1.,0.],[0.,1.],[.5,2.]])
        intervals=dict(lower=[[-2,1]]*3,upper=[[2,1]]*3,widths=[[4,1]]*3)
        with tempfile.TemporaryDirectory() as folder:
            p.O=a.O=Path(folder);p.HERE=a.HERE=ROOT/'B'
            x=np.array([.25,-.5,.1]);origin=time.monotonic()
            result=p.exact_margin(H,intervals,x[:-1],origin)
            self.assertTrue(result['passed']);self.assertEqual(result['exact_eta'],[9,32])
            self.assertEqual(a.audit_exact(H,intervals,x,origin),result)
            with self.assertRaises(AssertionError):a.audit_exact(H,intervals,np.array([.5,-.5,.1]),origin)
    def test_boundary_point_has_no_strict_margin(self):
        import time
        p=module('exact_boundary_producer','B/producer.py')
        a=module('exact_boundary_auditor','B/independent_return.py')
        H=sparse.csr_matrix([[1.,0.]])
        intervals=dict(lower=[[-2,1]],upper=[[2,1]],widths=[[4,1]])
        with tempfile.TemporaryDirectory() as folder:
            p.O=a.O=Path(folder);p.HERE=a.HERE=ROOT/'B';origin=time.monotonic()
            x=np.array([2.,0.,.5]);result=p.exact_margin(H,intervals,x[:-1],origin)
            self.assertFalse(result['passed']);self.assertEqual(result['exact_eta'],[0,1])
            self.assertEqual(a.audit_exact(H,intervals,x,origin),result)

class NonlinearGateTests(unittest.TestCase):
    def verify(self,before,after,mu=100.,phi=90.,predicted=20.,ratio=.5,aux=True,expected=True):
        hard=np.array([True,True,True,False]);b=np.asarray(before,dtype=float);a=np.asarray(after,dtype=float)
        x=progress.evaluate(mu,phi,predicted,ratio,aux,b,a,hard,(0,1))
        y=progress_audit.evaluate(mu,phi,predicted,ratio,aux,b,a,hard,(0,1))
        self.assertEqual(x,y);self.assertEqual(x['passed'],expected);self.assertFalse(x['physical_admission'])
    def test_actual_target_repair(self):self.verify([-2,-3,.1,-1],[0,0,.1,-100])
    def test_new_hard_failure_rejects_even_with_better_worst(self):self.verify([-2,-3,.1,0],[0,0,-.01,0],expected=False)
    def test_tiny_negative_target_rejects(self):self.verify([-2,-3,.1,0],[-1e-16,0,.1,0],expected=False)
    def test_worse_existing_failure_rejects(self):self.verify([-2,-3,-4,0],[0,0,-5,0],expected=False)
    def test_insufficient_actual_progress_rejects(self):self.verify([-2,-3,.1,0],[0,0,.1,0],phi=99.5,expected=False)
    def test_exact_one_percent_boundary_passes(self):self.verify([-2,-3,.1,0],[0,0,.1,0],phi=99.)
    def test_auxiliary_or_filter_failure_rejects(self):
        self.verify([-2,-3,.1,0],[0,0,.1,0],aux=False,expected=False)
        self.verify([-2,-3,.1,0],[0,0,.1,0],ratio=.09,expected=False)

class WatchdogTests(unittest.TestCase):
    def test_late_process_is_killed_and_not_admitted(self):
        import time
        old_argv=sys.argv
        try:
            sys.argv=[str(ROOT/'supervise.py'),'A',repr(time.monotonic())]
            supervisor=module('watchdog_fixture','supervise.py')
        finally:sys.argv=old_argv
        with tempfile.TemporaryDirectory() as folder:
            supervisor.HERE=supervisor.O=Path(folder)
            (supervisor.HERE/'sleep.py').write_text('import time; time.sleep(10)\n')
            supervisor.ORIGIN=time.monotonic();supervisor.PRODUCER_END=.15
            result=supervisor.run('producer','sleep.py','synthetic-only')
            self.assertFalse(result['completed_within_deadline'])
            self.assertEqual(result['hard_termination_reason'],'absolute_stage_deadline')
            self.assertEqual(result['actual_exit'],-9)
    def test_successful_child_is_reaped(self):
        import time
        old_argv=sys.argv
        try:
            sys.argv=[str(ROOT/'supervise.py'),'A',repr(time.monotonic())]
            supervisor=module('watchdog_success_fixture','supervise.py')
        finally:sys.argv=old_argv
        with tempfile.TemporaryDirectory() as folder:
            supervisor.HERE=supervisor.O=Path(folder)
            (supervisor.HERE/'done.py').write_text('print("synthetic-only")\n')
            supervisor.ORIGIN=time.monotonic();supervisor.PRODUCER_END=2.
            result=supervisor.run('producer','done.py','synthetic-only')
            self.assertTrue(result['completed_within_deadline'])
            self.assertEqual(result['actual_exit'],0)

if __name__=='__main__':unittest.main(verbosity=2)
