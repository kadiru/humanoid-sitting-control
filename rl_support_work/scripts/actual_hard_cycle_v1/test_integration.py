"""Synthetic stage integration; no optimizer or mechanics evaluations."""
import math
import random
import unittest
from fractions import Fraction as F
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
from unittest.mock import patch
import json
import hashlib
import numpy as np
from scipy import sparse
import policy
import reserve_bridge as bridge
from dyadic import D,ZERO,directed
import assemble,audit
import posterior,posterior_audit


class IntegrationTests(unittest.TestCase):
    def test_no_approval_no_launch_and_no_owner_claim(self):
        with self.assertRaises(PermissionError):policy.authorize()
        self.assertFalse((policy.ROOT/'owner_claim.json').exists())

    def test_closed_cycle_approval_does_not_match_new_method(self):
        receipt=dict(execution_approved=True,proposal_sha256=policy.PROPOSAL_SHA256,
            freeze_sha256='synthetic',explicit_human_message='fixture',human_message_reference='fixture',
            exceptions=['single_cold_targeted_gap_retry','600_600_45_seconds_including_audits','strengthened_problem_optimality'],maximum_cycle_executions=1)
        with self.assertRaises(AssertionError):policy.validate_approval(receipt,'synthetic')

    def test_failed_or_late_stage_cannot_advance(self):
        for stage in 'ABC':
            self.assertIsNone(policy.advance(stage,dict(execution_and_audit_passed=True,candidate_accepted=False),dict(passed=True)))
            self.assertIsNone(policy.advance(stage,dict(execution_and_audit_passed=True,candidate_accepted=True),dict(passed=False)))
        self.assertEqual(policy.advance('A',dict(execution_and_audit_passed=True,candidate_accepted=True),dict(passed=True)),'B')
        self.assertIsNone(policy.advance('C',dict(execution_and_audit_passed=True,candidate_accepted=True),dict(passed=True)))

    def test_worker_and_helper_virtual_budgets_sum_to_40gib(self):
        self.assertEqual(policy.WORKER_LIMIT+2*policy.HELPER_LIMIT,40*1024**3)
        for cap,solver,producer,independent in policy.STAGES.values():
            self.assertLess(solver,producer);self.assertLess(producer,independent);self.assertLess(independent,cap)

    def test_cached_upward_eta_implication(self):
        rng=random.Random(341)
        accepted=0
        for _ in range(80):
            st=assemble.State((D(1,-3),)*75,({0:D(3,-3)},)+({},)*74,(D(1,-4),)*75)
            coefficient=rng.uniform(-1,1)
            row=assemble.compose(0,[(0,coefficient)],0.2,0.012,[st])
            enc=assemble.encode(row,{0:(D(-1),D(1))})
            upper=directed(row.kappa,row.scale,upward=True)
            self.assertGreaterEqual(F(upper),audit.q(row.kappa)/audit.q(row.scale))
            for x in [F(-1),F(0),F(1)]:
                for eta in [F(0),F(1,8),F(1,2)]:
                    lhs=sum(F(a)*x for a in enc.values)+F(upper)*eta
                    if lhs<=F(enc.rhs):
                        accepted+=1
                        self.assertGreaterEqual(audit.q(row.intercept)+audit.q(row.tangent.get(0,ZERO))*x-audit.q(row.kappa)*eta,0)
        self.assertGreater(accepted,0)

    def fixture_cache(self,folder):
        # All arrays are generated below, not read from any project dataset.
        n=156647;G=sparse.csr_matrix(([0.25,-0.125],([0,1],[0,1])),shape=(n,78791))
        rhs=np.ones(n);kappa=np.zeros(n);kappa[:2]=[0.125,0.25]
        sparse.save_npz(folder/'hard_matrix.npz',G)
        np.savez(folder/'hard_arrays.npz',rhs=rhs,kappa_upper=kappa,original_ids=np.arange(n,dtype=np.uint32))
        (folder/'hard_boxes.json').write_text(json.dumps({'0':dict(column=0,feature=75,interval=0,lower=-1.0,upper=1.0),
            '1':dict(column=1,feature=76,interval=1,lower=-1.0,upper=1.0)}))
        sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
        read=lambda p:json.loads(Path(p).read_text())
        write=lambda p,x:Path(p).write_text(json.dumps(x))
        write(folder/'hard_input_audit.json',dict(passed=True,upward_kappa_audited=True,hard_rows=n,
            files_sha256={p:sha(folder/p) for p in ('hard_matrix.npz','hard_arrays.npz','hard_boxes.json')}))
        c=SimpleNamespace(PRIMARY=folder,O=folder,read=read,write=write,sha=sha,check=lambda origin:None)
        return c,G,rhs,kappa

    def test_B_reuses_immutable_guard_and_adds_eta_column(self):
        with TemporaryDirectory(prefix='actual_hard_synthetic_') as td:
            c,G,rhs,kappa=self.fixture_cache(Path(td))
            out,rr,boxes=bridge.cache(c,0,interior=True)
            bridge.sparse_same(out[:,:78790],G[:,:78790]);bridge.same(rr,rhs)
            np.testing.assert_array_equal(out[:,-1].toarray().ravel(),kappa)
            self.assertEqual(set(boxes),{0,1})

    def test_append_keeps_full_interior_prefix_and_bounds(self):
        with TemporaryDirectory(prefix='actual_hard_synthetic_') as td:
            c,G,rhs,kappa=self.fixture_cache(Path(td))
            base=sparse.csr_matrix(([2.0],([837063],[4])),shape=(837065,78791))
            arrays=dict(rhs=np.zeros(837065),cost=np.r_[np.zeros(78790),-1.],
                bounds=np.c_[np.full(78791,-np.inf),np.full(78791,np.inf)],equality_rhs=np.zeros(1366))
            arrays['bounds'][-1]=[0,0.5]
            out,aa=bridge.append(c,0,base,arrays,interior=True)
            self.assertTrue(bridge.audit_extension(c,0,out,aa,base,arrays,interior=True))
            self.assertEqual(out.shape,(993712,78791))
            self.assertEqual(out[837063,4],2.0)
            aa['bounds'][0,1]=2
            with self.assertRaises(AssertionError):bridge.audit_extension(c,0,out,aa,base,arrays,True)

    def test_two_exact_new_point_implementations_agree_on_failure(self):
        with TemporaryDirectory(prefix='actual_hard_synthetic_') as td:
            c,G,rhs,kappa=self.fixture_cache(Path(td))
            point=np.zeros(78791);point[-1]=0.125
            good=bridge.point_certificate(c,0,point,True)
            self.assertTrue(good['passed'])
            self.assertEqual(good,bridge.point_certificate(c,0,point,True,True))
            point[0]=math.nextafter(1.0,math.inf)
            bad=bridge.point_certificate(c,0,point,True)
            self.assertFalse(bad['passed']);self.assertFalse(bad['exact_boxes'])
            self.assertEqual(bad,bridge.point_certificate(c,0,point,True,True))

    def test_posterior_pass_and_nonlinear_failure_diagnostics_agree(self):
        p=[[(j,1.0)] for j in range(24)];x=[0.0]*24;y=[0.0]*51
        J=[[0.0]*24 for _ in range(51)];J[47][2]=0.5
        raw=[(x,y,J,p,[0.125]*24,[0.25]*51,False)]
        dx=[[ZERO]*24];dx[0][2]=D(1,-10)
        pairs=[[v.ratio() for v in dx[0]]]
        actual=[[0.0]*75];actual[0][2]=2**-10;actual[0][71]=2**-11
        terms=[[(71,1.0)],[(71,-1.0)],[]]
        args=(terms.__getitem__,[True]*3,[False]*3,[1.0]*3,actual,0.125,lambda:None)
        good=posterior.evaluate(raw,dx,*args)
        self.assertTrue(good['passed']);self.assertEqual(good,posterior_audit.evaluate(raw,pairs,*args))
        actual[0][71]+=0.5
        bad=posterior.evaluate(raw,dx,*args)
        self.assertFalse(bad['passed']);self.assertEqual(bad['first_feature_error']['feature'],71)
        self.assertEqual(bad,posterior_audit.evaluate(raw,pairs,*args))


if __name__=='__main__':unittest.main()
