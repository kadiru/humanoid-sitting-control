"""Synthetic algebra only; no study imports, data, models or solvers."""
import unittest
import math
import random
from fractions import Fraction as F
from reserve import compose, certify, require_coverage
from independent import audit


class ReserveTests(unittest.TestCase):
    def check(self, terms, constant, scale, base, transform, delta, actual,
              profile, eta, reserve=0):
        row = compose(terms,constant,scale,base,transform,profile)
        result = certify(row,base,transform,delta,actual,profile,eta,reserve)
        audit(terms,constant,scale,base,transform,delta,actual,profile,eta,reserve,result)
        return result

    def test_signed_gap_pair_exact_boundary(self):
        r=self.check([(0,-1),(1,1)],1,2,{0:0,1:0},{0:{0:1},1:{1:1}},
                     {0:0,1:1},{0:F(1,4),1:F(3,4)},{0:1,1:1},F(1,4),F(3,4))
        self.assertTrue(r['passed']);self.assertEqual(r['actual'],r['lower'])

    def test_negative_coefficient_error_is_not_discarded(self):
        r=self.check([(0,-2)],1,1,{0:0},{0:{0:1}},{0:0},{0:F(3,5)},
                     {0:1},F(1,2))
        self.assertFalse(r['passed']);self.assertFalse(r['error_pass'])

    def test_minimum_vertex_branch_switch(self):
        # At the tie min(x,-x), the stored tangent can be +1. No Hessian premise.
        r=self.check([(0,1)],1,1,{0:0},{0:{0:1}},{0:F(1,4)},
                     {0:-F(1,4)},{0:1},F(1,2))
        self.assertTrue(r['passed']);self.assertEqual(r['remainders'][0],-F(1,2))

    def test_same_branch_switch_rejects_insufficient_margin(self):
        r=self.check([(0,1)],1,1,{0:0},{0:{0:1}},{0:F(1,4)},
                     {0:-F(1,4)},{0:1},F(1,4))
        self.assertFalse(r['passed'])

    def test_finite_encoding_one_ulp_difference_included(self):
        a=math.nextafter(1.,math.inf)
        r=self.check([(0,1)],0,1,{0:1.},{0:{0:1}},{0:0},
                     {0:a},{0:1},F(1,2**52))
        self.assertTrue(r['passed']);self.assertEqual(r['remainders'][0],F(1,2**52))

    def test_zero_profile_requires_exact_error_zero(self):
        r=self.check([(0,1)],1,1,{0:0},{0:{}},{},{0:F(1,2**100)},
                     {0:0},F(1,2))
        self.assertFalse(r['passed'])

    def test_alias_cancellation_precedes_absolute_value(self):
        r=self.check([(0,1),(0,-1)],1,1,{}, {}, {}, {}, {},F(1,2))
        self.assertTrue(r['passed']);self.assertEqual(r['sensitivity'],0)

    def test_constant_negative_row_is_not_omitted(self):
        r=self.check([],-1,1,{}, {}, {}, {}, {},0)
        self.assertFalse(r['passed']);self.assertFalse(r['margin_pass'])

    def test_multiple_physical_coordinates_composed(self):
        r=self.check([(0,2),(1,-3)],10,5,{0:1,1:2},
                     {0:{0:2,1:-1},1:{0:-1,1:2}}, {0:1,1:1},
                     {0:2,1:3},{0:1,1:1},F(1,10))
        self.assertTrue(r['passed'])

    def test_missing_row_is_rejected(self):
        with self.assertRaises(ValueError):require_coverage([1,2],[(1,{'passed':True})])

    def test_duplicate_row_is_rejected(self):
        with self.assertRaises(ValueError):require_coverage([1],[(1,{'passed':True}),(1,{'passed':True})])

    def test_extra_row_is_rejected(self):
        with self.assertRaises(ValueError):require_coverage([1],[(1,{'passed':True}),(2,{'passed':True})])

    def test_failed_row_fails_whole_certificate(self):
        self.assertFalse(require_coverage([1,2],[(2,{'passed':False}),(1,{'passed':True})]))

    def test_complete_coverage(self):
        self.assertTrue(require_coverage([1,2],[(2,{'passed':True}),(1,{'passed':True})]))

    def test_wrong_tangent_detected(self):
        row=compose([(0,1)],1,1,{0:0},{0:{0:1}},{0:1});row['tangent'][0]=F(2)
        with self.assertRaises(ValueError):certify(row,{0:0},{0:{0:1}},{0:1},{0:1},{0:1},0)

    def test_wrong_reserve_weight_detected(self):
        row=compose([(0,-1)],1,1,{0:0},{0:{}},{0:1});row['sensitivity']=F(-1)
        with self.assertRaises(ValueError):certify(row,{0:0},{0:{}},{},{0:0},{0:1},0)

    def test_wrong_intercept_detected(self):
        row=compose([(0,1)],1,1,{0:0},{0:{}},{0:1});row['intercept']=F(2)
        with self.assertRaises(ValueError):certify(row,{0:0},{0:{}},{},{0:0},{0:1},0)

    def test_independent_audit_rejects_tampered_result(self):
        row=compose([(0,1)],1,1,{0:0},{0:{}},{0:1})
        result=certify(row,{0:0},{0:{}},{},{0:0},{0:1},0);result['lower']=F(2)
        with self.assertRaises(AssertionError):audit([(0,1)],1,1,{0:0},{0:{}},{},{0:0},{0:1},0,0,result)

    def test_invalid_scales_and_nonfinite(self):
        for scale in [0,-1,float('nan'),float('inf')]:
            with self.assertRaises(ValueError):compose([],0,scale,{}, {}, {})

    def test_negative_profile_and_eta(self):
        with self.assertRaises(ValueError):compose([(0,1)],0,1,{0:0},{0:{}},{0:-1})
        row=compose([],1,1,{}, {}, {})
        with self.assertRaises(ValueError):certify(row,{}, {}, {}, {}, {},-1)

    def test_small_exact_randomized_identity(self):
        rng=random.Random(20261024)
        for _ in range(100):
            coeff=[(j,F(rng.randint(-5,5),7)) for j in range(4)]
            base={j:F(rng.randint(-5,5),11) for j in range(4)}
            transform={j:{k:F(rng.randint(-5,5),13) for k in range(3)} for j in range(4)}
            delta={k:F(rng.randint(-5,5),17) for k in range(3)}
            profile={j:F(j+1,9) for j in range(4)};eta=F(1,8)
            actual={j:base[j]+sum((a*delta[k] for k,a in transform[j].items()),F(0))
                    +eta*profile[j]*F(rng.randint(-2,2),2) for j in range(4)}
            result=self.check(coeff,10,3,base,transform,delta,actual,profile,eta)
            self.assertTrue(result['passed'])


if __name__=='__main__':unittest.main()
