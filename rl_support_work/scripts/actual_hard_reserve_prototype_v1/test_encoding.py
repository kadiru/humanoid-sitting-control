"""Synthetic exact implication and rounding-cell checks, no study inputs."""
import unittest
import math
from fractions import Fraction as F
from encoding import encode
from audit_encoding import audit


class EncodingTests(unittest.TestCase):
    def test_mixed_sign_coefficient_errors(self):
        a={0:F(1,3),1:-F(2,7)};b=F(1,11);box={0:(-2,3),1:(-4,5)}
        z=encode(a,b,box);self.assertTrue(audit(a,b,box,z))
        for i in range(-20,31):
            for j in range(-40,51):
                x={0:F(i,10),1:F(j,10)}
                lhs=sum((F(z['coefficients'][k])*x[k] for k in a),F(0))
                if lhs<=F(z['rhs']):
                    self.assertLessEqual(sum((a[k]*x[k] for k in a),F(0)),b)

    def test_exact_coefficients_keep_zero_encoding_error(self):
        a={0:F(1,4),1:-2};box={0:(-1,2),1:(0,1)}
        z=encode(a,F(1,8),box);self.assertTrue(audit(a,F(1,8),box,z))
        self.assertEqual(z['minimum_error'],0)

    def test_eta_box_included(self):
        a={0:F(2,3),99:F(5,7)};box={0:(-1,1),99:(0,F(1,2))}
        z=encode(a,F(1,5),box);self.assertTrue(audit(a,F(1,5),box,z))

    def test_zero_coefficient_and_constant_row(self):
        z=encode({0:0},-F(1,3),{})
        self.assertTrue(audit({0:0},-F(1,3),{},z));self.assertEqual(z['coefficients'],{})

    def test_subnormal_coefficient(self):
        a={0:F(1,2**1076)};box={0:(-1,2)}
        z=encode(a,0,box);self.assertTrue(audit(a,0,box,z))
        self.assertEqual(z['coefficients'][0],0.)

    def test_halfway_nearest_even(self):
        a={0:(F(1.)+F(math.nextafter(1.,math.inf)))/2};box={0:(-1,1)}
        z=encode(a,1,box);self.assertTrue(audit(a,1,box,z))
        self.assertEqual(z['coefficients'][0],1.)

    def test_outward_rhs_tampering_rejected(self):
        a={0:F(1,3)};box={0:(-1,2)};z=encode(a,F(1,7),box)
        z['rhs']=math.nextafter(z['rhs'],math.inf)
        with self.assertRaises(AssertionError):audit(a,F(1,7),box,z)

    def test_missing_box_and_reversed_box(self):
        with self.assertRaises(KeyError):encode({0:F(1,3)},1,{})
        with self.assertRaises(ValueError):encode({0:1},1,{0:(1,-1)})


if __name__=='__main__':unittest.main()
