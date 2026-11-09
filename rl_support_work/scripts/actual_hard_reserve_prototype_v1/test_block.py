import unittest
from fractions import Fraction as F
from block import build


class BlockTests(unittest.TestCase):
    def fixture(self):
        return [0]*24,[0]*51,[[0]*24 for _ in range(51)],[{} for _ in range(24)],[1]*24,[1]*51

    def test_actual_gap_uses_output47_and_physical_control(self):
        x,y,j,p,v,s=self.fixture();y[47]=F(3,1000);j[47][2]=2;p[2]={27:1}
        b=build(x,y,j,p,v,s)
        self.assertEqual(b['base_actual'][71],F(3,1000))
        self.assertEqual(b['transform'][71],{27:F(2)})
        self.assertEqual(b['transform'][2],{27:F(1)})

    def test_conserved_load_pivot_is_included(self):
        x,y,j,p,v,s=self.fixture();p[16]={5:1};p[23]={5:-1}
        j[31][16]=3;j[31][23]=7
        b=build(x,y,j,p,v,s)
        self.assertEqual(b['transform'][55],{5:F(-4)})

    def test_fixed_state_has_zero_profile_and_no_direction(self):
        x,y,j,p,v,s=self.fixture();j[0][0]=1
        b=build(x,y,j,p,v,s,fixed=True)
        self.assertTrue(all(not row for row in b['transform'].values()))
        self.assertTrue(all(width==0 for width in b['profile'].values()))

    def test_fixed_state_nonzero_basis_rejected(self):
        x,y,j,p,v,s=self.fixture();p[0]={1:1}
        with self.assertRaises(ValueError):build(x,y,j,p,v,s,fixed=True)

    def test_shape_mismatch_rejected(self):
        x,y,j,p,v,s=self.fixture()
        with self.assertRaises(ValueError):build(x[:-1],y,j,p,v,s)


if __name__=='__main__':unittest.main()
