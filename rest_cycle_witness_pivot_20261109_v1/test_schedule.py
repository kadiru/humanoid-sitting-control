import sys,unittest
from pathlib import Path
R=Path('/home/kadir/humanoid-sitting')
sys.path[1:1]=[str(R/'experiments/upper_thigh_support_20261108_v4'),str(R/'sitting_task/src')]
import numpy as np
from pivot import schedule

class Scheduling(unittest.TestCase):
    def nodes(self,a,b,normal=False):
        return [dict(command=np.r_[np.full(23,x),np.ones(14)].tolist(),phase='landing' if normal else 'approach',normal=normal,gap=.012 if i==0 else 0.) for i,x in enumerate([a,b])]
    def verify(self,n):
        s=schedule(n,np.full(23,-2.),np.full(23,2.));c=np.array(s['commands']);d=np.array(s['desired']);p=np.vstack([n[0]['command'],c[:-1]])
        self.assertLessEqual(abs(c-p).max(),.003)
        self.assertTrue((d>=-1.97).all() and (d<=1.97).all())
        np.testing.assert_allclose(c[:,:23],p[:,:23]+np.clip(.1*(d-p[:,:23]),-.003,.003),atol=1e-13,rtol=0)
        np.testing.assert_array_equal(c[:,23:],np.ones((len(c),14)))
        return s
    def test_upper_headroom(self):self.verify(self.nodes(1.96,1.969))
    def test_lower_headroom(self):self.verify(self.nodes(-1.96,-1.969))
    def test_forward_and_reverse(self):
        n=self.nodes(.1,.4);self.verify(n);self.verify(n[::-1])
    def test_no_zero_length_failure(self):self.verify(self.nodes(.4,.4))
    def test_normal_speed(self):self.assertGreaterEqual(self.verify(self.nodes(.1,.11,True))['seconds'],.012/.0055)
    def test_outside_margin_rejected(self):
        with self.assertRaisesRegex(ValueError,'ORIGINAL_MARGIN'):schedule(self.nodes(1.5,1.98),np.full(23,-2.),np.full(23,2.))
    def test_finite_return(self):
        with self.assertRaisesRegex(ValueError,'RETURN_CAP'):schedule(self.nodes(0,1.969999999),np.full(23,-2.),np.full(23,2.))
if __name__=='__main__':unittest.main()
