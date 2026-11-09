import unittest
import numpy as np
from corner_edges import fit_flanks
from edge import classify_depth

class CornerTests(unittest.TestCase):
    def segments(self):
        rows=[]
        for side in [-1,1]:
            for y in np.linspace(.065,.295,50)*side:
                x=-.1-abs(y)
                rows.append([x-.001,y,0,x+.001,y,0,0,0,0,0])
        return np.asarray(rows)
    def test_two_lines_recover_corner(self):
        r,l=fit_flanks(self.segments());self.assertTrue(r['passed']);np.testing.assert_allclose(r['tip_xy_m'],[-.1,0],atol=1e-12)
    def test_single_flank_is_unknown(self):
        r,l=fit_flanks(self.segments()[:50]);self.assertFalse(r['passed'])
    def test_bent_edge_is_rejected(self):
        s=self.segments();s[10:20,0]+=.03;s[10:20,3]+=.03
        r,l=fit_flanks(s);self.assertFalse(r['passed'])
    def test_foreground_is_not_dropoff(self):
        on,occluded,beyond=classify_depth(np.array([.8,1.,1.2]),np.ones(3))
        self.assertEqual(occluded.tolist(),[True,False,False]);self.assertEqual(beyond.tolist(),[False,False,True])

if __name__=='__main__':unittest.main()
