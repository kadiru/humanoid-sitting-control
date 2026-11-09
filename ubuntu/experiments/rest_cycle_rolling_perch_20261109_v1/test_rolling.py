import unittest
import numpy as np
from rolling import Rolling
class ShapeRolling(unittest.TestCase):
    def setUp(self):
        self.v=np.array([[x,y,z] for x in [-.1,.1] for y in [-.05,.05] for z in [-.15,.05]])
    def test_roll_does_not_sink(self):
        r=Rolling(self.v,[0,0,1],[0,0,0],0,.5)
        self.assertTrue(r.audit([0,0,.15],0)['passed'])
    def test_material_point_stays_fixed_between_events(self):
        r=Rolling(self.v,[0,0,1],[0,0,0],0,.5);p=r.translation(.4,[0,0,.15])
        idx=r.segments[0][2]
        np.testing.assert_allclose(p+r.rotation(.4)@r.v[idx],np.array([0,0,.15])+r.v[idx],atol=1e-13,rtol=0)
    def test_vertex_switch_continuity(self):
        r=Rolling(self.v,[0,0,1],[0,-.2,0],-.2,.3)
        initial=np.array([0,0,-(self.v@r.rotation(-.2).T)[:,2].min()])
        self.assertGreaterEqual(len(r.segments),2)
        self.assertTrue(r.audit(initial,0)['passed'])
    def test_bad_start_not_admitted(self):
        r=Rolling(self.v,[0,0,1],[0,0,0],0,.5)
        self.assertFalse(r.audit([0,0,.14],0)['passed'])
    def test_frozen_interval(self):
        r=Rolling(self.v,[0,0,1],[0,0,0],0,.5)
        with self.assertRaises(ValueError):r.translation(.6,[0,0,.15])
if __name__=='__main__':unittest.main()
