import unittest
import numpy as np
from observed_patch_v3 import project_mask
class TopologyTests(unittest.TestCase):
    def project(self,p):
        p=np.asarray(p,float);return np.c_[(p[:,0]-.7)/(.4+.071*p[:,1]),(p[:,1]-.3)/(.4+.071*p[:,1])]
    def test_connected_staircase(self):
        m=np.zeros((30,40),bool)
        for r in range(3,27):m[r,2+r//3:37-r//4]=True
        p,t=project_mask(m,self.project);self.assertEqual(p.geom_type,'Polygon');self.assertEqual(t['pixel_components'],1);self.assertEqual(p.interiors.__len__(),0)
    def test_hole_is_not_filled(self):
        m=np.ones((20,20),bool);m[7:12,8:13]=False
        p,t=project_mask(m,self.project);self.assertEqual(len(p.interiors),1);self.assertEqual(t['pixel_holes'],1)
    def test_separate_regions_stay_separate(self):
        m=np.zeros((20,20),bool);m[2:5,2:5]=True;m[12:17,12:17]=True
        p,t=project_mask(m,self.project);self.assertEqual(p.geom_type,'MultiPolygon');self.assertEqual(len(p.geoms),2)
    def test_empty_is_unknown(self):
        with self.assertRaisesRegex(ValueError,'NO_LOCAL_PLANAR_PIXELS'):project_mask(np.zeros((5,5),bool),self.project)
if __name__=='__main__':unittest.main()
