import unittest
import numpy as np
from edge import classify_depth
from release import separation

class Components(unittest.TestCase):
    def test_foreground_is_not_edge(self):
        on,occ,back=classify_depth(np.array([.5,1.,1.5,np.inf,np.nan]),np.ones(5))
        self.assertEqual(on.tolist(),[False,True,False,False,False])
        self.assertEqual(occ.tolist(),[True,False,False,False,False])
        self.assertEqual(back.tolist(),[False,False,True,True,False])
    def base(self,v):
        return dict(vertices={k:v for k in ['pelvis','left_hip_yaw_link','right_hip_yaw_link','torso_link','logo_link']},
          plane_normal=[0,0,1],plane_offset=0,front_upper=0,
          front_error=.003,normal_error=.003,normal_angle_error=0.,qualified_edge=True,geometry_age=0.)
    def test_convex_edge_crossing_not_clear(self):
        v=np.array([[.1,0,-.1],[-.1,0,.1]])
        self.assertEqual(separation(**self.base(v))['state'],'unknown')
    def test_full_above(self):
        self.assertEqual(separation(**self.base(np.array([[-.1,0,.02],[.1,0,.03]])))['state'],'geometrically_separated')
    def test_full_forward(self):
        self.assertEqual(separation(**self.base(np.array([[.02,0,-.1],[.03,0,.1]])))['state'],'geometrically_separated')
    def test_unknown_edge_cannot_release(self):
        a=self.base(np.array([[.02,0,.02]]));a['qualified_edge']=False
        self.assertIsNone(separation(**a)['load_upper_bw'])
    def test_stale_edge_cannot_release(self):
        a=self.base(np.array([[.02,0,.02]]));a['geometry_age']=.22
        self.assertIsNone(separation(**a)['load_upper_bw'])
    def test_unknown_camera_age_cannot_assert_zero_load(self):
        for age in (np.nan, np.inf, -np.inf):
            with self.subTest(age=age):
                a=self.base(np.array([[.02,0,.02]]));a['geometry_age']=age
                result=separation(**a)
                self.assertEqual(result['state'],'unknown')
                self.assertIsNone(result['load_upper_bw'])
    def test_empty_collider_cannot_certify_release(self):
        a=self.base(np.array([[.02,0,.02]]))
        a['vertices']['pelvis']=np.empty((0,3))
        self.assertIsNone(separation(**a)['load_upper_bw'])
    def test_malformed_plane_cannot_certify_release(self):
        for field in ('plane_normal','plane_center'):
            with self.subTest(field=field):
                a=self.base(np.array([[.02,0,.02]]));a[field]=[0.,0.]
                self.assertIsNone(separation(**a)['load_upper_bw'])
    def test_original_age_boundary_is_inclusive(self):
        a=self.base(np.array([[.02,0,.02]]));a['geometry_age']=.21
        self.assertEqual(separation(**a)['state'],'geometrically_separated')
        a['geometry_age']=np.nextafter(.21,np.inf)
        self.assertIsNone(separation(**a)['load_upper_bw'])
    def test_buffer_is_strict(self):
        a=self.base(np.array([[.009,0,.009]]))
        self.assertIsNone(separation(**a)['load_upper_bw'])

if __name__=='__main__':unittest.main()
