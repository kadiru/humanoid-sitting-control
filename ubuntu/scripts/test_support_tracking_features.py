"""Regression checks for the frozen actor's complete measured-patch contract."""
import unittest
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
from test_support_landing_geometry import LandingTests
from humanoid_sitting.support_tracking_features import measured_tracking_features
from humanoid_sitting.support_depth import CurrentSupportDepth
from humanoid_sitting.shared_frame import level_rotation


class FeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        LandingTests.setUpClass();cls.fixture=LandingTests()

    def test_distance_only_never_claims_observed_bounds(self):
        o=self.fixture.observer()
        for source in ('kinematic_prediction','seat_from_observed_floor','unavailable'):
            values,info=measured_tracking_features(o,dict(valid=True,source=source),None,None,None)
            np.testing.assert_array_equal(values,np.array([0.]*9+[1.],dtype=np.float32))
            self.assertFalse(info['usable'])

    def test_visible_seat_without_floor_emits_actual_patch(self):
        f=self.fixture;o=f.observer();R=level_rotation(f.g);height=f.floor+o.seat_height
        x,y=np.meshgrid(np.linspace(-.5,.1,20),np.linspace(-.5,.5,20))
        plane=dict(normal=np.array([0.,0.,1.]),offset=-height,
            points=np.c_[x.ravel(),y.ravel(),np.full(x.size,height)],rms=.001,height=height)
        scene=dict(R=R,planes=[plane],diagnostic={})
        with patch('humanoid_sitting.support_landing_geometry.extract_scene',return_value=scene):
            result=o.update(.02,f.q,f.g,np.zeros(3),np.ones((96,128)))
        self.assertEqual(result['source'],'depth_with_kinematic_floor')
        with patch('humanoid_sitting.support_tracking_features.extract_scene',return_value=scene):
            values,info=measured_tracking_features(o,result,None,f.q,f.g)
        self.assertTrue(info['usable']);self.assertEqual(values[8],1.)
        self.assertAlmostEqual(values[0],result['clearance_m'],places=6)
        points=plane['points']@R
        np.testing.assert_allclose(values[4:6],points[:,:2].mean(0),atol=1e-7)
        np.testing.assert_allclose(values[6:8],np.ptp(points[:,:2],axis=0),atol=1e-7)

    def test_ambiguous_patch_is_withheld(self):
        f=self.fixture;o=f.observer();R=level_rotation(f.g);height=f.floor+o.seat_height
        plane=dict(normal=np.array([0.,0.,1.]),offset=-height,points=np.zeros((100,3)))
        result=dict(valid=True,source='depth_with_kinematic_floor',normal_P=(R.T@plane['normal']).tolist(),
            clearance_m=float((o.model.vertices['pelvis']@(R.T@plane['normal'])-height).min()))
        with patch('humanoid_sitting.support_tracking_features.extract_scene',return_value=dict(R=R,planes=[plane,plane])):
            values,info=measured_tracking_features(o,result,None,f.q,f.g)
        self.assertEqual(values[8],0.);self.assertEqual(info['reason'],'ambiguous_patch_identity')

    def test_existing_valid_measurement_features_are_exactly_preserved(self):
        cap=Path('/data/humanoid-sitting-perception/support_landing_regression_capture_01')
        cal=json.loads((cap/'calibration.json').read_text());c=cal['cases']['0']
        with np.load(cap/'depth.npz') as d, np.load(cap/'proprioception.npz') as p:
            depth=d['depth'][0];q=p['q'][0,0];g=p['gravity'][0,0];gyro=p['gyro'][0,0]
        from humanoid_sitting.support_landing_geometry import LandingGeometryTracker
        o=LandingGeometryTracker(cap/'model.json',c['K'],c['T_P_C'],c['estimated_seat_height_m'])
        r=o.update(.02,q,g,gyro,depth)
        self.assertEqual(r['source'],'current_floor_and_seat')
        values,info=measured_tracking_features(o,r,depth,q,g)
        legacy,_=CurrentSupportDepth(cap/'model.json',np.array(c['K']),np.array(c['T_P_C']),c['estimated_seat_height_m']).read(depth,q,g)
        self.assertTrue(info['usable']);np.testing.assert_array_equal(values,legacy)

if __name__=='__main__':unittest.main()
