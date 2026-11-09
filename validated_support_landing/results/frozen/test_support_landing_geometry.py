"""Reference propagation, identity, expiry and contact-assumption tests."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from humanoid_sitting.support_landing_geometry import LandingGeometryTracker,foot_geometry
from humanoid_sitting.shared_frame import level_rotation
from seat_depth_estimator import estimate_body_frame
CAP=Path('/data/humanoid-sitting-perception/support_landing_development_capture_01')

class LandingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cal=json.loads((CAP/'calibration.json').read_text());a=np.load(CAP/'startup.npz')
        cls.q=a['q'][0].copy();cls.g=a['gravity'][0].copy();cls.c=cls.cal['cases']['0']
        m=cls.cal['startup_T_P_C']
        cls.floor=estimate_body_frame(a['depth'][0],a['K'][0],np.array(m['mount']),np.array(m['quaternion_xyzw']),cls.g)['floor_height_body_m']

    def observer(self):
        o=LandingGeometryTracker(CAP/'model.json',self.c['K'],self.c['T_P_C'],self.c['estimated_seat_height_m'])
        self.assertTrue(o.initialize_from_measured_floor(0.,self.q,self.g,self.floor))
        return o

    def test_encoder_propagation_changes_clearance_by_leg_height(self):
        o=self.observer();a=o.update(.02,self.q,self.g,np.zeros(3))
        q=self.q.copy()
        for name in ('left_knee_joint','right_knee_joint'):q[o.model.names.index(name)]+=.05
        b=o.update(.04,q,self.g,np.zeros(3))
        R=level_rotation(self.g)
        z0,_=foot_geometry(o.model,o.model.fk(self.q),R);z1,_=foot_geometry(o.model,o.model.fk(q),R)
        self.assertTrue(a['valid'] and b['valid']);self.assertEqual(b['source'],'kinematic_prediction')
        self.assertGreater(abs(b['clearance_m']-a['clearance_m']),1e-5)
        self.assertAlmostEqual(b['clearance_m']-a['clearance_m'],-float((z1-z0).mean()),places=10)

    def test_prediction_expires_and_reset_removes_reference(self):
        o=self.observer()
        for stamp in np.arange(.02,.44,.02):r=o.update(stamp,self.q,self.g,np.zeros(3))
        self.assertFalse(r['valid']);self.assertEqual(r['reason'],'prediction_expired')
        self.assertIsNone(r['clearance_m'])
        o.reset();r=o.update(.02,self.q,self.g,np.zeros(3))
        self.assertFalse(r['valid']);self.assertEqual(r['reason'],'no_measured_reference')

    def test_asymmetric_foot_lift_and_motion_invalidate_prediction(self):
        o=self.observer();q=self.q.copy();q[o.model.names.index('left_knee_joint')]+=.8
        r=o.update(.02,q,self.g,np.zeros(3));self.assertFalse(r['valid'])
        self.assertGreater(r['foot_disagreement_m'],.015)
        o=self.observer();self.assertFalse(o.update(.02,self.q,self.g,np.array([3.,0.,0.]))['valid'])

    def test_common_mode_foot_motion_consumes_uncertainty_budget(self):
        o=self.observer();q=self.q.copy()
        for name in ('left_knee_joint','right_knee_joint'):q[o.model.names.index(name)]+=.5
        r=o.update(.02,q,self.g,np.zeros(3))
        self.assertLess(r['foot_disagreement_m'],.015)
        self.assertFalse(r['valid']);self.assertIsNone(r['clearance_m'])

    def test_timestamp_gap_and_invalid_sensor_do_not_keep_stale_anchor(self):
        o=self.observer();o.update(.02,self.q,self.g,np.zeros(3))
        with self.assertRaises(ValueError):o.update(.02,self.q,self.g,np.zeros(3))
        self.assertFalse(o.update(.3,self.q,self.g,np.zeros(3))['valid'])
        o=self.observer();g=self.g.copy();g[0]=np.nan
        self.assertFalse(o.update(.02,self.q,g,np.zeros(3))['valid']);self.assertIsNone(o.floor_anchor)

    def test_visible_seat_survives_common_mode_foot_lift_without_a_floor(self):
        o=self.observer();q=self.q.copy()
        for name in ('left_knee_joint','right_knee_joint'):q[o.model.names.index(name)]+=.5
        R=level_rotation(self.g);height=self.floor+self.c['estimated_seat_height_m']+.002
        x,y=np.meshgrid(np.linspace(-.5,.1,20),np.linspace(-.5,.5,20))
        points=np.c_[x.ravel(),y.ravel(),np.full(x.size,height)]
        plane=dict(normal=np.array([0.,0.,1.]),offset=-height,points=points,rms=.001,height=height)
        scene=dict(R=R,planes=[plane],diagnostic={})
        with patch('humanoid_sitting.support_landing_geometry.extract_scene',return_value=scene):
            r=o.update(.02,q,self.g,np.zeros(3),np.ones((96,128)))
        self.assertTrue(r['valid']);self.assertEqual(r['source'],'depth_with_temporal_identity')
        self.assertAlmostEqual(r['clearance_m'],float((o.model.vertices['pelvis']@R.T)[:,2].min()-height),places=10)

    def test_visible_floor_does_not_refresh_the_occluded_seat_forever(self):
        o=self.observer();R=level_rotation(self.g)
        x,y=np.meshgrid(np.linspace(-.5,.1,20),np.linspace(-.5,.5,20))
        points=np.c_[x.ravel(),y.ravel(),np.full(x.size,self.floor)]
        plane=dict(normal=np.array([0.,0.,1.]),offset=-self.floor,points=points,rms=.001,height=self.floor)
        with patch('humanoid_sitting.support_landing_geometry.extract_scene',return_value=dict(R=R,planes=[plane],diagnostic={})):
            for t in np.arange(.1,.61,.1):
                r=o.update(t,self.q,self.g,np.zeros(3),np.ones((96,128)))
                self.assertTrue(r['valid']);self.assertEqual(r['source'],'seat_from_observed_floor')
            r=o.update(.7,self.q,self.g,np.zeros(3),np.ones((96,128)))
        self.assertFalse(r['valid']);self.assertEqual(o.seat_last_seen,0.)

    def test_lone_seat_can_be_identified_but_lone_floor_is_not_a_seat(self):
        o=self.observer();R=level_rotation(self.g)
        x,y=np.meshgrid(np.linspace(-.5,.1,20),np.linspace(-.5,.5,20))
        height=self.floor+self.c['estimated_seat_height_m']
        def scene(z):
            points=np.c_[x.ravel(),y.ravel(),np.full(x.size,z)]
            plane=dict(normal=np.array([0.,0.,1.]),offset=-z,points=points,rms=.001,height=z)
            return dict(R=R,planes=[plane],diagnostic={})
        with patch('humanoid_sitting.support_landing_geometry.extract_scene',return_value=scene(height)):
            r=o.update(.02,self.q,self.g,np.zeros(3),np.ones((96,128)))
        self.assertTrue(r['valid']);self.assertEqual(r['source'],'depth_with_kinematic_floor')
        o=self.observer()
        with patch('humanoid_sitting.support_landing_geometry.extract_scene',return_value=scene(self.floor)):
            r=o.update(.02,self.q,self.g,np.zeros(3),np.ones((96,128)))
        self.assertEqual(r['source'],'seat_from_observed_floor')
        self.assertAlmostEqual(r['clearance_m'],float((o.model.vertices['pelvis']@R.T)[:,2].min()-height),places=10)

if __name__=='__main__':unittest.main()
