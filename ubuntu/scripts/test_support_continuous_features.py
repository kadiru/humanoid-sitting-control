"""Rigid-motion and expiry checks for observed-patch transport."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from scipy.spatial.transform import Rotation
from humanoid_sitting.support_continuous_features import ContinuousPatchFeatures


class Model:
    vertices={'pelvis':np.array([[-.1,-.1,-.1],[.1,.1,-.1]])}
    def fk(self,q):
        result={}
        for side,x in [('left',-.1),('right',.1)]:
            T=np.eye(4);T[:3,3]=np.array([x,0.,-.8])+q
            result[side+'_ankle_roll_link']=T
        return result


class ContinuousTests(unittest.TestCase):
    def setUp(self):
        self.o=ContinuousPatchFeatures(SimpleNamespace(model=Model(),K=None,T_P_C=None))
        x,y=np.meshgrid(np.linspace(-.4,.1,10),np.linspace(-.3,.3,10))
        self.points=np.c_[x.ravel(),y.ravel(),np.full(x.size,-.3)]
        self.scene=dict(R=np.eye(3),planes=[dict(normal=np.array([0.,0.,1.]),offset=.3,points=self.points)])
        self.result=dict(valid=True,source='current_floor_and_seat',normal_P=[0,0,1],clearance_m=.2,allowance_m=.009)
        with patch('humanoid_sitting.support_continuous_features.extract_scene',return_value=self.scene):
            self.initial,_=self.o.update(0.,self.result,np.zeros(3),[0,0,-1],[0,0,0],np.ones((2,2)))

    def test_stationary_age_and_patch_shape(self):
        v,d=self.o.update(.02,self.result|dict(source='kinematic_prediction'),np.zeros(3),[0,0,-1],[0,0,0])
        np.testing.assert_array_equal(v[:9],self.initial[:9]);self.assertAlmostEqual(v[9],.02,places=7)
        np.testing.assert_allclose(d['current_probes_P'],d['anchor_probes_P'])

    def test_known_body_translation_moves_observed_patch(self):
        v,d=self.o.update(.02,self.result|dict(source='kinematic_prediction',clearance_m=.195),np.array([.002,0,.005]),[0,0,-1],[0,0,0])
        self.assertTrue(d['usable']);self.assertAlmostEqual(v[0],.195,places=7)
        np.testing.assert_allclose(np.array(d['current_probes_P'])-d['anchor_probes_P'],np.tile([.002,0,.005],(len(d['anchor_probes_P']),1)),atol=1e-14)

    def test_known_body_yaw_rotates_patch_in_current_pelvis_frame(self):
        R=Rotation.from_rotvec([0,0,.01]).as_matrix()
        poses=self.o.tracker.model.fk(np.zeros(3))
        for pose in poses.values():pose[:3,3]=pose[:3,3]@R
        with patch.object(self.o.tracker.model,'fk',return_value=poses):
            v,d=self.o.update(.02,self.result|dict(source='kinematic_prediction'),np.zeros(3),[0,0,-1],[0,0,.5])
        self.assertTrue(d['usable'])
        np.testing.assert_allclose(d['current_probes_P'],np.array(d['anchor_probes_P'])@R,atol=1e-14)

    def test_full_common_motion_spends_budget(self):
        v,d=self.o.update(.02,self.result|dict(source='kinematic_prediction',clearance_m=.17),np.array([0,0,.03]),[0,0,-1],[0,0,0])
        self.assertFalse(d['usable']);self.assertEqual(v[8],0.);self.assertEqual(d['reason'],'patch_motion_budget')

    def test_expiry_unknown_distance_and_clock(self):
        v,d=self.o.update(.11,self.result,np.zeros(3),[0,0,-1],[0,0,0]);self.assertFalse(d['usable'])
        with self.assertRaises(ValueError):self.o.update(.11,self.result,np.zeros(3),[0,0,-1],[0,0,0])
        v,d=self.o.update(.12,self.result|dict(valid=False),np.zeros(3),[0,0,-1],[0,0,0]);self.assertEqual(v[8],0.)

if __name__=='__main__':unittest.main()
