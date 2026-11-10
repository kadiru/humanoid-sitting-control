"""Analytic rigid-body kinematics; no simulation and no fitted parameters."""
import sys
import os
from pathlib import Path
import unittest
import numpy as np
ROOT=Path(os.environ.get('RESTING_PROJECT_ROOT',Path(__file__).resolve().parents[1]))
EXPERIMENTS=ROOT/'experiments' if (ROOT/'experiments').exists() else ROOT
PACKAGE=ROOT/'sitting_task/src' if (ROOT/'sitting_task/src').exists() else ROOT/'ubuntu/sitting_task/src'
sys.path[:0]=[str(EXPERIMENTS/'rest_cycle_load_validation_20261109_v1'),str(PACKAGE)]
from dynamic_load import shift_acceleration


class PointAccelerationTests(unittest.TestCase):
    def test_uniform_translation(self):
        np.testing.assert_array_equal(shift_acceleration(np.array([1.,2.,3.]),np.zeros(3),np.zeros(3),
                                                         np.array([.2,-.8,.1])),[1.,2.,3.])

    def test_radial_and_tangential_acceleration(self):
        # r=2m, omega=3rad/s, alpha=4rad/s² about z: (-18,8,0)m/s².
        np.testing.assert_allclose(shift_acceleration(np.zeros(3),np.array([0.,0.,3.]),
            np.array([0.,0.,4.]),np.array([2.,0.,0.])),[-18.,8.,0.])

    def test_round_trip_point_shift(self):
        a=np.array([1.,-2.,.3]);omega=np.array([.1,.2,-.4]);alpha=np.array([-.3,.9,.2]);r=np.array([.2,.5,-.1])
        shifted=shift_acceleration(a,omega,alpha,r)
        np.testing.assert_allclose(shift_acceleration(shifted,omega,alpha,-r),a,atol=1e-14)


if __name__=='__main__':unittest.main()
