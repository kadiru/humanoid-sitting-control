"""Invalid sensor packets must never become an available load measurement."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
from observer import SpatialLoadObserver


class InputChecks(unittest.TestCase):
    def setUp(self):
        # A deliberately well-conditioned static two-foot system isolates input
        # validity from geometric conditioning. It is not a robot validation.
        self.observer = SpatialLoadObserver.__new__(SpatialLoadObserver)
        self.observer.model = SimpleNamespace(names=list(range(12)))
        self.observer.sides = [list(range(6)), list(range(6, 12))]
        self.observer.weight = 100.
        self.observer.equations = Mock(return_value=(
            np.stack([np.eye(6), np.eye(6)]),
            np.array([[0., 0., 50., 0., 0., 0.]] * 2)))
        self.inputs = [np.zeros(12), np.zeros(12), np.array([0., 0., -1.])]

    def test_valid_static_balance_unchanged(self):
        value, condition, _, _ = self.observer.estimate(*self.inputs)
        np.testing.assert_array_equal(value[:3], np.zeros(3))
        np.testing.assert_array_equal(condition, np.ones(2))
        self.assertEqual(value[5], 50.)
        self.assertEqual(value[11], 50.)

    def test_nonfinite_sensor_is_unavailable_before_kinematics(self):
        for channel in range(3):
            for bad in (np.nan, np.inf, -np.inf):
                with self.subTest(channel=channel, value=bad):
                    inputs = [x.copy() for x in self.inputs]
                    inputs[channel][0] = bad
                    self.assertIsNone(self.observer.estimate(*inputs)[0])
        self.observer.equations.assert_not_called()

    def test_missing_or_zero_gravity_is_unavailable(self):
        for gravity in ([], [0., 0.], [0., 0., 0.], [[0., 0., -1.]]):
            with self.subTest(gravity=gravity):
                self.assertIsNone(self.observer.estimate(*self.inputs[:2], gravity)[0])
        self.observer.equations.assert_not_called()

    def test_truncated_joint_channels_are_unavailable(self):
        self.assertIsNone(self.observer.estimate(self.inputs[0][:-1], *self.inputs[1:])[0])
        self.assertIsNone(self.observer.estimate(self.inputs[0], self.inputs[1][:-1], self.inputs[2])[0])

    def test_nonfinite_equation_cannot_report_available(self):
        jac, gravity = self.observer.equations.return_value
        gravity[0, 0] = np.nan
        self.assertIsNone(self.observer.estimate(*self.inputs)[0])

    def test_original_condition_limit_unchanged(self):
        jac, _ = self.observer.equations.return_value
        jac[0, 0, 0] = 200.
        value, condition, _, _ = self.observer.estimate(*self.inputs)
        self.assertIsNone(value)
        self.assertEqual(condition[0], 200.)


if __name__ == '__main__':
    unittest.main()
