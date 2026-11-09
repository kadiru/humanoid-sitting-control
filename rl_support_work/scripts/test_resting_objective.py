"""Behavioral loopholes for the isolated resting objective; no simulator needed."""
from dataclasses import replace
import math
import unittest
import torch
from humanoid_sitting.resting_objective import RestingObjectiveCfg, resting_objective


class RestingObjectiveTests(unittest.TestCase):
    def sample(self):
        # 300 N body: 210 N on the seat, 90 N on its feet.
        return dict(torque=torch.full((1, 23), 10.), torque_scale=torch.full((23,), 60.),
            joint_velocity=torch.zeros(1, 23),
            force_w=torch.tensor([[[0., 0., 210.], [0., 0., 90.]]]),
            normal_w=torch.tensor([[[0., 0., 1.], [0., 0., 1.]]]),
            friction=torch.full((1, 2), .6), permitted=torch.ones(1, 2, dtype=torch.bool),
            rest_patch=torch.tensor([[True, False]]), inside_patch=torch.ones(1, 2, dtype=torch.bool),
            slip_m_s=torch.zeros(1, 2), bodyweight_N=torch.tensor([300.]),
            speed_m_s=torch.zeros(1), angular_speed_rad_s=torch.zeros(1),
            trunk_tilt_rad=torch.zeros(1), joint_margin_ok=torch.tensor([True]),
            failure=torch.tensor([False]))

    def call(self, m, cfg=RestingObjectiveCfg(), alive=True):
        return resting_objective(m, torch.tensor([alive]), cfg)

    def test_stationary_holding_torque_is_still_effort(self):
        m=self.sample(); costly=self.call(m)
        m['torque'] *= .5; cheap=self.call(m)
        self.assertEqual(float(costly[2]['absolute_mechanical_power_W']), 0.)
        self.assertAlmostEqual(float(cheap[2]['normalized_torque_squared'] /
                                     costly[2]['normalized_torque_squared']), .25)
        self.assertGreater(float(cheap[0]), float(costly[0]))

    def test_back_lean_can_keep_all_weight_on_feet(self):
        m=self.sample()
        m['normal_w'][0, 0]=torch.tensor([1., 0., 0.])
        m['force_w'][0]=torch.tensor([[60., 0., 0.], [-60., 0., 300.]])
        m['trunk_tilt_rad'][:]=math.radians(20)
        reward, good, d=self.call(m)
        self.assertTrue(bool(good)); self.assertGreater(float(reward), 0.)
        self.assertEqual(float(d['rest_vertical_load_bw']), 0.)
        self.assertAlmostEqual(float(d['rest_normal_load_bw']), .2)

    def test_oblique_contact_is_measured_in_surface_frame(self):
        m=self.sample(); a=math.radians(45)
        n=torch.tensor([math.sin(a), 0., math.cos(a)])
        m['normal_w'][0, 0]=n; m['force_w'][0, 0]=100*n
        m['force_w'][0, 1]=torch.tensor([-100*math.sin(a), 0., 300-100*math.cos(a)])
        self.assertTrue(bool(self.call(m)[1]))

    def test_standing_or_falling_does_not_earn_rest_reward(self):
        m=self.sample(); m['torque'].zero_(); m['force_w'][0, 0].zero_()
        self.assertFalse(bool(self.call(m)[1])); self.assertEqual(float(self.call(m)[0]), 0.)
        m=self.sample(); m['speed_m_s'][:]=1.
        self.assertEqual(float(self.call(m)[0]), 0.)

    def test_wrong_contact_slip_and_traction_cannot_earn_rest(self):
        for condition in ('forbidden', 'outside', 'slip', 'friction', 'joint_margin', 'tilt'):
            m=self.sample()
            if condition=='forbidden': m['permitted'][0, 0]=False
            if condition=='outside': m['inside_patch'][0, 0]=False
            if condition=='slip': m['slip_m_s'][0, 0]=.08
            if condition=='friction': m['force_w'][0, 1, 0]=100.
            if condition=='joint_margin': m['joint_margin_ok'][:]=False
            if condition=='tilt': m['trunk_tilt_rad'][:]=math.radians(80)
            self.assertFalse(bool(self.call(m)[1]), condition)

    def test_all_joints_count_and_signed_torques_do_not_cancel(self):
        m=self.sample(); m['torque'].zero_(); low=self.call(m)
        m['torque'][0, -2:]=torch.tensor([30., -30.]); high=self.call(m)
        self.assertGreater(float(high[2]['normalized_torque_squared']), 0.)
        self.assertLess(float(high[0]), float(low[0]))

    def test_control_arm_differs_only_in_effort_cost(self):
        m=self.sample(); control=self.call(m, replace(RestingObjectiveCfg(), effort_weight=0.))
        effort=self.call(m)
        self.assertTrue(torch.equal(control[1], effort[1]))
        for key in control[2]:
            if key!='effort_cost': self.assertTrue(torch.equal(control[2][key], effort[2][key]), key)
        self.assertAlmostEqual(float(effort[0]-control[0]), float(effort[2]['effort_cost']*.02))

    def test_failure_is_latched_by_caller_and_no_future_reward(self):
        m=self.sample(); m['failure'][:]=True
        self.assertEqual(float(self.call(m)[0]), -5.)
        self.assertEqual(float(self.call(m, alive=False)[0]), 0.)

    def test_dead_robot_has_no_hold_or_economy_reward(self):
        m=self.sample(); reward, good, _=self.call(m, alive=False)
        self.assertFalse(bool(good)); self.assertEqual(float(reward), 0.)

    def test_malformed_geometry_and_missing_effort_scales_fail(self):
        m=self.sample(); m['normal_w'] *= 2.
        with self.assertRaises(ValueError): self.call(m)
        m=self.sample(); m['torque_scale'][0]=0.
        with self.assertRaises(ValueError): self.call(m)
        m=self.sample(); m['force_w'][0, 0, 0]=float('nan')
        with self.assertRaises(ValueError): self.call(m)


if __name__=='__main__': unittest.main()
