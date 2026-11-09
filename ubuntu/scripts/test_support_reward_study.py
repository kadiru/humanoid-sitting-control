"""Reward loopholes, exact control-arm parity, and per-robot peak reset."""
import unittest
import torch
from humanoid_sitting.support_objective import support_objective
from humanoid_sitting.support_reward_study import SittingRewardStudy


class RewardStudyTests(unittest.TestCase):
    def sample(self):
        m = {k: torch.tensor([v, v]) for k, v in dict(gravity_z=-1., speed=0.,
            angular_speed=0., support=.75, goal_error2=0., tilt2=0., leg_effort2=.1,
            leg_rms_nm=7., effort2=.1, peak_force_ratio=.75, vz=0.).items()}
        m.update({k: torch.tensor([v, v]) for k, v in dict(inside=True, height_ok=True, failure=False).items()})
        return m

    def call(self, obj, m, alive=None, dt=.02):
        return obj(m, torch.zeros(2, 23), torch.zeros(2, 23),
                   torch.ones(2, dtype=torch.bool) if alive is None else alive, dt)

    def test_legacy_exact(self):
        m = self.sample()
        for force in (.7, 1.5, 2.5):
            m['peak_force_ratio'][:] = force
            for a, b in zip(self.call(SittingRewardStudy('legacy'), m)[:2], self.call(support_objective, m)[:2]):
                self.assertTrue(torch.equal(a, b))

    def test_peak_charge_is_duration_independent(self):
        for dt, repeats in ((.02, 1), (.01, 2), (.02, 20)):
            obj = SittingRewardStudy('gentle'); m = self.sample(); charge = 0.
            for force in (1.2, 1.5, 1.1, 1.8):
                m['peak_force_ratio'][:] = force
                for _ in range(repeats):
                    charge += self.call(obj, m, dt=dt)[2]['impact'][0].item() * dt
            self.assertAlmostEqual(charge, -12. * .8**2, places=5)

    def test_reset_is_independent_and_failure_stays_zero(self):
        obj = SittingRewardStudy('gentle'); m = self.sample(); m['peak_force_ratio'][:] = 1.5
        self.call(obj, m); obj.reset(torch.tensor([0]))
        impact = self.call(obj, m)[2]['impact'] * .02
        self.assertAlmostEqual(impact[0].item(), -3.)
        self.assertEqual(impact[1].item(), 0.)
        m['peak_force_ratio'][:] = 3.
        self.assertTrue(torch.equal(self.call(obj, m, torch.zeros(2, dtype=torch.bool))[0], torch.zeros(2)))

    def test_free_fall_and_wrong_contact_cannot_earn_economy(self):
        for field, value in (('support', 0.), ('gravity_z', -.6), ('failure', True)):
            m = self.sample(); m[field][:] = value; m['leg_rms_nm'][:] = 0.
            self.assertTrue(torch.equal(self.call(SittingRewardStudy('gentle'), m)[2]['leg_economy'], torch.zeros(2)))

    def test_lower_leg_effort_and_gentler_contact_are_preferred(self):
        m = self.sample(); m['leg_rms_nm'][:] = torch.tensor([5., 10.])
        rewards = self.call(SittingRewardStudy('gentle'), m)[0]
        self.assertGreater(rewards[0].item(), rewards[1].item())
        m['leg_rms_nm'][:] = 7.; m['peak_force_ratio'][:] = torch.tensor([1.1, 1.8])
        rewards = self.call(SittingRewardStudy('gentle'), m)[0]
        self.assertGreater(rewards[0].item(), rewards[1].item())

    def test_diagnostics_reconcile_with_actual_reward(self):
        obj = SittingRewardStudy('gentle'); m = self.sample()
        m['peak_force_ratio'][:] = 1.6; m['failure'][0] = True
        self.call(obj, m)
        t = obj.pop_totals(); total = t.pop('actual_total')
        self.assertAlmostEqual(total, sum(t.values()), places=5)


if __name__ == '__main__':
    unittest.main()
