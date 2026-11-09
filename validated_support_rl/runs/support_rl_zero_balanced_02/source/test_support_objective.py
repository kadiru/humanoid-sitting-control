"""Reward loophole checks, not an imitation of the objective implementation."""
import unittest
import torch
from humanoid_sitting.support_objective import support_objective


class ObjectiveTests(unittest.TestCase):
    def sample(self):
        m = {k: torch.tensor([v]) for k, v in dict(gravity_z=-1., speed=0., angular_speed=0.,
            support=.75, goal_error2=0., tilt2=0., leg_effort2=.1, effort2=.1,
            peak_force_ratio=.75, vz=0.).items()}
        m.update({k: torch.tensor([v]) for k, v in dict(inside=True, height_ok=True, failure=False).items()})
        return m

    def reward(self, m, alive=True):
        return support_objective(m, torch.zeros(1, 23), torch.zeros(1, 23), torch.tensor([alive]))

    def test_free_fall_cannot_collect_economy(self):
        seated = self.sample(); falling = self.sample()
        falling.update(support=torch.tensor([0.]), leg_effort2=torch.tensor([0.]), vz=torch.tensor([-.7]), speed=torch.tensor([.7]))
        r, good, terms = self.reward(falling)
        self.assertFalse(good.item()); self.assertEqual(terms['leg_economy'].item(), 0.)
        self.assertLess(r.item(), self.reward(seated)[0].item())

    def test_load_on_wrong_part_cannot_collect_hold(self):
        m = self.sample(); m['failure'][:] = True
        reward, good, terms = self.reward(m)
        self.assertFalse(good.item()); self.assertEqual(terms['hold'].item(), 0.)
        self.assertLess(reward.item(), 0.)

    def test_economy_only_when_supported_and_upright(self):
        costly = self.sample(); cheap = self.sample(); costly['leg_effort2'][:] = 2.
        self.assertGreater(self.reward(cheap)[0].item(), self.reward(costly)[0].item())
        cheap['gravity_z'][:] = -.6
        self.assertEqual(self.reward(cheap)[2]['leg_economy'].item(), 0.)

    def test_failed_episode_cannot_recover_reward(self):
        reward, good, _ = self.reward(self.sample(), False)
        self.assertEqual(reward.item(), 0.); self.assertFalse(good.item())

    def test_foot_unloading_is_not_all_or_nothing(self):
        m = self.sample(); _, good, _ = self.reward(m)
        self.assertTrue(good.item())  # 25% of static load may remain on the feet.


if __name__ == '__main__':
    unittest.main()
