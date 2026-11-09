import unittest
import torch
from test_resting_objective import RestingObjectiveTests
from humanoid_sitting.resting_reward import calibrated_rest_reward


class CalibratedRewardTests(unittest.TestCase):
    def test_all_heights_keep_a_non_saturated_effort_signal(self):
        sample=RestingObjectiveTests().sample(); alive=torch.tensor([True]); previous=None
        for effort in (.0001,.001,.01,.1,1.):
            sample['torque'][:]=60*effort**.5
            reward,good,d=calibrated_rest_reward(sample,alive,.005)
            self.assertTrue(bool(good)); self.assertGreater(float(reward),.02)
            if previous is not None:self.assertLess(float(reward),previous)
            previous=float(reward)

    def test_control_arm_and_failure_are_unchanged(self):
        sample=RestingObjectiveTests().sample(); alive=torch.tensor([True])
        a=calibrated_rest_reward(sample,alive,.005,0.)
        b=calibrated_rest_reward(sample,alive,.005,1.)
        self.assertTrue(torch.equal(a[1],b[1]))
        self.assertAlmostEqual(float(b[0]-a[0]),float(b[2]['effort_cost']*.02))
        sample['failure'][:]=True
        self.assertEqual(float(calibrated_rest_reward(sample,alive,.005)[0]),-5.)
        self.assertEqual(float(calibrated_rest_reward(sample,~alive,.005)[0]),0.)


if __name__=='__main__':unittest.main()
