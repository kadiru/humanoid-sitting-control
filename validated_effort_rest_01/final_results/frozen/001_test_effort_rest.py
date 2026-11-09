import unittest
import torch
from humanoid_sitting.effort_rest import reference_command, effort_reward


class EffortRestTests(unittest.TestCase):
    def test_bias_cannot_accumulate_and_zero_returns(self):
        reference=torch.tensor([[.2,-.4,.8]])
        previous=reference.clone();span=torch.tensor([.04,.15,.3])
        lower=torch.full_like(reference,-2.);upper=-lower
        for _ in range(1500):
            previous=reference_command(reference,previous,torch.full_like(previous,10),lower,upper,span)
        self.assertTrue(torch.all(previous-reference<=span+1.e-6))
        self.assertTrue(torch.allclose(previous,reference+span,atol=1.e-5))
        for _ in range(1500):
            previous=reference_command(reference,previous,torch.zeros_like(previous),lower,upper,span)
        self.assertTrue(torch.allclose(previous,reference,atol=1.e-6))

    def test_exact_zero_and_slew(self):
        r=torch.zeros(2,23);span=torch.full((23,),.15);lo=torch.full_like(r,-2.)
        self.assertTrue(torch.equal(reference_command(r,r,r,lo,-lo,span),r))
        q=reference_command(r,r,torch.ones_like(r),lo,-lo,span)
        self.assertLessEqual(float(q.abs().max()),.002001)

    def test_reward_directly_prefers_lower_total_torque(self):
        tau=torch.tensor([[1.,1.],[2.,2.],[4.,4.]])
        r,c=effort_reward(tau,torch.full((3,),8.),torch.ones(3,dtype=torch.bool),torch.zeros(3,dtype=torch.bool),torch.zeros(3,23))
        self.assertTrue(torch.equal(c,torch.tensor([2.,8.,32.])))
        self.assertTrue(bool(r[0]>r[1]>r[2]>0))

    def test_no_savings_from_falling_or_unsupported(self):
        r,_=effort_reward(torch.zeros(2,37),torch.ones(2),torch.zeros(2,dtype=torch.bool),torch.tensor([False,True]),torch.zeros(2,23))
        self.assertEqual(r.tolist(),[0.,-5.])

if __name__=='__main__':unittest.main()
