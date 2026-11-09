import unittest
import torch
from humanoid_sitting.two_surface_path import sample_path,bounded_path_command

class PathTests(unittest.TestCase):
    def test_endpoints_and_no_extrapolation(self):
        values=torch.tensor([[[0.,1.],[1.,3.],[3.,7.]]]).repeat(4,1,1)
        actual=sample_path(values,torch.tensor([-2.,0.,2.,9.]))
        self.assertTrue(torch.equal(actual,torch.tensor([[0.,1.],[0.,1.],[3.,7.],[3.,7.]])))
    def test_continuous_at_waypoint(self):
        values=torch.tensor([[[0.],[1.],[3.]]]).repeat(3,1,1)
        result=sample_path(values,torch.tensor([.9999,1.,1.0001]))
        self.assertLess(float((result-result[1]).abs().max()),1.e-6)
    def test_zero_action_retains_and_returns(self):
        reference=torch.tensor([[.2,-.3]]);lower=torch.full_like(reference,-2.);upper=-lower;span=.025
        out=bounded_path_command(reference,reference,torch.zeros_like(reference),lower,upper,span)
        self.assertTrue(torch.equal(reference,out))
        changed=reference+.02
        for _ in range(200):changed=bounded_path_command(reference,changed,torch.zeros_like(reference),lower,upper,span)
        self.assertLess(float((changed-reference).abs().max()),1.e-6)
    def test_bias_bounded_and_slew_limited(self):
        reference=torch.zeros(1,2);previous=reference.clone();lower=torch.full_like(reference,-2.);upper=-lower
        for _ in range(1000):
            value=bounded_path_command(reference,previous,torch.full_like(previous,100.),lower,upper,.025)
            self.assertLessEqual(float(abs(value-previous).max()),.002001)
            self.assertLessEqual(float(abs(value-reference).max()),.025001)
            previous=value
if __name__=='__main__':unittest.main()
