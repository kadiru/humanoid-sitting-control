import sys
from pathlib import Path
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'sitting_task/src'))
from humanoid_sitting.resting_bounded_control import bounded_command, MeasuredRestHold


class Tests(unittest.TestCase):
    def test_zero_action_retains_reference_exactly(self):
        r=torch.tensor([[.123,-.37]]);p=r.clone()
        for _ in range(1000):p=bounded_command(r,p,torch.zeros_like(p),r-1,r+1,.05)
        self.assertTrue(torch.equal(r,p))

    def test_bias_is_bounded_and_zero_returns_to_reference(self):
        r=torch.tensor([[.123,-.37]]);p=r.clone()
        for _ in range(2000):
            new=bounded_command(r,p,torch.full_like(p,100.),r-1,r+1,.05)
            self.assertLessEqual(float((new-p).abs().max()),.0020001);p=new
        self.assertLessEqual(float((p-r).abs().max()),.050001)
        for _ in range(1000):p=bounded_command(r,p,torch.zeros_like(p),r-1,r+1,.05)
        self.assertTrue(torch.allclose(r,p,atol=1.e-6))

    def inputs(self,n=2):
        return dict(effort=torch.full((n,),.5),qd=torch.zeros(n,23),gyro=torch.zeros(n,3),
            gravity=torch.tensor([[0.,0.,-1.]]).expand(n,-1),clearance=torch.zeros(n),
            geometry_valid=torch.ones(n,dtype=torch.bool),age=torch.zeros(n))

    def test_low_effort_without_geometry_does_not_freeze(self):
        g=MeasuredRestHold(2,'cpu');x=self.inputs();x['geometry_valid'][:]=False
        for _ in range(200):g.update(**x)
        self.assertFalse(g.held.any())

    def test_hold_requires_dwell_and_releases_with_motion(self):
        g=MeasuredRestHold(2,'cpu');x=self.inputs();g.filtered_effort[:]=.5
        for _ in range(99):g.update(**x)
        self.assertFalse(g.held.any());g.update(**x);self.assertTrue(g.held.all())
        x['gyro'][0,0]=.2;g.update(**x)
        self.assertFalse(g.held[0]);self.assertTrue(g.held[1])
        g.reset(torch.tensor([1]),torch.ones(1),x['gravity'][1:])
        self.assertFalse(g.held.any());self.assertEqual(int(g.entries[0]),1)


if __name__=='__main__':unittest.main()
