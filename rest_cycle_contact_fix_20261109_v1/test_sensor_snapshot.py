"""Producer-buffer reuse must never rewrite a timestamped control packet."""
import importlib.util
import unittest
import numpy as np
from contact_gate import Packet
from cycle_executor import Measurements


class FlagChecks(unittest.TestCase):
    def test_unknown_flags_do_not_become_true(self):
        for flag in (float('nan'),float('inf'),'False',2,None):
            with self.subTest(flag=flag):
                with self.assertRaisesRegex(ValueError,'BOOLEAN_SENSOR_FLAG'):
                    Packet(0,(0.,)*23,0.,0.,flag,0.,True)
                with self.assertRaisesRegex(ValueError,'BOOLEAN_SENSOR_FLAG'):
                    Measurements(0,(0.,)*23,(0.,0.,-1.),0.,True,0.,True,flag)
    def test_packet_clock_cannot_be_a_mutable_or_fractional_scalar(self):
        for step in (-1,1.,True):
            with self.assertRaisesRegex(ValueError,'INTEGER_PHYSICS_STEP'):
                Packet(step,(0.,)*23,0.,0.,True,0.,True)


@unittest.skipIf(importlib.util.find_spec('torch') is None,'Torch needed for actual producer-buffer test')
class TensorProducerChecks(unittest.TestCase):
    def test_all_measurement_channels_own_their_values(self):
        import torch
        velocity=torch.zeros(23);gravity=torch.tensor([0.,0.,-1.])
        scalar=torch.zeros(4);flags=torch.ones(4,dtype=torch.bool)
        m=Measurements(0,velocity,gravity,scalar[0],flags[0],scalar[1],flags[1],
                       flags[2],scalar[2],scalar[3],flags[3])
        velocity.fill_(.1);gravity.zero_();scalar.fill_(100.);flags.zero_()
        self.assertTrue(m.available());self.assertTrue(m.released());self.assertTrue(m.packet(0.).quiet())
        self.assertTrue(all(type(v) is np.float32 for v in m.raw_joint_velocity+m.gravity))
    def test_direct_gate_packet_cannot_change_after_capture(self):
        import torch
        velocity=torch.zeros(23);scalar=torch.tensor([0.,0.,0.,.06,.10])
        flags=torch.ones(3,dtype=torch.bool)
        p=Packet(0,velocity,scalar[0],scalar[1],flags[0],scalar[2],flags[1],flags[2],scalar[3],scalar[4])
        velocity.fill_(.1);scalar.fill_(1.);flags.zero_()
        self.assertTrue(p.quiet());self.assertTrue(p.loaded())
    def test_reused_zero_buffer_cannot_erase_an_ankle_failure(self):
        import torch
        velocity=torch.zeros(23);velocity[-1]=.055
        p=Packet(0,velocity,0.,0.,True,0.,True)
        velocity.zero_()
        self.assertFalse(p.quiet());self.assertGreater(p.raw_joint_velocity[-1],.05)
    def test_original_float32_threshold_comparisons_are_preserved(self):
        import torch
        p=Packet(0,torch.full((23,),.05),torch.tensor(0.),torch.tensor(.02),True,torch.tensor(.21),
                 True,True,torch.tensor(.05),torch.tensor(.09))
        self.assertTrue(p.quiet());self.assertTrue(p.loaded())
        m=Measurements(0,torch.zeros(23),torch.tensor([0.,0.,-1.]),0.,True,0.,True,
                       True,torch.tensor(0.),torch.tensor(.01),True)
        self.assertFalse(m.released())  # strict upper <.01, not <=


if __name__=='__main__':unittest.main()
