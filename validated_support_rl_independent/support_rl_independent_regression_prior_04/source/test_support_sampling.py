"""Catch cross-robot resets, stale sensor history, and early-evaluation inflation."""
import unittest
from collections import deque
from types import SimpleNamespace
import torch
from test_support_readiness import ReadinessTests
from humanoid_sitting.support_sampling import (ResettableSensorInterface,
    reset_controller_rows, prime_contact_history, episode_boundaries)


class SamplingTests(unittest.TestCase):
    def test_sensor_reset_isolated_and_no_old_episode_sample(self):
        fixtures=ReadinessTests(); robot=fixtures.robot(3); e=fixtures.estimate()
        test=ResettableSensorInterface(robot,[e]*3,'cpu',delay_steps=2,seed=51)
        reference=ResettableSensorInterface(robot,[e]*3,'cpu',delay_steps=2,seed=51)
        previous=torch.zeros(3,23)
        for i in range(4):
            robot.data.joint_pos.torch.fill_(i)
            robot.data.applied_torque.torch.fill_(100)
            torch.testing.assert_close(test.read(previous),reference.read(previous),rtol=0,atol=0)
        robot.data.joint_pos.torch[1]=30
        robot.data.joint_vel.torch[1]=7
        test.reset_rows(torch.tensor([1]),[dict(valid=False)])
        packet=test.read(previous); control=reference.read(previous)
        torch.testing.assert_close(packet[[0,2]],control[[0,2]],rtol=0,atol=0)
        self.assertLess((packet[1,:23]-30).abs().max(),.01)
        self.assertLess((packet[1,23:46]-7).abs().max(),.05)
        self.assertLess(packet[1,46:69].abs().max(),1.)
        self.assertEqual(float(test.torque[1].abs().sum()),0.)
        self.assertTrue(torch.equal(packet[1,98:],torch.zeros(11)))
        for frame in test.buffer:
            self.assertLess((frame[0][1]-30).abs().max(),.01)

    def test_controller_reset_does_not_touch_other_robots(self):
        state=torch.randn(1,3,128); before=state.clone()
        controller=SimpleNamespace(contact_state=torch.randn(1,3,64),last_contact=torch.randn(3,2),
            last_rate=torch.randn(3),contact_window=deque([torch.randn(3,1,109) for _ in range(4)]))
        others=[controller.contact_state[:,[0,2]].clone(),controller.last_contact[[0,2]].clone(),
                controller.last_rate[[0,2]].clone()]
        history=[f.clone() for f in controller.contact_window]
        reset_controller_rows(controller,state,torch.tensor([1]))
        torch.testing.assert_close(state[:,[0,2]],before[:,[0,2]],rtol=0,atol=0)
        self.assertEqual(float(state[:,1].abs().sum()),0.)
        for value,old in zip((controller.contact_state[:,[0,2]],controller.last_contact[[0,2]],controller.last_rate[[0,2]]),others):
            torch.testing.assert_close(value,old,rtol=0,atol=0)
        packet=torch.randn(3,109); prime_contact_history(controller,packet,torch.tensor([False,True,False]))
        for frame,old in zip(controller.contact_window,history):
            torch.testing.assert_close(frame[[0,2]],old[[0,2]],rtol=0,atol=0)
            torch.testing.assert_close(frame[1,0],packet[1],rtol=0,atol=0)

    def test_failure_resets_training_only_and_horizon_resets_both(self):
        alive=torch.tensor([True,False,False,True]); ready=torch.tensor([True,True,False,True])
        age=torch.tensor([50,100,1,1000])
        self.assertEqual(episode_boundaries(alive,ready,age,1000,True).tolist(),[False,True,True,True])
        self.assertEqual(episode_boundaries(alive,ready,age,1000,False).tolist(),[False,False,False,True])


if __name__=='__main__': unittest.main()
