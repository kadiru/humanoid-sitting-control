"""Unknown geometry must not become a sit command or a fabricated measurement."""
import unittest
from types import SimpleNamespace
import torch
from humanoid_sitting.deployment import BODY_JOINT_NAMES
from humanoid_sitting.sim_deployment_sensor import SimSensorInterface
from humanoid_sitting.support_readiness import (startup_measurement,
    AdmissionSensorInterface, admitted_targets, admitted_corrections)


class ReadinessTests(unittest.TestCase):
    def estimate(self):
        return dict(valid=True,position_initial_body_level=[-.4,0.,-.4],surface_center=[-.4,0.,.3],depth=.5,width=.9,heading=0.,confidence=1.)

    def robot(self,n=1):
        wrap=lambda t: SimpleNamespace(torch=t)
        data=SimpleNamespace(joint_pos=wrap(torch.zeros(n,23)),joint_vel=wrap(torch.zeros(n,23)),
            applied_torque=wrap(torch.zeros(n,23)),root_ang_vel_b=wrap(torch.zeros(n,3)),
            projected_gravity_b=wrap(torch.tensor([[0.,0.,-1.]]).repeat(n,1)))
        return SimpleNamespace(joint_names=BODY_JOINT_NAMES,data=data)

    def test_unobservable_estimator_has_no_numeric_fallback(self):
        def missing(*args): raise ValueError('Floor hidden')
        result=startup_measurement(missing,None)
        self.assertEqual(result,dict(valid=False,reason='Floor hidden'))

    def test_invalid_geometry_cannot_issue_sit(self):
        sensor=AdmissionSensorInterface(self.robot(),[dict(valid=False)],'cpu')
        packet=sensor.read(torch.zeros(1,23))
        self.assertTrue(torch.equal(packet[:,98:],torch.zeros(1,11)))

    def test_refused_actions_hold_previous_target(self):
        ready=torch.tensor([False,True]); previous=torch.randn(2,23)
        nominal=torch.randn(2,23); actions=torch.ones(2,23)*100
        target=admitted_targets(nominal,previous,ready)+.08*admitted_corrections(actions,ready).clamp(-1,1)
        self.assertTrue(torch.equal(target[0],previous[0]))
        self.assertTrue(torch.allclose(target[1],nominal[1]+.08))

    def test_valid_sensor_stream_is_unchanged(self):
        robot=self.robot(); estimates=[self.estimate()]
        old=SimSensorInterface(robot,estimates,'cpu',seed=17)
        new=AdmissionSensorInterface(robot,estimates,'cpu',seed=17)
        for _ in range(3):
            self.assertTrue(torch.equal(old.read(torch.zeros(1,23)),new.read(torch.zeros(1,23))))


if __name__=='__main__':unittest.main()
