"""Independent sensor/controller resets. No simulator truth is an input."""
import torch
from .deployment import SensorPacket, pack, seat_vector
from .support_readiness import AdmissionSensorInterface


class ResettableSensorInterface(AdmissionSensorInterface):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pending = torch.ones_like(self.ready)

    def reset_rows(self, ids, estimates):
        self.geometry[ids] = torch.tensor([seat_vector(e) for e in estimates], device=self.device)
        self.ready[ids] = torch.tensor([e['valid'] for e in estimates], device=self.device)
        self.torque[ids] = 0
        self.pending[ids] = True

    def read(self, previous):
        d = self.robot.data
        q, dq = d.joint_pos.torch[:, self.ids], d.joint_vel.torch[:, self.ids]
        self.torque = .8*self.torque + .2*d.applied_torque.torch[:, self.ids]
        # Applied torque may still be cached from the preceding physics step.
        # A new episode has no torque history; its first filtered sample is zero.
        self.torque[self.pending] = 0
        gyro, gravity = d.root_ang_vel_b.torch, d.projected_gravity_b.torch
        measured = (q+self.noise(q,.001), dq+self.noise(dq,.01),
                    self.torque+self.noise(self.torque,.25),
                    gyro+self.noise(gyro,.002), gravity+self.noise(gravity,.002))
        # Fill only reset rows of the delay line with the first new sample.
        # Other robots retain their genuine delayed readings and filter state.
        for frame in self.buffer:
            for old, new in zip(frame, measured):
                old[self.pending] = new[self.pending]
        self.buffer.append(tuple(x.clone() for x in measured))
        self.pending.zero_()
        return pack(SensorPacket(*self.buffer[0], previous[:,self.ids], self.geometry,
                                 self.ready[:,None].float()))


def reset_controller_rows(controller, state, ids):
    if state is not None:
        state[:,ids] = 0
    if controller.contact_state is not None:
        controller.contact_state[:,ids] = 0
    for name in ('last_contact', 'last_rate'):
        value = getattr(controller, name)
        if value is not None:
            value[ids] = 0


def prime_contact_history(controller, packet, pending):
    """Windowed estimators repeat a new episode's first sample, as on cold start."""
    for frame in controller.contact_window:
        frame[pending] = packet[pending,None,:]


def episode_boundaries(alive, ready, lengths, horizon, training):
    """Evaluation always lasts the full horizon, even after a latched failure."""
    timeout = lengths >= horizon
    return timeout | (~alive | ~ready) if training else timeout
