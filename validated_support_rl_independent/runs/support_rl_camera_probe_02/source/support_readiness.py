"""Sensor-derived sitting admission; unknown geometry means hold, not fallback."""
from collections import deque
import torch
from .deployment import BODY_JOINT_NAMES, seat_vector
from .sim_deployment_sensor import SimSensorInterface


def startup_measurement(estimator, *args):
    try:
        return dict(valid=True, **estimator(*args))
    except ValueError as error:
        return dict(valid=False, reason=str(error))


def admitted_targets(nominal, previous, ready):
    return torch.where(ready[:, None], nominal, previous)


def admitted_corrections(actions, ready):
    return torch.where(ready[:, None], actions, torch.zeros_like(actions))


class AdmissionSensorInterface(SimSensorInterface):
    """Same emulation as the old interface, with explicit refused commands.

    Valid observations are numerically unchanged, including RNG consumption.
    Unknown seat channels are zero and the sit-command channel is zero. The
    caller must enforce the sensor-derived admission at the action boundary.
    """
    def __init__(self, robot, estimates, device, delay_steps=1, seed=901):
        if delay_steps < 0:
            raise ValueError('Sensor delay cannot be negative')
        self.robot=robot; self.ids=[robot.joint_names.index(n) for n in BODY_JOINT_NAMES]
        self.device=device
        self.geometry=torch.tensor([seat_vector(e) for e in estimates],device=device,dtype=torch.float32)
        self.ready=torch.tensor([e['valid'] for e in estimates],device=device,dtype=torch.bool)
        self.generator=torch.Generator(device=device); self.generator.manual_seed(seed)
        # Preserve the old emulator's random stream even with zero geometry noise.
        self.geometry[:,:3]+=self.noise(self.geometry[:,:3],0.)
        self.geometry[:,3]+=self.noise(self.geometry[:,3],0.)
        self.delay_steps=delay_steps; self.buffer=deque(maxlen=delay_steps+1)
        self.torque=torch.zeros(len(estimates),23,device=device)

    def read(self, previous):
        packet=super().read(previous)
        packet[:,-1]=self.ready.float()
        return packet
