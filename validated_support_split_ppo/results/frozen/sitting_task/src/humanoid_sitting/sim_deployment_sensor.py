"""Simulation sensor emulation. Only physical sensor equivalents are read.

No simulator pose, linear velocity, COM, contact force, geometry bank, or clock.
Seat estimates must be supplied by an independent RGB-D/LiDAR pipeline.
"""
from collections import deque
import torch
from .deployment import SensorPacket,pack,BODY_JOINT_NAMES,seat_vector

class SimSensorInterface:
    def __init__(self,robot,estimates,device,delay_steps=1,seed=901,position_error=0.,height_error=0.):
        if delay_steps < 0:raise ValueError('Sensor delay cannot be negative')
        if not all(e.get('valid',False) for e in estimates):
            raise ValueError('Unobservable sitting area: refuse the sit command; no simulator geometry fallback')
        self.robot=robot;self.ids=[robot.joint_names.index(n) for n in BODY_JOINT_NAMES];self.device=device
        self.geometry=torch.tensor([seat_vector(e) for e in estimates],device=device,dtype=torch.float32)
        generator=torch.Generator(device=device);generator.manual_seed(seed);self.generator=generator
        self.geometry[:,:3]+=self.noise(self.geometry[:,:3],position_error)
        self.geometry[:,3]+=self.noise(self.geometry[:,3],height_error)
        self.delay_steps=delay_steps;self.buffer=deque(maxlen=delay_steps+1);self.torque=torch.zeros(len(estimates),23,device=device)
    def noise(self,x,std):return torch.randn(x.shape,device=self.device,generator=self.generator)*std
    def read(self,previous):
        d=self.robot.data
        q=d.joint_pos.torch[:,self.ids];dq=d.joint_vel.torch[:,self.ids]
        self.torque=.8*self.torque+.2*d.applied_torque.torch[:,self.ids]
        gyro=d.root_ang_vel_b.torch;gravity=d.projected_gravity_b.torch
        measured=(q+self.noise(q,.001),dq+self.noise(dq,.01),self.torque+self.noise(self.torque,.25),gyro+self.noise(gyro,.002),gravity+self.noise(gravity,.002))
        self.buffer.append(tuple(x.clone() for x in measured))
        delayed=self.buffer[0]
        return pack(SensorPacket(*delayed,previous[:,self.ids],self.geometry,torch.ones(len(q),1,device=self.device)))
