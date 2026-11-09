"""Strict pure-tensor deployment contract: no simulator imports or reference code."""
from dataclasses import dataclass
import torch
from torch import nn

BODY_JOINT_NAMES=['left_hip_pitch_joint','right_hip_pitch_joint','torso_joint','left_hip_roll_joint','right_hip_roll_joint','left_shoulder_pitch_joint','right_shoulder_pitch_joint','left_hip_yaw_joint','right_hip_yaw_joint','left_shoulder_roll_joint','right_shoulder_roll_joint','left_knee_joint','right_knee_joint','left_shoulder_yaw_joint','right_shoulder_yaw_joint','left_ankle_pitch_joint','right_ankle_pitch_joint','left_elbow_pitch_joint','right_elbow_pitch_joint','left_ankle_roll_joint','right_ankle_roll_joint','left_elbow_roll_joint','right_elbow_roll_joint']
# Torso -> waist yaw and elbow_roll -> wrist_roll require physical frame/sign checks.
SDK_WIRE_SLOTS=[0,6,12,1,7,15,22,2,8,16,23,3,9,17,24,4,10,18,25,5,11,19,26]

@dataclass(frozen=True)
class SensorPacket:
    joint_position:torch.Tensor
    joint_velocity:torch.Tensor
    estimated_motor_torque:torch.Tensor
    imu_gyro:torch.Tensor
    imu_gravity:torch.Tensor
    previous_joint_target:torch.Tensor
    seat_estimate:torch.Tensor  # initial level-body XYZ, floor height, depth,width,sin/cos heading,confidence,valid
    sit_command:torch.Tensor

def pack(packet:SensorPacket):
    fields=[packet.joint_position,packet.joint_velocity,packet.estimated_motor_torque,packet.imu_gyro,packet.imu_gravity,packet.previous_joint_target,packet.seat_estimate,packet.sit_command]
    expected=[23,23,23,3,3,23,10,1]
    for x,d in zip(fields,expected):
        if x.shape[-1]!=d or not torch.isfinite(x).all():raise ValueError('Invalid sensor packet')
    return torch.cat(fields,dim=-1)  # 109 channels, zero foot-feedback channels

class DeploymentStudent(nn.Module):
    def __init__(self,inputs=109,outputs=23,hidden=128,variant='full'):
        super().__init__();self.variant=variant
        if variant not in ('full','geometry_recurrent'):raise ValueError(variant)
        self.register_buffer('mean',torch.zeros(inputs));self.register_buffer('std',torch.ones(inputs));self.register_buffer('target_mean',torch.zeros(outputs));self.register_buffer('target_std',torch.ones(outputs))
        self.encoder=nn.Sequential(nn.Linear(11 if variant=='geometry_recurrent' else inputs,128),nn.SiLU());self.rnn=nn.GRU(128,hidden,batch_first=True);self.decoder=nn.Linear(hidden,outputs)
        if variant=='geometry_recurrent':
            self.ankle_head=nn.Sequential(nn.Linear(inputs+hidden,128),nn.SiLU(),nn.Linear(128,64),nn.SiLU(),nn.Linear(64,2))
            # Long recurrent memory permits learning the slow demonstration. No
            # explicit clock or externally authored phase schedule is supplied.
            with torch.no_grad():self.rnn.bias_ih_l0[hidden:2*hidden].fill_(4.)
    def forward(self,x,state=None):
        normalized=((x-self.mean)/self.std).clamp(-20,20)
        features=self.encoder(normalized[...,98:] if self.variant=='geometry_recurrent' else normalized)
        latent,state=self.rnn(features,state);prediction=self.decoder(latent)
        if self.variant=='geometry_recurrent':
            prediction=prediction.clone();prediction[...,15:17]=self.ankle_head(torch.cat((normalized,latent),-1))
        return prediction*self.target_std+self.target_mean,state
    @classmethod
    def load(cls,path,device='cpu'):
        d=torch.load(path,map_location=device,weights_only=False);m=cls(**d['architecture']).to(device);m.load_state_dict(d['state_dict']);m.eval();return m,d

def seat_vector(est):
    import math
    if not est.get('valid',False):return [0.]*10
    return est['position_initial_body_level']+[est['surface_center'][2],est['depth'],est['width'],math.sin(est['heading']),math.cos(est['heading']),est['confidence'],1.]
