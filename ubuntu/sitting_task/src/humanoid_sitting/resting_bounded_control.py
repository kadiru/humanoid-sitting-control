"""Sensor-side bounded posture targets and an optional measured-effort hold gate."""
import torch


def bounded_command(reference, previous, action, lower, upper, span, alpha=.2, slew=.002):
    """Zero action means the reference; persistent bias cannot integrate away."""
    assert torch.isfinite(action).all()
    lo=torch.maximum(reference-span,lower+.03)
    hi=torch.minimum(reference+span,upper-.03)
    if not bool(((reference>=lo)&(reference<=hi)).all()):
        raise ValueError('Starting command does not have the required joint margin')
    target=(reference+span*action.clamp(-1,1)).clamp(min=lo,max=hi)
    return (previous+(alpha*(target-previous)).clamp(-slew,slew)).clamp(min=lo,max=hi)


class MeasuredRestHold:
    """Holds PD targets, never turns motors off. No simulation-contact input.

    Entry: measured effort <=90% of settled starting effort and two seconds of
    low joint/IMU motion with a fresh nearby depth plane. Hysteresis releases on
    renewed motion, lost geometry, changed attitude, or effort above105%.
    These sensor gates propose a hold; physical evaluation still decides success.
    """
    def __init__(self,n,device,ratio=.9):
        self.ratio=ratio
        self.reference_effort=torch.ones(n,device=device)
        self.filtered_effort=self.reference_effort.clone()
        self.reference_gravity=torch.zeros(n,3,device=device)
        self.reference_gravity[:,2]=-1
        self.ticks=torch.zeros(n,dtype=torch.long,device=device)
        self.held=torch.zeros(n,dtype=torch.bool,device=device)
        self.entries=torch.zeros_like(self.ticks);self.releases=torch.zeros_like(self.ticks)

    def reset(self,ids,effort,gravity):
        self.reference_effort[ids]=effort.clamp_min(1.e-8)
        self.filtered_effort[ids]=effort
        self.reference_gravity[ids]=gravity
        self.ticks[ids]=0;self.held[ids]=False
        self.entries[ids]=0;self.releases[ids]=0

    def update(self,effort,qd,gyro,gravity,clearance,geometry_valid,age):
        self.filtered_effort=.96*self.filtered_effort+.04*effort
        joint_speed=qd.abs().max(1).values
        gyro_speed=gyro.norm(dim=1)
        alignment=(gravity*self.reference_gravity).sum(1)
        enter=(self.filtered_effort<=self.ratio*self.reference_effort)&(joint_speed<.08)&(gyro_speed<.06)&(alignment>.996)
        enter &= geometry_valid&(age<=.12)&(clearance.abs()<.015)
        release=(joint_speed>.2)|(gyro_speed>.12)|(alignment<.99)|~geometry_valid|(age>.2)|(clearance.abs()>.025)
        release |= self.filtered_effort>1.05*self.reference_effort
        old=self.held.clone()
        self.held &= ~release
        self.ticks=torch.where(enter,self.ticks+1,torch.zeros_like(self.ticks))
        self.held |= self.ticks>=100
        self.entries+=(self.held&~old)
        self.releases+=(old&~self.held)
        return self.held
