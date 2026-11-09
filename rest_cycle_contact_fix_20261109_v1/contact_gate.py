"""Fresh sensor admission at hover and first-contact boundaries.

This gate prevents a geometric schedule from asserting contact. It does not
repair or replace native velocities, validate a force estimator, or establish
physical success. Runtime integration needs a separately admitted live load
adapter and current geometry certificate before motion can be launched.
"""
from dataclasses import dataclass
import math

@dataclass(frozen=True)
class Packet:
    physics_step: int
    raw_joint_velocity: tuple
    pitch_error_rad: float
    gyro_y_rad_s: float
    geometry_valid: bool
    geometry_age_s: float
    current_pose_admitted: bool
    load_interval_admitted: bool = False
    load_lower_bw: float | None = None
    load_upper_bw: float | None = None

    def quiet(self):
        # Original raw thresholds; no finite differences, clipping or filtering.
        finite=(len(self.raw_joint_velocity)==23 and all(math.isfinite(v) for v in
            (*self.raw_joint_velocity,self.pitch_error_rad,self.gyro_y_rad_s,self.geometry_age_s)))
        return bool(finite and self.geometry_valid and 0<=self.geometry_age_s<=.21
                    and max(map(abs,self.raw_joint_velocity))<=.05
                    and abs(self.pitch_error_rad)<=math.radians(.5)
                    and abs(self.gyro_y_rad_s)<=.02)

    def loaded(self):
        lo,hi=self.load_lower_bw,self.load_upper_bw
        return bool(self.load_interval_admitted and lo is not None and hi is not None
                    and math.isfinite(lo) and math.isfinite(hi) and .05<=lo<=hi)

class ContactGate:
    def __init__(self, hover_index, contact_index):
        if not 0<hover_index<contact_index:raise ValueError('ORDERED_CONTACT_BOUNDARIES_REQUIRED')
        self.hover_index=hover_index;self.contact_index=contact_index
        self.hover_released=False;self.contact_confirmed=False
        self.hover_quiet=self.contact_quiet=0
        self.last_step=None;self.calls=0;self.aborted=False

    def update(self,index,p:Packet):
        self.calls+=1
        if self.aborted:return self.result(False,'latched_sensor_or_schedule_fault')
        if self.last_step is not None and p.physics_step!=self.last_step+4:
            self.aborted=True;return self.result(False,'stale_or_skipped_sensor_packet')
        self.last_step=p.physics_step
        if index>self.hover_index and not self.hover_released:
            self.aborted=True;return self.result(False,'schedule_skipped_hover_gate')
        if index>self.contact_index and not self.contact_confirmed:
            self.aborted=True;return self.result(False,'schedule_skipped_contact_gate')
        if self.calls>=1000 and not self.contact_confirmed:
            self.aborted=True;return self.result(False,'original20s_contact_deadline')
        valid=(p.geometry_valid and math.isfinite(p.geometry_age_s)
               and 0<=p.geometry_age_s<=.21 and p.current_pose_admitted)
        if not valid:
            self.hover_quiet=self.contact_quiet=0
            return self.result(False,'current_geometry_or_tracking_not_admitted')
        if index==self.hover_index and not self.hover_released:
            self.hover_quiet=self.hover_quiet+1 if p.quiet() else 0
            self.hover_released=self.hover_quiet>=25
            return self.result(self.hover_released,'hover_quiet' if self.hover_released else 'await_fresh_hover_quiet')
        if index==self.contact_index and not self.contact_confirmed:
            self.contact_quiet=self.contact_quiet+1 if p.quiet() and p.loaded() else 0
            self.contact_confirmed=self.contact_quiet>=25
            return self.result(self.contact_confirmed,'contact_confirmed' if self.contact_confirmed else 'await_qualified_measured_contact')
        return self.result(True,'admitted_primitive_segment')

    def result(self,advance,reason):
        return dict(advance=advance,reason=reason,aborted=self.aborted,
                    hover_quiet=self.hover_quiet,contact_quiet=self.contact_quiet,
                    hover_released=self.hover_released,contact_confirmed=self.contact_confirmed,
                    physical_success=False)
