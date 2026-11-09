"""Sensor-driven rest cycle. No simulator contact, world pose or case IDs.

The primitive executor must supply measured tracking/clearance and a separately
validated load interval. A stage event is never a physical-success label.
"""
from dataclasses import dataclass
from enum import Enum
import math


class Stage(str, Enum):
    OBSERVE='observe'
    POSITION='position'
    CONTACT='contact'
    TRANSFER='transfer'
    HOLD='hold'
    UNLOAD='unload'
    STAND='stand'
    DONE='done'
    ABORT='abort'


@dataclass(frozen=True)
class Sensors:
    map_valid: bool
    tracking_ok: bool
    quiet: bool
    # Qualified rest evidence is distinct from the stricter precontact quiet gate.
    # It defaults false; only a validated sensor adapter may supply it at runtime.
    rest_stability_qualified: bool=False
    positioned: bool=False
    standing: bool=False
    clear_of_support: bool=False
    load_lower_bw: float | None=None
    load_upper_bw: float | None=None
    sensor_fault: bool=False

    def interval(self):
        lo,hi=self.load_lower_bw,self.load_upper_bw
        return (lo is not None and hi is not None and math.isfinite(lo)
                and math.isfinite(hi) and lo<=hi)


class RestCycle:
    def __init__(self):
        self.stage=Stage.OBSERVE;self.tick=0;self.entered=0
        self.quiet_ticks=self.qualified_ticks=self.hold_ticks=self.stand_ticks=0
        self.events=[{'tick':0,'stage':self.stage.value}];self.reason=None
        self.rest_window_passed=False

    def transition(self,stage,reason):
        self.stage=stage;self.entered=self.tick;self.quiet_ticks=self.qualified_ticks=0
        self.events.append({'tick':self.tick,'stage':stage.value,'reason':reason})
        if stage==Stage.ABORT:self.reason=reason

    def update(self,s: Sensors):
        if self.stage in (Stage.DONE,Stage.ABORT):return self.decision(False)
        self.tick+=1
        if s.sensor_fault or ((s.load_lower_bw is not None or s.load_upper_bw is not None) and not s.interval()):
            self.transition(Stage.ABORT,'invalid_sensor_packet');return self.decision(False)
        if self.tick>2000:
            self.transition(Stage.ABORT,'complete_cycle_deadline');return self.decision(False)
        valid=s.map_valid and s.tracking_ok
        quiet=valid and s.quiet
        self.quiet_ticks=self.quiet_ticks+1 if quiet else 0
        known=s.interval();loaded=known and s.load_lower_bw>=.05
        transferred=known and s.load_lower_bw>=.10
        unloaded=known and s.load_upper_bw<.01
        qualified={Stage.OBSERVE: quiet, Stage.POSITION: quiet and s.positioned,
                   Stage.CONTACT: quiet and loaded, Stage.TRANSFER: quiet and transferred,
                   Stage.UNLOAD: quiet and unloaded}.get(self.stage,False)
        self.qualified_ticks=self.qualified_ticks+1 if qualified else 0
        if self.stage==Stage.OBSERVE and self.quiet_ticks>=25:
            self.transition(Stage.POSITION,'localized_and_quiet')
        elif self.stage==Stage.POSITION and self.qualified_ticks>=25:
            self.transition(Stage.CONTACT,'precontact_pose_measured')
        elif self.stage==Stage.CONTACT and self.qualified_ticks>=25:
            self.transition(Stage.TRANSFER,'qualified_contact')
        elif self.stage==Stage.TRANSFER and self.qualified_ticks>=25:
            self.transition(Stage.HOLD,'qualified_weight_transfer')
        elif self.stage==Stage.HOLD:
            resting=valid and s.rest_stability_qualified and loaded
            self.hold_ticks=self.hold_ticks+1 if resting else 0
            if self.tick==1000:
                # Score exactly controls751..1000. Keep control1000 in HOLD;
                # the original physical audit disallows unloading in that window.
                self.rest_window_passed=self.hold_ticks>=250
                if not self.rest_window_passed:
                    self.transition(Stage.ABORT,'original20s_hold_not_completed')
            elif self.tick>1000:
                if self.rest_window_passed:
                    self.transition(Stage.UNLOAD,'five_second_hold_and_original20s_window')
                else:self.transition(Stage.ABORT,'original20s_hold_not_completed')
        elif self.stage==Stage.UNLOAD and self.qualified_ticks>=25:
            self.transition(Stage.STAND,'qualified_unloading')
        elif self.stage==Stage.STAND:
            ready=quiet and unloaded and s.standing and s.clear_of_support
            self.stand_ticks=self.stand_ticks+1 if ready else 0
            if self.stand_ticks>=100:self.transition(Stage.DONE,'two_second_standing_settle')
        if self.tick>=1000 and self.stage in (Stage.OBSERVE,Stage.POSITION,Stage.CONTACT,Stage.TRANSFER):
            self.transition(Stage.ABORT,'original20s_rest_deadline')
        # Only a validated bounded approach may seek first contact. Weight
        # transfer/unloading never uses unknown load as zero or as permission.
        movement=valid and self.stage in (Stage.POSITION,Stage.CONTACT,Stage.TRANSFER,Stage.UNLOAD,Stage.STAND)
        if self.stage in (Stage.TRANSFER,Stage.UNLOAD,Stage.STAND):movement &= known
        if self.stage in (Stage.CONTACT,Stage.TRANSFER,Stage.UNLOAD):movement &= self.quiet_ticks>=25
        return self.decision(bool(movement))

    def decision(self,movement):
        return dict(stage=self.stage.value,tick=self.tick,advance_primitive=movement,
                    hold_ticks=self.hold_ticks,stand_ticks=self.stand_ticks,rest_window_passed=self.rest_window_passed,
                    aborted=self.stage==Stage.ABORT,reason=self.reason,
                    physical_success=False)
