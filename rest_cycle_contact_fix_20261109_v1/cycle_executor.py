"""Bounded sensor-driven sit, supported hold, release, and stand sequencing.

This module executes an independently admitted path, never constructs or
certifies one. Simulator contact truth is not an input. A sensor completion is
not a physical success; the unchanged independent rollout scorer remains
required. Historical calibration paths are not automatically admitted.
"""
from dataclasses import dataclass
import math
import numpy as np
from contact_gate import Packet, snapshot_flag, snapshot_numeric

MOVING = 23
TOTAL = 37
FILTER = .1
STEP = .003
MARGIN = .03
QUIET_SAMPLES = 25
HOLD_SAMPLES = 250
STANDING_SAMPLES = 100
LEG_CONTROLS = 1000
PHYSICS_PER_CONTROL = 4


def encode_float32_path(commands, initial, lower, upper):
    """Encode frozen command targets for the unchanged float32 actuator.

    Rounding a double-precision inverse-filter input accumulates error over
    hundreds of commands. Reconstruct each latent input from the actual
    float32 predecessor instead, without changing the frozen target path.
    Both the double planning box and the original float32 clamp remain in force.
    """
    low = np.maximum(lower+MARGIN, (lower.astype(np.float32)+np.float32(MARGIN)).astype(float))
    high = np.minimum(upper-MARGIN, (upper.astype(np.float32)-np.float32(MARGIN)).astype(float))
    if np.any(low > high):
        raise ValueError('NO_FLOAT32_INTERIOR_IN_ORIGINAL_BOX')
    previous = np.asarray(initial[:MOVING], np.float32).copy()
    encoded = []
    for command in commands:
        target = np.asarray(command[:MOVING], np.float32)
        desired = np.clip(previous.astype(float)+(target.astype(float)-previous)/FILTER,
                          low, high).astype(np.float32)
        desired = np.where(desired.astype(float) < low, np.nextafter(desired, np.float32(np.inf)), desired)
        desired = np.where(desired.astype(float) > high, np.nextafter(desired, np.float32(-np.inf)), desired)
        if np.any(desired.astype(float) < low) or np.any(desired.astype(float) > high):
            raise ValueError('NO_REPRESENTABLE_INPUT_IN_ORIGINAL_BOX')
        applied = previous+np.clip(np.float32(FILTER)*(desired-previous),
                                   np.float32(-STEP), np.float32(STEP))
        if np.max(abs(applied-target)) > 2e-7:
            raise ValueError('FLOAT32_PATH_EXCEEDS_RECORDED_COMMAND_ALLOWANCE')
        encoded.append(desired)
        previous = applied
    return np.asarray(encoded, np.float32)


def pitch_from_gravity(gravity):
    g = np.asarray(gravity, dtype=float)
    if (g.shape != (3,) or not np.isfinite(g).all() or
        not np.isfinite(np.linalg.norm(g)) or np.linalg.norm(g) == 0):
        return float('nan')
    return math.atan2(g[0], -g[2])


def pitch_from_root(root):
    r = np.asarray(root, dtype=float)
    if (r.shape != (4, 4) or not np.isfinite(r).all() or
        not np.allclose(r[3], [0, 0, 0, 1], rtol=0, atol=1e-10) or
        not np.allclose(r[:3, :3].T @ r[:3, :3], np.eye(3), rtol=0, atol=1e-8) or
        abs(np.linalg.det(r[:3, :3])-1) > 1e-8):
        raise ValueError('INVALID_FROZEN_REFERENCE_ROTATION')
    # The admitted model's root is expressed in the gravity-aligned frame.
    return pitch_from_gravity(-r[2, :3])


@dataclass(frozen=True)
class Measurements:
    physics_step: int
    raw_joint_velocity: tuple
    gravity: tuple
    gyro_y_rad_s: float
    geometry_valid: bool
    geometry_age_s: float
    current_pose_admitted: bool
    load_interval_admitted: bool = False
    load_lower_bw: float | None = None
    load_upper_bw: float | None = None
    geometry_separated: bool = False

    def __post_init__(self):
        # Snapshot potentially reused numpy/torch producer buffers immediately.
        # tuple(tensor) retains scalar tensor views into producer storage.
        # Copy numeric scalars, preserving float32 threshold semantics.
        object.__setattr__(self, 'raw_joint_velocity', snapshot_numeric(self.raw_joint_velocity,vector=True))
        object.__setattr__(self, 'gravity', snapshot_numeric(self.gravity,vector=True))
        for name in ('gyro_y_rad_s','geometry_age_s'):
            object.__setattr__(self,name,snapshot_numeric(getattr(self,name)))
        for name in ('load_lower_bw','load_upper_bw'):
            value=getattr(self,name)
            object.__setattr__(self,name,None if value is None else snapshot_numeric(value))
        for name in ('geometry_valid','current_pose_admitted','load_interval_admitted','geometry_separated'):
            object.__setattr__(self,name,snapshot_flag(getattr(self,name)))
        if type(self.physics_step) is not int or self.physics_step < 0:
            raise ValueError('INTEGER_PHYSICS_STEP_REQUIRED')

    def packet(self, reference_pitch):
        measured = pitch_from_gravity(self.gravity)
        difference = measured-reference_pitch
        error = math.atan2(math.sin(difference), math.cos(difference)) if math.isfinite(difference) else float('nan')
        return Packet(self.physics_step, self.raw_joint_velocity, error,
                      self.gyro_y_rad_s, self.geometry_valid, self.geometry_age_s,
                      self.current_pose_admitted, self.load_interval_admitted,
                      self.load_lower_bw, self.load_upper_bw)

    def available(self):
        return bool(self.geometry_valid and self.current_pose_admitted and
                    len(self.raw_joint_velocity) == MOVING and
                    all(math.isfinite(v) for v in self.raw_joint_velocity) and
                    math.isfinite(pitch_from_gravity(self.gravity)) and
                    math.isfinite(self.gyro_y_rad_s) and
                    math.isfinite(self.geometry_age_s) and 0 <= self.geometry_age_s <= .21)

    def released(self):
        lo, hi = self.load_lower_bw, self.load_upper_bw
        return bool(self.geometry_separated and self.load_interval_admitted and
                    lo is not None and hi is not None and
                    math.isfinite(lo) and math.isfinite(hi) and 0 <= lo <= hi < .01)


@dataclass(frozen=True)
class Boundary:
    command_count: int
    pitch: float


class CyclePlan:
    """Freeze commands and explicit model-node associations; do not optimize."""
    def __init__(self, plan, lower, upper, *, hover_node, touch_node,
                 loaded_node, release_node):
        self.lower, self.upper = np.array(lower, float), np.array(upper, float)
        if (self.lower.shape != (MOVING,) or self.upper.shape != (MOVING,) or
            not np.isfinite(self.lower).all() or not np.isfinite(self.upper).all() or
            np.any(self.upper-self.lower < 2*MARGIN)):
            raise ValueError('INVALID_JOINT_LIMITS')
        forward, backward = plan['nodes'], plan['return_nodes']
        if not 0 < hover_node < touch_node < loaded_node < len(forward):
            raise ValueError('ORDERED_FORWARD_BOUNDARIES_REQUIRED')
        if not 0 < release_node < len(backward)-1:
            raise ValueError('ORDERED_RETURN_BOUNDARY_REQUIRED')
        for node, gap in ((forward[hover_node], .012), (forward[touch_node], 0.),
                          (backward[release_node], .012)):
            if not node.get('normal') or abs(node['gap']-gap) > 1e-10:
                raise ValueError('BOUNDARY_NOT_ASSOCIATED_WITH_NORMAL_PATH')
        # The 2%BW observer interval must have room above the unchanged 5%BW
        # qualification threshold. A zero-load touch node cannot serve as this
        # gate. The model load is only a premise, never a sensor measurement.
        if not math.isfinite(forward[loaded_node]['load']) or forward[loaded_node]['load'] < .07:
            raise ValueError('LOADED_GATE_REQUIRES_POSITIVE_MODEL_SUPPORT')
        self.initial = np.asarray(forward[0]['command'], dtype=float).copy()
        if self.initial.shape != (TOTAL,) or not np.isfinite(self.initial).all():
            raise ValueError('INVALID_INITIAL_COMMAND')
        self.forward, fends = self._path(forward, plan['forward'])
        self.backward, bends = self._path(backward, plan['backward'])
        if (not np.array_equal(forward[-1]['command'], backward[0]['command']) or
            not np.array_equal(backward[-1]['command'], self.initial)):
            raise ValueError('RETURN_MUST_JOIN_REST_AND_ORIGINAL_STANDING')
        self.hover = Boundary(fends[hover_node], pitch_from_root(forward[hover_node]['root']))
        self.touch = Boundary(fends[touch_node], pitch_from_root(forward[touch_node]['root']))
        self.loaded = Boundary(fends[loaded_node], pitch_from_root(forward[loaded_node]['root']))
        self.release = Boundary(bends[release_node], pitch_from_root(backward[release_node]['root']))
        self.standing_pitch = pitch_from_root(forward[0]['root'])
        self.rest_pitch = pitch_from_root(forward[-1]['root'])
        if (len(self.forward)+4*QUIET_SAMPLES+HOLD_SAMPLES > LEG_CONTROLS or
            len(self.backward)+QUIET_SAMPLES+STANDING_SAMPLES > LEG_CONTROLS):
            raise ValueError('ORIGINAL_TIME_WINDOWS_HAVE_NO_QUALIFICATION_RESERVE')
        for array in (self.initial, self.lower, self.upper, self.forward, self.backward):
            array.flags.writeable = False

    def _path(self, nodes, scheduled):
        commands, desired = np.array(scheduled['commands'], float), np.array(scheduled['desired'], float)
        if (commands.ndim != 2 or commands.shape[1] != TOTAL or
            not 0 < len(commands) <= LEG_CONTROLS or desired.shape != (len(commands), MOVING) or
            not np.isfinite(commands).all() or not np.isfinite(desired).all()):
            raise ValueError('INVALID_PATH_ARRAYS')
        previous = np.vstack([nodes[0]['command'], commands[:-1]])
        predicted = previous[:, :MOVING]+np.clip(FILTER*(desired-previous[:, :MOVING]), -STEP, STEP)
        if (np.max(abs(predicted-commands[:, :MOVING])) > 1e-10 or
            np.max(abs(commands-previous)) > STEP+1e-12 or
            np.any(desired < self.lower+MARGIN) or np.any(desired > self.upper-MARGIN) or
            not np.array_equal(commands[:, MOVING:], np.broadcast_to(self.initial[MOVING:], commands[:, MOVING:].shape))):
            raise ValueError('ORIGINAL_COMMAND_CONTRACT_VIOLATED')
        ends, begin = [0], 0
        for node in nodes[1:]:
            target = np.asarray(node['command'], float)
            matches = np.flatnonzero(np.max(abs(commands[begin:]-target), axis=1) < 1e-12)
            if not len(matches):
                raise ValueError('NODE_NOT_PRESENT_IN_COMMAND_PATH')
            begin += int(matches[0])+1
            ends.append(begin)
            pitch_from_root(node['root'])
        if begin != len(commands):
            raise ValueError('COMMANDS_AFTER_FINAL_NODE')
        return encode_float32_path(commands, nodes[0]['command'], self.lower, self.upper), ends


class CycleExecutor:
    """One continuous cycle; callers apply the original filter exactly once."""
    def __init__(self, plan: CyclePlan, *, current_path_admitted, loaded_observer_admitted):
        if not current_path_admitted or not loaded_observer_admitted:
            raise ValueError('CURRENT_PATH_AND_LOADED_OBSERVER_ADMISSION_REQUIRED')
        self.plan = plan
        self.phase = 'sit'
        self.index = self.hold = self.quiet = 0
        self.gate = 0
        self.last_step = self.start_step = self.return_step = None
        self.aborted = False
        self.expected = plan.initial.astype(np.float32)

    def _result(self, desired, reason):
        return desired.tolist(), dict(phase=self.phase, path_index=self.index,
            reason=reason, aborted=self.aborted, hold_samples=self.hold,
            quiet_samples=self.quiet, sensor_cycle_complete=self.phase == 'complete' and not self.aborted,
            actuation_allowed=not self.aborted, physical_success=False)

    def _abort(self, reason):
        self.aborted = True
        return self._result(self.expected[:MOVING].copy(), reason)

    def _emit(self, desired, reason):
        desired = np.asarray(desired, np.float32)
        self.expected[:MOVING] += np.clip(np.float32(FILTER)*(desired-self.expected[:MOVING]),
                                         np.float32(-STEP), np.float32(STEP))
        return self._result(desired, reason)

    def command(self, sample: Measurements, previous_command):
        previous = np.asarray(previous_command, np.float32)
        if (previous.shape != (TOTAL,) or not np.isfinite(previous).all() or
            np.max(abs(previous-self.expected)) > 2e-7 or
            not np.array_equal(previous[MOVING:], self.plan.initial[MOVING:].astype(np.float32))):
            # A mismatch must not request motion back to an old expectation.
            # The caller must also honor actuation_allowed=False on faults.
            if previous.shape == (TOTAL,) and np.isfinite(previous).all():
                self.expected = previous.copy()
            return self._abort('applied_command_does_not_match_single_filter')
        # Follow actual float32 actuator state; only the recorded rounding
        # allowance is used here, never to broaden a physical threshold.
        self.expected = previous.copy()
        if self.aborted:
            return self._result(previous[:MOVING], 'latched_fault')
        if self.last_step is not None and sample.physics_step != self.last_step+PHYSICS_PER_CONTROL:
            return self._abort('stale_or_skipped_sensor_packet')
        self.last_step = sample.physics_step
        if self.start_step is None:
            self.start_step = sample.physics_step
        elapsed = (sample.physics_step-self.start_step)//PHYSICS_PER_CONTROL
        if self.phase == 'complete':
            return self._result(previous[:MOVING], 'completed_no_further_motion')
        valid = sample.available()
        if self.phase == 'sit':
            if self.index == len(self.plan.forward):
                p = sample.packet(self.plan.rest_pitch)
                self.hold = self.hold+1 if valid and p.quiet() and p.loaded() else 0
            if elapsed >= LEG_CONTROLS:
                if self.index != len(self.plan.forward) or self.hold < HOLD_SAMPLES:
                    return self._abort('first20s_final5s_hold_not_qualified')
                self.phase, self.index, self.quiet = 'return', 0, 0
                self.return_step = sample.physics_step
            elif not valid:
                self.quiet = self.hold = 0
                return self._result(previous[:MOVING], 'current_sensors_or_pose_not_admitted')
            else:
                gates = ((0, self.plan.standing_pitch, False),
                         (self.plan.hover.command_count, self.plan.hover.pitch, False),
                         (self.plan.touch.command_count, self.plan.touch.pitch, False),
                         (self.plan.loaded.command_count, self.plan.loaded.pitch, True))
                if self.gate < len(gates) and self.index == gates[self.gate][0]:
                    _, pitch, require_load = gates[self.gate]
                    p = sample.packet(pitch)
                    good = p.quiet() and (p.loaded() or not require_load)
                    self.quiet = self.quiet+1 if good else 0
                    if self.quiet < QUIET_SAMPLES:
                        return self._result(previous[:MOVING], 'await_current_stage_quiet_and_load' if require_load else 'await_current_stage_quiet')
                    self.gate += 1
                    self.quiet = 0
                if self.index < len(self.plan.forward):
                    desired = self.plan.forward[self.index]
                    self.index += 1
                    return self._emit(desired, 'admitted_forward_segment')
                return self._result(previous[:MOVING], 'qualifying_supported_hold')

        return_elapsed = (sample.physics_step-self.return_step)//PHYSICS_PER_CONTROL
        at_stand = self.index == len(self.plan.backward)
        at_release = self.index == self.plan.release.command_count and self.gate == 4
        if at_stand or at_release:
            pitch = self.plan.standing_pitch if at_stand else self.plan.release.pitch
            good = valid and sample.packet(pitch).quiet() and sample.released()
            self.quiet = self.quiet+1 if good else 0
            needed = STANDING_SAMPLES if at_stand else QUIET_SAMPLES
            if self.quiet >= needed:
                if at_stand:
                    self.phase = 'complete'
                    return self._result(previous[:MOVING], 'sensor_cycle_complete_requires_independent_physical_audit')
                self.gate = 5
                self.quiet = 0
            elif return_elapsed < LEG_CONTROLS:
                return self._result(previous[:MOVING], 'await_unloaded_standing_quiet' if at_stand else 'await_sensor_qualified_release')
        if return_elapsed >= LEG_CONTROLS:
            return self._abort('original20s_return_deadline')
        if not valid:
            self.quiet = 0
            return self._result(previous[:MOVING], 'current_sensors_or_pose_not_admitted')
        desired = self.plan.backward[self.index]
        self.index += 1
        return self._emit(desired, 'admitted_return_segment')
