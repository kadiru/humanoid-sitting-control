"""Bounded geometric landing for load calibration; no estimated force control.

The force estimator is read and recorded separately. Unsupported force readings
cannot alter commands. This supplies missing loaded observations without using
simulator force truth as a teacher input. Success still needs the original audit.
"""
import numpy as np
from complete_seat_path import arc_clock, interpolate
from humanoid_sitting.shared_frame import level_rotation


class NormalLanding:
    def __init__(self, plan, construction, path):
        self.plan, self.construction = plan, construction
        self.approach = np.asarray(path['approach_commands'])
        self.land = np.asarray([n['command'] for n in path['landing']])
        self.distance = np.asarray([n['travel_m'] for n in path['landing']])
        self.pitch = np.asarray([np.arctan2(-np.asarray(n['root'])[2,0], np.asarray(n['root'])[0,0])
                                 for n in construction['nodes']])
        self.ankles = [plan.model.names.index(s+'_ankle_pitch_joint') for s in ('left','right')]
        self.phase, self.quiet_count, self.travel, self.pitch_feedback = 'approach', 0, 0., 0.

    def command(self, tick, q, qd, gravity, gyro, geometry, previous):
        level = level_rotation(np.asarray(gravity, float).copy())
        fk = self.plan.model.fk(q)
        root = np.mean([anchor[:3,3]-level@fk[side+'_ankle_roll_link'][:3,3]
                        for side,anchor in zip(('left','right'),self.plan.feet)],axis=0)
        normal = level@np.asarray(geometry[2:5],float)
        normal_error = float(np.linalg.norm(normal-self.plan.n))
        offset_error = abs(float(geometry[5]-normal@root-self.plan.d))
        associated = normal_error <= .03 and offset_error <= .01
        pitch = float(np.arctan2(gravity[0], -gravity[2]))
        if tick < len(self.approach):
            target = self.approach[tick].copy()
            progress,_ = arc_clock((tick+1)*.02,self.construction['command_arc'][-1])
            reference = float(interpolate(self.pitch,np.asarray(self.construction['command_arc']),progress))
        else:
            if self.phase == 'approach': self.phase = 'hover'
            reference = float(self.pitch[-1])
            target = interpolate(self.land,self.distance,self.travel).copy()
        self.pitch_feedback += .08*(.72*(pitch-reference)+.12*float(gyro[1])-self.pitch_feedback)
        quiet = bool(np.max(abs(qd[:23])) <= .05 and abs(pitch-reference) <= np.deg2rad(.5)
                     and abs(gyro[1]) <= .02 and geometry[11] > .5 and geometry[12] <= .21 and associated)
        if self.phase != 'approach':
            self.quiet_count = self.quiet_count+1 if quiet else 0
            if self.phase == 'hover' and self.quiet_count >= 25: self.phase = 'landing'
            if self.phase == 'landing' and self.quiet_count >= 25:
                # Fixed, pre-audited15mm stroke; no additional contact search.
                self.travel = min(.015,self.travel+.006*.02)
                if self.travel >= .015: self.phase = 'hold'
            target = interpolate(self.land,self.distance,self.travel).copy()
        target[self.ankles] += np.clip(self.pitch_feedback,-.085,.085)
        previous = np.asarray(previous)[:23]
        applied = previous+np.clip(target[:23]-previous,-.003*.99999,.003*.99999)
        desired = previous+(applied-previous)/.1
        return desired,dict(tick=tick,phase=self.phase,quiet=quiet,quiet_count=self.quiet_count,
                            travel_m=self.travel,pitch=pitch,pitch_reference=reference,
                            normal_error=normal_error,offset_error_m=offset_error,
                            measured_geometry_valid=bool(geometry[11]),geometry_age_s=float(geometry[12]),
                            estimated_force_controls_action=False)
