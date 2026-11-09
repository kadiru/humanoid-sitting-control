"""Sensor-only quiet hover, normal landing and measured-load compensation.

Inputs: encoders, motor torques, projected gravity, gyro, observed rear-camera
geometry and previous command. No contact force or simulator pose is accepted.
"""
import numpy as np
from foot_wrench_observer import FootWrenchObserver
from complete_seat_path import arc_clock,interpolate
from humanoid_sitting.shared_frame import level_rotation

class SeatController:
    def __init__(self,plan,construction,path):
        self.plan=plan;self.construction=construction;self.path=path
        self.approach=np.asarray(path['approach_commands'])
        self.wrench=FootWrenchObserver(plan.model,plan.inertia)
        self.phase='approach';self.quiet_count=0;self.travel=0.;self.fraction=0.;self.bias=None
        self.filtered_force=np.zeros(3);self.pitch_feedback=0.
        self.land=np.array([n['command'] for n in path['landing']]);self.distance=np.array([n['travel_m'] for n in path['landing']])
        self.zero=np.array([n['zero'] for n in path['landing']]);self.full=np.array([n['full'] for n in path['landing']]);self.share=np.array([n['share'] for n in path['landing']])
        self.pitch=np.array([np.arctan2(-np.asarray(n['root'])[2,0],np.asarray(n['root'])[0,0]) for n in construction['nodes']])
        self.ankles=[plan.model.names.index(s+'_ankle_pitch_joint') for s in ('left','right')]

    def command(self,tick,q,qd,torque,gravity,gyro,geometry,previous):
        q=np.asarray(q);qd=np.asarray(qd);previous=np.asarray(previous)
        foot,condition=self.wrench.estimate(q,torque,gravity)
        valid_wrench=bool(np.max(condition)<200)
        force=np.array([-foot[:,0].sum(),0.,self.wrench.weight-foot[:,1].sum()])
        if self.bias is None:
            if not valid_wrench:raise ValueError('STARTUP_WRENCH_UNOBSERVABLE')
            self.bias=force.copy()
        observed=force-self.bias
        if valid_wrench:self.filtered_force+=.06*(observed-self.filtered_force)
        support=float(self.filtered_force@self.plan.n/self.wrench.weight)
        actual_pitch=float(np.arctan2(gravity[0],-gravity[2]))
        # Associate the live camera plane with the selected startup target using
        # encoder feet and gravity only. A fresh but different plane cannot release
        # the normal landing gate.
        level=level_rotation(np.asarray(gravity,float).copy());fk=self.plan.model.fk(q)
        root_estimate=np.mean([anchor[:3,3]-level@fk[side+'_ankle_roll_link'][:3,3]
                              for side,anchor in zip(('left','right'),self.plan.feet)],axis=0)
        current_normal=level@np.asarray(geometry[2:5],float)
        normal_error=float(np.linalg.norm(current_normal-self.plan.n))
        offset_error=abs(float(geometry[5]-current_normal@root_estimate-self.plan.d))
        associated=normal_error<=.03 and offset_error<=.01
        if tick<len(self.approach):
            target=self.approach[tick].copy()
            s,_=arc_clock((tick+1)*.02,self.construction['command_arc'][-1])
            pref=float(interpolate(self.pitch,np.array(self.construction['command_arc']),s))
        else:
            if self.phase=='approach':self.phase='hover'
            pref=float(self.pitch[-1]);target=interpolate(self.land,self.distance,self.travel)
        self.pitch_feedback+=.08*(.72*(actual_pitch-pref)+.12*float(gyro[1])-self.pitch_feedback)
        quiet=bool(np.max(abs(qd[:23]))<=.05 and abs(actual_pitch-pref)<=np.deg2rad(.5)
                   and abs(gyro[1])<=.02 and geometry[11]>.5 and geometry[12]<=.21 and associated and valid_wrench)
        if self.phase!='approach':
            self.quiet_count=self.quiet_count+1 if quiet else 0
            if self.quiet_count>=25 and self.phase=='hover':self.phase='landing'
            if self.phase in ('landing','load'):
                # No descent on stale geometry, moving joints or unobserved load.
                if self.quiet_count>=25 and support<.16:
                    self.travel=min(.015,self.travel+.006*.02)
                if support>=.12:self.phase='load'
                target=interpolate(self.land,self.distance,self.travel)
                maximum=float(interpolate(self.share,self.distance,self.travel))
                requested=np.clip((support-.02)/max(maximum,.05),0,1) if valid_wrench else 0.
                self.fraction+=float(np.clip(requested-self.fraction,-.008,.008))
                delta=interpolate(self.full-self.zero,self.distance,self.travel)*self.fraction
                target[self.plan.legids]+=delta
        target[self.ankles]+=np.clip(self.pitch_feedback,-.085,.085)
        applied=previous[:23]+np.clip(target[:23]-previous[:23],-.003*.99999,.003*.99999)
        desired=previous[:23]+(applied-previous[:23])/.1
        signal=dict(tick=int(tick),phase=self.phase,quiet=quiet,quiet_count=self.quiet_count,
                    travel_m=self.travel,observed_support_bw=support,load_fraction=self.fraction,
                    wrench_condition=condition.tolist(),pitch=actual_pitch,pitch_reference=pref,
                    pitch_feedback=self.pitch_feedback,geometry_age=float(geometry[12]),
                    target_associated=associated,normal_error=normal_error,offset_error_m=offset_error)
        return desired,signal
