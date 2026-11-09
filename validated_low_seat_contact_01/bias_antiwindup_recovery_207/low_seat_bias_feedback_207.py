"""One bounded integral-bias experiment on160, not a gain sweep.

Only the existing stationary12mm hover can accumulate a correction. Integral
 time constant2s givesKi=.6/2=.3/s; cap.03rad within UNCHANGED.06 total
 authority and.0375rad/s feedback allocation. No path/settling guard relaxed.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from humanoid_sitting.shared_frame import level_rotation

class BiasFeedback:
    def __init__(self,model,plan):
        self.model=model;self.ids=np.array([model.names.index(s+'_ankle_pitch_joint') for s in ['left','right']])
        self.feet=plan['path'][0]['foot_targets'];self.roots=np.array([e['root'] for e in plan['path']])
        self.pitch=Rotation.from_quat(self.roots[:,3:]).as_euler('xyz')[:,1]
        Q=np.array([e['load_checks'][0]['command'] for e in plan['path']]);self.arc=np.r_[0,np.cumsum(abs(np.diff(Q,axis=0)).max(1))]
        self.tau=np.array([e['load_checks'][0]['static_torque_Nm'] for e in plan['path']])[:,[4,10]]
        limits={j['name']:np.deg2rad(j['limits']) for j in model.joints if j['type']=='PhysicsRevoluteJoint'}
        self.lo=np.array([limits[model.names[j]][0]+.03 for j in self.ids]);self.hi=np.array([limits[model.names[j]][1]-.03 for j in self.ids])
        self.previous_goal=float(self.pitch[0]);self.offset=np.zeros(2,np.float32);self.ticks=0;self.integral=0.;self.integral_ticks=0

    def update(self,q,gravity,gyro,position,nominal_target,foot_wrench,conditions,weight,geometry=None,at_hover=False):
        """Current measured inputs only. Neither truth nor recorded paths enter."""
        q=np.asarray(q);g=np.asarray(gravity);omega=np.asarray(gyro);nominal=np.asarray(nominal_target,np.float32)
        finite=np.isfinite(np.r_[q,g,omega,position,nominal,foot_wrench.ravel(),conditions,weight]).all()
        valid=bool(finite and np.linalg.norm(g)>.5 and weight>0)
        goal=float(np.interp(position,self.arc,self.pitch));goal_rate=(goal-self.previous_goal)/.02
        self.previous_goal=goal;self.ticks+=1
        pitch=None;root=None;anchor_valid=False;raw=0.;load_factor=0.
        if valid:
            pitch=float(np.arctan2(g[0],-g[2]));R=level_rotation(g);fk=self.model.fk(q)
            root=np.mean([np.array(self.feet[s]['position'])-R@fk[s+'_ankle_roll_link'][:3,3] for s in ['left','right']],axis=0)
            errors=[np.linalg.norm(Rotation.from_matrix(np.array(self.feet[s]['rotation']).T@R@fk[s+'_ankle_roll_link'][:3,:3]).as_rotvec()) for s in ['left','right']]
            anchor_valid=bool(max(errors)<=np.deg2rad(.5))
            goal_x=float(np.interp(position,self.arc,self.roots[:,0]))
            raw=.6*(pitch-goal)+.15*(float(omega[1])-goal_rate)+(float(root[0]-goal_x) if anchor_valid else 0.)
            load_factor=float(np.clip(foot_wrench[:,1].sum()/weight,0,1)) if max(conditions)<200 and min(foot_wrench[:,1])>0 else 0.
        # Per-ankle static torque authority after allowing the frozen parent's
        # old.012rad body span. Measured forces still need physical validation.
        tau=np.array([np.interp(position,self.arc,self.tau[:,j]) for j in range(2)])
        authority=np.minimum(.06,np.maximum(0,(20-abs(tau)-160*.012)/160))
        u=min(self.ticks*.02/.5,1.);ramp=u*u*(3-2*u)
        # Integrate only a current measured, slowly rotating, well-loaded hover.
        # Raw joint speed STILL gates descent in unchanged175; it is not
        # replaced by finite differences or bypassed here.
        geo=None if geometry is None else np.asarray(geometry)
        fresh=bool(geo is not None and np.isfinite(geo).all() and geo[11] and 0<=geo[12]<=.5)
        integral_before=float(self.integral)
        eligible=bool(valid and fresh and at_hover and load_factor>=.8 and
                      abs(pitch-goal)<=np.deg2rad(3.) and abs(float(omega[1]))<=.05)
        proposed=float(np.clip(integral_before+.3*(pitch-goal)*.02,-.03,.03)) if eligible else integral_before
        # State freezes on invalid input; its contribution is removed smoothly
        # through the unchanged output-rate allocator until measurements return.
        enabled=bool(valid and fresh and load_factor>=.8)
        before=self.offset.copy()
        low=.1*(self.lo-nominal[self.ids]-before);high=.1*(self.hi-nominal[self.ids]-before)
        feasible=bool(np.all(low<=high) and np.all(low<=.00075) and np.all(high>=-.00075))
        if not feasible:raise RuntimeError('Nominal/feedback command headroom exhausted')
        lower=np.maximum(low,-.00075);upper=np.minimum(high,.00075)
        proposed_raw=(raw+(proposed if enabled else 0.))*load_factor*ramp
        proposed_wanted=np.clip(proposed_raw,-authority,authority).astype(np.float32)
        unmet=proposed_raw-proposed_wanted.astype(float)
        rate_unmet=(proposed_wanted-before)-np.clip(proposed_wanted-before,lower,upper)
        delta_i=proposed-integral_before
        windup=bool(delta_i and (np.any(unmet*delta_i>0) or np.any(rate_unmet*delta_i>0)))
        accepted=bool(eligible and not windup)
        if accepted:self.integral=proposed;self.integral_ticks+=1
        integral_applied=self.integral if enabled else 0.
        wanted=np.clip((raw+integral_applied)*load_factor*ramp,-authority,authority).astype(np.float32)
        step=np.clip(wanted-before,lower,upper).astype(np.float32)
        self.offset=(before+step).astype(np.float32)
        ref=(before+step/np.float32(.1)).astype(np.float32)
        target=nominal.copy();target[self.ids]+=ref
        return target,dict(integral_before_rad=integral_before,integral_after_rad=float(self.integral),integral_applied_rad=float(integral_applied),integral_eligible=eligible,integral_update_accepted=accepted,integral_antiwindup=windup,integral_ticks=self.integral_ticks,integral_ki_per_s=.3,integral_limit_rad=.03,at_hover=bool(at_hover),geometry_fresh=fresh,valid=valid,pitch_rad=pitch,target_pitch_rad=goal,target_pitch_rate_rad_s=goal_rate,
            estimated_root=None if root is None else root.tolist(),anchor_valid=anchor_valid,raw_correction_rad=raw,
            estimated_foot_load_fraction=load_factor,authority_rad=authority.tolist(),offset_before=before.tolist(),
            offset_after=self.offset.tolist(),reference_correction=ref.tolist(),ramp=ramp,
            nominal_rate_rad_s=.1125,feedback_rate_rad_s=.0375,applied_rate_rad_s=.15)
