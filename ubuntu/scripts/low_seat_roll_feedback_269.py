"""Bounded raw ankle-roll velocity feedback; existing pitch controller retained.

One fixed hypothesis: add negative measured-velocity feedback with time scale
Kd/Kp=4/160=.025s, derived from installed ankle drive parameters. No moment
observer or contact truth. Begins only at the stationary12mm hover; thereafter
current validity/load gates remain necessary. Not a gain/pose sweep.
"""
import numpy as np
from low_seat_bias_feedback_v2_207 import BiasFeedback
from low_seat_touchdown_path_175 import TouchdownPath

class RollTouchdownPath(TouchdownPath):
 def __init__(self,plan,m):
  super().__init__(plan,m)
  for i,j in enumerate(self.ids):
   if m.names[j] in ['left_ankle_roll_joint','right_ankle_roll_joint']:self.cap[i]=np.float32(.00225)

class RollFeedback(BiasFeedback):
 def __init__(self,model,plan):
  super().__init__(model,plan);self.roll_ids=np.array([model.names.index(s+'_ankle_roll_joint') for s in ['left','right']])
  self.roll_offset=np.zeros(2,np.float32);self.roll_started=False
  limits={j['name']:np.deg2rad(j['limits']) for j in model.joints if j['type']=='PhysicsRevoluteJoint'}
  self.roll_lo=np.array([limits[model.names[j]][0]+.03 for j in self.roll_ids]);self.roll_hi=np.array([limits[model.names[j]][1]-.03 for j in self.roll_ids])
  self.roll_tau=np.array([e['load_checks'][0]['static_torque_Nm'] for e in plan['path']])[:,[5,11]]
 def update(self,q,gravity,gyro,position,nominal_target,foot_wrench,conditions,weight,geometry=None,at_hover=False,qd=None):
  target,record=super().update(q,gravity,gyro,position,nominal_target,foot_wrench,conditions,weight,geometry=geometry,at_hover=at_hover)
  velocity=np.asarray(qd) if qd is not None else np.full(37,np.nan)
  valid=bool(velocity.shape==(37,) and np.isfinite(velocity).all() and record['valid'] and record['geometry_fresh'] and record['estimated_foot_load_fraction']>=.8)
  if at_hover and valid:self.roll_started=True
  enabled=bool(self.roll_started and valid)
  before=self.roll_offset.copy();raw=-.025*velocity[self.roll_ids] if enabled else np.zeros(2)
  tau=np.array([np.interp(position,self.arc,self.roll_tau[:,j]) for j in range(2)])
  # L1 pitch+roll bound retains the original .06rad per-ankle total authority.
  # Pitch remains EXACT207. Its .00075 step ensures this interval cannot shrink
  # faster than roll can decay. Fixed .01 roll cap, original .03 desired margin.
  authority=np.minimum(.01,np.maximum(0,.06-np.abs(self.offset.astype(float))))
  static_authority=np.maximum(0,(20-abs(tau)-160*.012)/160)
  authority=np.minimum(authority,static_authority)
  wanted=np.clip(raw,-authority,authority)
  nominal=np.asarray(nominal_target,np.float32)
  lower=np.maximum.reduce([-np.full(2,.00075),-authority-before,.1*(self.roll_lo-nominal[self.roll_ids]-before)])
  upper=np.minimum.reduce([np.full(2,.00075),authority-before,.1*(self.roll_hi-nominal[self.roll_ids]-before)])
  assert np.all(lower<=upper),'Roll feedback command/authority headroom exhausted'
  step=np.clip(wanted-before,lower,upper).astype(np.float32);after=(before+step).astype(np.float32)
  self.roll_offset=after;target[self.roll_ids]+=before+step/np.float32(.1)
  record['roll']=dict(qd_available=qd is not None,raw_qd=None if not np.isfinite(velocity).all() else velocity.tolist(),valid=valid,started=self.roll_started,enabled=enabled,at_hover=bool(at_hover),gain_s=.025,cap_rad=.01,total_pitch_plus_roll_cap_rad=.06,rate_rad_s=.0375,nominal_rate_rad_s=.1125,authority_rad=authority.tolist(),static_torque_Nm=tau.tolist(),raw_correction_rad=raw.tolist(),wanted_rad=wanted.tolist(),before=before.tolist(),after=after.tolist(),target_correction=(before+step/np.float32(.1)).tolist())
  return target,record
