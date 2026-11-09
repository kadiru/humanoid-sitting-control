"""Measured-settling touchdown schedule on the unchanged150/164 path.

A separately versioned pre-training timing controller. Reserve feedback rate
on ankle pitch only; other moving joints use the unchanged applied ceiling.
At12mm planned gap require25 consecutive current quiet samples, then cap
modeled normal approach at6mm/s. Static loads remain assumed and inherited
current-sensor load permission is unchanged. No truth or saved-input access.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from low_seat_gated_command_128 import GatedCommandPath
class TouchdownPath(GatedCommandPath):
 def __init__(self,plan,m):
  super().__init__(plan,m);self.cap=np.full(len(self.ids),np.float32(.003),np.float32)
  for j,n in enumerate(self.ids):
   if m.names[n] in ['left_ankle_pitch_joint','right_ankle_pitch_joint']:self.cap[j]=np.float32(.00225)
  self.gaps=np.array([x['pelvis_gap_m'] for x in self.path]);self.gaps[80:]=0.
  self.pitch=Rotation.from_quat(np.array([x['root'][3:] for x in self.path])).as_euler('xyz')[:,1]
  assert np.all(np.diff(self.gaps)<=1e-10)
  self.hover=float(np.interp(.012,self.gaps[:81][::-1],self.arc[:81][::-1]));self.hold_ticks=0;self.released=False;self.settle=None;self.settle_tick=None
 def observe_settling(self,tick,q,gravity,gyro,qd,geometry):
  # Current raw measurements only. Repeated reference callbacks are idempotent.
  if self.settle_tick==tick:return
  self.settle_tick=tick
  q=np.asarray(q);g=np.asarray(gravity);w=np.asarray(gyro);qd=np.asarray(qd);geo=np.asarray(geometry)
  finite=bool(np.isfinite(np.r_[q,g,w,qd,geo]).all())
  fresh=bool(finite and geo[11] and 0<=geo[12]<=.5 and np.linalg.norm(g)>.5)
  pitch=float(np.arctan2(g[0],-g[2])) if fresh else None
  goal=float(np.interp(self.position,self.arc,self.pitch))
  error=abs(pitch-goal) if fresh else None
  # Controller admission reserves only; unchanged evaluator still checks every
  # loaded slip/contact and full20s/final5s support independently.
  quiet=bool(fresh and error<=np.deg2rad(.5) and abs(w[1])<=.02 and max(abs(qd[self.ids]))<=.05)
  self.settle=dict(available=fresh,quiet=quiet,pitch_rad=pitch,target_pitch_rad=goal,pitch_error_rad=error,
                  gyro_y_rad_s=float(w[1]) if finite else None,joint_speed_rad_s=float(max(abs(qd[self.ids]))) if finite else None,
                  current_geometry_age_s=float(geo[12]) if finite else None,tick=int(tick))
 def advance(self,allowed,load):
  prev=self.nominal.copy();start=self.position;limit=float(np.interp(float(np.clip(load,0,.1)),self.loads[80:],self.arc[80:]))
  if not self.released:
   if abs(start-self.hover)<1e-10:
    if self.hold_ticks<25:
     self.hold_ticks=self.hold_ticks+1 if allowed and self.settle is not None and self.settle['quiet'] else 0
     allowed=False
    elif allowed and self.settle is not None and self.settle['quiet']:self.released=True
    else:self.hold_ticks=0;allowed=False
   if not self.released:limit=min(limit,self.hover)
  if not allowed or start>=limit:
   self.last_target=prev.copy();self.last_info=dict(start=start,end=start,paused=True,hold=self.hold_ticks,released=self.released,settling=self.settle,max_normal_bw=float(load),limit=limit);return prev.copy()
  gap=float(np.interp(start,self.arc,self.gaps))
  def valid(s):
   goal=self.at(s);raw=prev[self.ids]+(goal[self.ids]-prev[self.ids])/self.alpha
   normal_ok=not self.released or gap-float(np.interp(s,self.arc,self.gaps))<=.006*.02+1e-14
   return bool(normal_ok and np.all(raw>=self.lo) and np.all(raw<=self.hi) and np.all(abs(self.alpha*(raw-prev[self.ids]))<=self.cap)),goal,raw
  edge=int(np.searchsorted(self.arc,start,side='right'));safe=start
  while edge<len(self.arc):
   p=min(float(self.arc[edge]),limit)
   if valid(p)[0]:
    safe=p
    if p==limit:break
    edge+=1
   else:
    lo,hi=safe,p
    for _ in range(45):
     mid=(lo+hi)/2
     if valid(mid)[0]:lo=mid
     else:hi=mid
    safe=lo;break
  assert safe>start+1e-12
  okay,goal,raw=valid(safe);assert okay
  target=prev.copy();target[self.ids]=raw;self.nominal[self.ids]=prev[self.ids]+np.clip(self.alpha*(np.clip(raw,self.lo,self.hi)-prev[self.ids]),-self.cap,self.cap)
  self.position=safe;self.last_target=target;self.last_info=dict(start=start,end=safe,paused=False,hold=self.hold_ticks,released=self.released,settling=self.settle,max_normal_bw=float(load),limit=limit,demanded_normal_bw=float(np.interp(safe,self.arc,self.loads)));return target.copy()
