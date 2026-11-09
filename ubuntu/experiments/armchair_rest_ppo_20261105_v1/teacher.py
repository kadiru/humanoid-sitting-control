"""Camera/encoder/IMU/motor-torque teacher; no simulator contact truth input."""
import numpy as np
from plan import LeanPlan
from wrench import WrenchObserver

class Teacher:
    def __init__(self,env,observer,profile):
        self.env=env
        q=env.robot.data.joint_pos.torch[0].cpu().numpy()
        imu=env.robot.data.projected_gravity_b.torch[0].cpu().numpy()
        prev=env.previous[0].cpu().numpy().copy()
        self.plan=LeanPlan(env.model,q,imu,observer.last[0],observer.last[1]['selected_points'],prev,profile=profile)
        self.nodes,self.t=self.plan.solve();assert self.t[-1]<14.4
        self.x=np.array([n['cmd'] for n in self.nodes]);self.pitch=np.array([n['pitch'] for n in self.nodes]);self.gap=np.array([n['gap'] for n in self.nodes])
        self.base=prev[:23].copy();self.wrench=WrenchObserver(self.plan,self.nodes)
        self.bias=[];self.latched=False;self.trim=self.fp=self.fl=self.fs=self.ts=self.optical=0.;self.hold=None
        self.approach=self.t[next(j for j,n in enumerate(self.nodes) if n['phase']=='approach')]
        self.load_time=self.t[next(j for j,n in enumerate(self.nodes) if n['phase']=='load')]
    def command(self,tick):
        env=self.env;plan=self.plan;sec=(tick+1)*.02
        gravity=env.robot.data.projected_gravity_b.torch[0].cpu().numpy();ang=float(env.robot.data.root_ang_vel_b.torch[0,1].cpu())
        actual_pitch=np.arctan2(gravity[0],-gravity[2]);pref=np.interp(sec,self.t,self.pitch);gref=np.interp(sec,self.t,self.gap);g=env.geometry[0]
        if gref<.04 and g[11] and g[1]-gref>.05:self.latched=True
        if not self.latched and g[11] and g[12]<.21:self.optical=float(np.clip(.9*(g[1]-gref),-.025,.025))
        idx=int(np.abs(self.t-sec).argmin());tau=env.robot.data.applied_torque.torch[0].cpu().numpy()[plan.ids].sum(0)
        if self.approach-.6<sec<self.approach+.4:
            self.bias.append(self.wrench.tables[idx][:,0]-tau);self.wrench.bias=np.median(self.bias,axis=0)
        estimated=self.wrench.estimate(idx,tau)
        self.fl+=.06*(estimated[0]/plan.weight-self.fl);self.fp+=.08*(.72*(actual_pitch-pref)+.12*ang-self.fp)
        self.fs+=.035*(estimated[1]/plan.weight-self.fs)
        if sec>self.load_time:
            self.trim=float(np.clip(self.trim+np.clip(.12*(.20-self.fl),-.015,.015)*.02,-.035,.04))
            self.ts=float(np.clip(self.ts-.14*self.fs*.02,-.045,.045))
        fb=np.clip(self.fp+self.optical+self.trim,-.085,.085);goal=self.base.copy()
        for j,ids in enumerate(plan.ids.T):goal[ids]+=np.interp(sec,self.t,self.x[:,j])-plan.prev[j]
        goal[plan.ids[:,2]]+=fb
        shear=-.5*plan.weight*self.ts*self.wrench.tables[idx][:,2]/np.array([200.,200.,160.])
        for j,ids in enumerate(plan.ids.T):goal[ids]+=shear[j]
        if sec>self.t[-1]+.30 and self.hold is None and .16<self.fl<.25:self.hold=goal.copy()
        if self.hold is not None:goal=self.hold
        # Exactly the previous teacher's bounded increment in absolute-target form.
        previous=env.previous[0,:23].cpu().numpy()
        desired=previous+np.clip(goal-previous,-.003*.99999,.003*.99999)/.1
        return desired,np.r_[sec,actual_pitch,pref,fb,gref,estimated,self.fl,self.trim,self.latched,self.hold is not None]
