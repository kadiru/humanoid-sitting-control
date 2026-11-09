"""Support first, then reduce model effort on a fixed material contact point.

Current camera/encoders plus static geometry only. Contact work differentiates
the SAME material body point, not a changing clipped support vertex. Forces are
planning variables, never represented as measured load. Physical gates remain.
"""
import importlib.util,json,time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
from humanoid_sitting.shared_frame import apply

path=Path(__file__).resolve().parent.parent/'furniture_saved_repair_20261107_v2/finite_back/finite_geometry.py'
spec=importlib.util.spec_from_file_location('finite_back_geometry',path);base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)

class LeanPlan(base.LeanGeometry):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        rt,_=self.pose(self.start)
        _,point,_,_=super().geometry(self.start)
        self.marker=(point-rt[:3,3])@rt[:3,:3]
        self.initial_root_height=float(rt[2,3]);self.leg_length=self.initial_root_height-self.floor
        self.mid=float(np.mean(self.foot_bounds));self.radius=float(np.ptp(self.foot_bounds)/2)
        self.joints={j['name']:j for j in self.model.joints}
        self.desc={}
        for ids in self.ids:
            for i in ids:
                name=self.model.names[i];bodies={self.joints[name]['body1'][0]}
                while True:
                    new={j['body1'][0] for j in self.model.joints if j['body0'] and j['body0'][0] in bodies}
                    if new<=bodies:break
                    bodies|=new
                self.desc[name]=bodies
        self.cached_key=None;self.cached_state=None
        self.calls=0;self.optimizations=0

    def state(self,x):
        key=np.asarray(x).tobytes()
        if key==self.cached_key:return self.cached_state
        rt,fk=self.pose(x);p=apply(rt,self.marker)
        centers=np.array([apply(rt@fk[b],c) for b,c in self.local])
        com=np.average(centers,weights=self.mass,axis=0)
        g=[];w=[]
        for k in range(3):
            frames=[]
            for ids in self.ids:
                name=self.model.names[ids[k]];j=self.joints[name]
                frame=rt@fk[j['body0'][0]]@self.model.frames[name][0]
                frames.append((frame[:3,1],frame[:3,3],self.desc[name]))
            a,o,_=frames[0]
            vel=-np.cross(a,centers-o)
            for axis,origin,desc in frames:
                selected=np.array([b in desc for b,c in self.local])
                vel[selected]+=np.cross(axis,centers[selected]-origin)
            g.append(self.weight*np.average(vel[:,2],weights=self.mass))
            w.append(self.weight*self.n@(-np.cross(a,p-o)))
        self.calls+=1
        value=dict(root=rt,fk=fk,point=p,com=com,gap=float(p@self.n+self.off),height=float(p@self.up),pitch=float(Rotation.from_matrix(rt[:3,:3]).as_euler('xyz')[1]),gravity_work=np.asarray(g),normal_work=np.asarray(w))
        self.cached_key=key;self.cached_state=value
        return value

    def material_bias(self,x,load):
        s=self.state(x)
        return .5*(s['gravity_work']-load*s['normal_work'])/np.array([200.,200.,160.])

    def cop(self,s,load):
        p=s['point'];c=s['com']
        return float((c[0]+load*((p[2]-self.floor)*self.n[0]-p[0]*self.n[2]))/(1-load*self.n[2]))

    def fixed_material_work_audit(self,x,load):
        # Independent central differences of total potential and external work;
        # root/whole-body kinematics are recomputed, marker is held body-fixed.
        values=[]
        for dx in np.eye(3)*1e-5:
            energy=[]
            for y in [x-dx,x+dx]:
                rt,fk=self.pose(y)
                U=sum(m*9.81*apply(rt@fk[b],c)[2] for m,(b,c) in zip(self.mass,self.local))
                energy.append(U-load*self.weight*self.n@apply(rt,self.marker))
            values.append((energy[1]-energy[0])/2e-5)
        s=self.state(x);analytic=s['gravity_work']-load*s['normal_work']
        error=float(np.max(abs(analytic-values)))
        if error>2e-6:raise ValueError(('MATERIAL_WORK_AUDIT_FAILED',error))
        return dict(passed=True,maximum_generalized_torque_error_Nm=error,finite_difference_only_in_audit=True)

    def solve(self):
        deadline=time.monotonic()+45.;start=self.state(self.start)
        self.work_audit=self.fixed_material_work_audit(self.start,.20)
        # Preserve the hardware/action limits. Positive knee branch is an extra
        # construction restriction, excluding the rejected hyperextended pose.
        lo=self.lower.copy();lo[1]=max(lo[1],.04)
        initial_bias=self.prev-self.start-self.material_bias(self.start,0.)
        nodes=[];x=self.start.copy();hover_height=None
        def constraints(y,load,preload):
            s=self.state(y);cmd=y+self.material_bias(y,load)+preload
            cop=self.cop(s,load);rt=s['root'];fk=s['fk']
            pelvis=apply(rt@fk['pelvis'],self.mesh['pelvis'])
            # Conservative vertices on the actually observed finite back only.
            inside=np.all(pelvis@self.clip_world[:,:3].T+self.clip_world[:,3]<=0.,axis=1)
            clearance=float(np.min(pelvis[inside]@self.n+self.off)-.006) if inside.any() else 1.
            return np.r_[s['height']-self.band[0]-.04,self.band[1]-.04-s['height'],cop-self.foot_bounds[0]-.01,self.foot_bounds[1]-.01-cop,.35-s['pitch'],.35+s['pitch'],clearance,cmd-self.lower-.02,self.upper-.02-cmd]
        def objective(y,load):
            if time.monotonic()>deadline:raise TimeoutError('45_SECOND_CONSTRUCTION_CAP')
            s=self.state(y);tau=.5*(s['gravity_work']-load*s['normal_work'])
            effort=float(np.sum((tau/np.array([300.,300.,20.]))**2))
            balance=((self.cop(s,load)-self.mid)/self.radius)**2
            height=(max(0.,self.initial_root_height-s['root'][2,3])/self.leg_length)**2
            ready=float(np.sum(((y-self.start)/np.pi)**2))
            return effort+balance+height+ready
        phases=[('approach',24),('normal',4),('load',12)]
        for phase,steps in phases:
            for i in range(steps+1):
                if nodes and i==0:continue
                u=i/steps
                gap=(1-u)*start['gap']+u*.012 if phase=='approach' else .012*(1-u) if phase=='normal' else 0.
                load=.20*u if phase=='load' else 0.
                preload=(1-u)*initial_bias if phase=='approach' else np.zeros(3)
                def equality(y):
                    s=self.state(y)
                    return np.array([s['gap']-gap]) if phase=='approach' else np.array([s['gap']-gap,s['height']-hover_height])
                if phase=='approach' and i==0:
                    accepted=x.copy()
                else:
                    self.optimizations+=1
                    fit=minimize(objective,x.copy(),args=(load,),method='SLSQP',bounds=list(zip(lo,self.upper)),constraints=[dict(type='eq',fun=equality),dict(type='ineq',fun=lambda y:constraints(y,load,preload))],options={'maxiter':60,'ftol':1e-9})
                    if not fit.success or max(abs(equality(fit.x)))>.002 or constraints(fit.x,load,preload).min()<-1e-6:
                        raise ValueError(dict(reason='SUPPORT_FIRST_NODE_REJECTED',phase=phase,node=i,message=str(fit.message),equality=equality(fit.x).tolist(),minimum_inequality=float(constraints(fit.x,load,preload).min())))
                    accepted=fit.x
                x=accepted;s=self.state(x);cmd=x+self.material_bias(x,load)+preload
                nodes.append(dict(q=x.tolist(),cmd=cmd.tolist(),gap=s['gap'],height=s['height'],pitch=s['pitch'],cop=self.cop(s,load),load=load,phase=phase,material_point=s['point'].tolist(),model_objective=objective(x,load)))
                if phase=='approach' and i==steps:hover_height=s['height']
        commands=np.array([n['cmd'] for n in nodes]);delta=np.max(abs(np.diff(commands,axis=0)),axis=1)
        durations=np.maximum(delta/.125,[.5 if n['phase']=='normal' else .19 if n['phase']=='load' else .06 for n in nodes[1:]])
        # Current quiet gate remains required, this reserve is not a timer bypass.
        durations[next(i-1 for i,n in enumerate(nodes) if n['phase']=='normal')]+=.5
        times=np.r_[0.,np.cumsum(durations)]
        self.provisional_path=dict(nodes=nodes,times=times.tolist(),contact_work_audit=self.work_audit)
        if times[-1]>=14.4:raise ValueError(('ORIGINAL_HOLD_RESERVE_EXCEEDED',float(times[-1])))
        final=self.state(x)
        self.posture_selection=dict(initial_root_height=self.initial_root_height,final_root_height=float(final['root'][2,3]),final_knee_rad=float(x[1]),final_pitch_rad=final['pitch'],model_effort_only=True,return_path='Reverse same fixed-contact equilibrium states while unloading, then reverse unloaded approach; dynamics not yet validated',model_return_seconds=float(times[-1]),state_evaluations=self.calls,optimizer_invocations=self.optimizations,material_point=self.marker.tolist(),contact_work_audit=self.work_audit)
        return nodes,times

    def geometry(self,x):
        if not hasattr(self,'marker'):return super().geometry(x)
        s=self.state(x)
        return s['gap'],s['point'],s['com'],s['pitch']
