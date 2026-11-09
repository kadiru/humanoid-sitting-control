"""Fixed-foot sagittal planner. Coordinates use startup pelvis and measured gravity."""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares, minimize_scalar
from scipy.spatial.transform import Rotation
from humanoid_sitting.shared_frame import apply, level_rotation, transform

ROOT=Path('/home/kadir/humanoid-sitting')
INERTIA=Path('/data/humanoid-sitting-perception/crouch_rest_bridge_01/balanced_standing10_01/static_inertial_model.json')
SHAPES=ROOT/'experiments/furniture_perception_repair_20261029_v1/render_colliders.json'
SIDES=('left','right')
JOINTS=('hip_pitch','knee','ankle_pitch')

class LeanPlan:
    def __init__(self,model,q,gravity,geometry,points,previous):
        self.model=model
        self.q=np.asarray(q,dtype=float).copy()
        self.ids=np.array([[model.names.index(f'{s}_{j}_joint') for j in JOINTS] for s in SIDES])
        self.start=q[self.ids].mean(0)
        self.prev=previous[self.ids].mean(0)
        self.level=level_rotation(np.asarray(gravity,dtype=float).copy())
        f=model.fk(q)
        self.anchor=transform(self.level)@f['left_ankle_roll_link']
        self.n=self.level@geometry[2:5].astype(float)
        self.n/=np.linalg.norm(self.n)
        self.off=geometry[5]
        self.cloud=np.asarray(points)@self.level.T
        self.up=np.array([0.,0.,1.])-self.n*self.n[2];self.up/=np.linalg.norm(self.up)
        v=self.cloud@self.up
        self.band=np.percentile(v,[2,98])
        inertia=json.loads(INERTIA.read_text())
        self.mass=np.array(inertia['mass'])
        self.local=list(zip(inertia['body_names'],np.array(inertia['local_com'])))
        inventory=json.loads(SHAPES.read_text())['shapes']
        self.mesh={name:np.concatenate([s['vertices'] for s in inventory if s['body']==name]) for name in {s['body'] for s in inventory}}
        self.trunk={b:self.mesh[b] for b in ['torso_link','logo_link']}
        self.lower=np.array([max(np.deg2rad(j['limits'][0]) for j in model.joints if j['name'] in [model.names[k] for k in ids])+.031 for ids in self.ids.T])
        self.upper=np.array([min(np.deg2rad(j['limits'][1]) for j in model.joints if j['name'] in [model.names[k] for k in ids])-.031 for ids in self.ids.T])
        self.sole=apply(self.anchor,self.mesh['left_ankle_roll_link'])
        self.floor=self.sole[:,2].min()
        self.foot_bounds=np.percentile(self.sole[self.sole[:,2]<self.floor+.004,0],[0,100])
        self.weight=self.mass.sum()*9.81

    def pose(self,x):
        q=self.q.copy();q[self.ids]+=x-self.start
        f=self.model.fk(q)
        root=self.anchor@np.linalg.inv(f['left_ankle_roll_link'])
        return root,f

    def geometry(self,x):
        root,f=self.pose(x)
        rotation=root[:3,:3]
        # Torso/cover are rigid relative to pelvis for a sagittal leg maneuver.
        points=np.concatenate([apply(root@f[b],v) for b,v in self.trunk.items()])
        h=points@self.n+self.off
        tip=points[h.argmin()]
        com=apply(root,np.average([apply(f[b],c) for b,c in self.local],weights=self.mass,axis=0))
        return float(h.min()),tip,com,float(Rotation.from_matrix(rotation).as_euler('xyz')[1])

    def feedforward(self,x,load):
        deriv=[]
        for dx in np.eye(3)*.0001:
            g1,_,c1,_=self.geometry(x+dx);g0,_,c0,_=self.geometry(x-dx)
            deriv.append(((c1[2]-c0[2])-load*(g1-g0))/.0002)
        return .5*self.weight*np.array(deriv)/[200.,200.,160.]

    def solve(self):
        g0,p0,c0,pitch0=self.geometry(self.start)
        # Leave a real interior buffer for finite contact width, small angle and
        # tracking errors. Approximate the highest likely contacting hull point.
        # Use the upper rear torso facet, whose outward normal was measured
        # from the frozen hull. This clears the light rear-cover collider.
        desired_pitch=np.arctan2(self.n[0],self.n[2])-np.arctan2(.947,-.322)
        ht=self.band[1]-.020
        # At a support-function vertex change, contact height is nonsmooth.
        # Estimate the rigid body's prospective contact landmark at final pitch.
        root,f=self.pose(self.start)
        verts=np.concatenate([apply(f[b],v) for b,v in self.trunk.items()])
        # The head and the cover compete for clearance at the upper-back
        # contact. Rotate away from the ideal facet normal only as far as
        # needed to maximize the smaller of the two clearances on the
        # finite observed face. Account for the top rim as well.
        trunk=apply(f['torso_link'],self.trunk['torso_link'])
        cover=apply(f['logo_link'],self.trunk['logo_link'])
        head=apply(f['head_link'],self.mesh['head_link'])
        def landmark(a):
            rot=Rotation.from_euler('y',a).as_matrix()
            pr=trunk@rot.T;dist=pr@self.n
            ids=np.flatnonzero(dist<dist.min()+.003)
            mark=ids[(pr[ids]@self.up).argmax()]
            return rot,trunk[mark],pr[mark]@self.up,dist.min()
        def clearance(a):
            rot,_,h,d=landmark(a)
            hd=head@rot.T
            near=hd@self.up-h+ht<self.band[1]+.025
            return min((hd[near]@self.n).min(initial=1)-d,(cover@rot.T@self.n).min()-d)
        desired_pitch=minimize_scalar(lambda a:-clearance(a),bounds=(desired_pitch,0.),method='bounded').x
        _,marker,_,_=landmark(desired_pitch)
        def residual(x,gap,height,pitch):
            rt,ff=self.pose(x)
            pt=apply(rt,marker)
            return [pt@self.n+self.off-gap,pt@self.up-height,.35*(Rotation.from_matrix(rt[:3,:3]).as_euler('xyz')[1]-pitch)]
        def solve_one(x,gap,h,pitch):
            fit=least_squares(residual,x,bounds=(self.lower,self.upper),args=(gap,h,pitch),max_nfev=60)
            if np.max(np.abs(fit.fun))>.002: raise ValueError(('unreachable',fit.fun,fit.x,gap,h,pitch))
            return fit.x
        initial=apply(root,marker)
        nodes=[]; seed=self.start.copy()
        # First descend with a positive wall gap, then ease to contact at fixed
        # height. Load ramp starts once the geometric gap has closed.
        bias=self.prev-self.start-self.feedforward(self.start,0)
        for phase,steps in [('lower',32),('approach',20),('load',12)]:
            for i in range(steps+1):
                if nodes and i==0:continue
                u=i/steps
                if phase=='lower':
                    gap=(1-u)*(initial@self.n+self.off)+u*.018
                    height=(1-u)*(initial@self.up)+u*ht
                    pitch=(1-u)*pitch0
                else:
                    gap=.018*(1-u) if phase=='approach' else 0.
                    height=ht;pitch=u*desired_pitch if phase=='approach' else desired_pitch
                load=.10*u if phase=='load' else 0
                seed=solve_one(seed,gap,height,pitch)
                realgap,tip,com,actualpitch=self.geometry(seed)
                # At final pitch the marker may be up to 3 mm ahead of the
                # actual tangent point. Correct the gap for that offset.
                if phase!='lower':
                    seed=solve_one(seed,gap+(gap-realgap),height,pitch)
                    realgap,tip,com,actualpitch=self.geometry(seed)
                cop=(com[0]+load*((tip[2]-self.floor)*self.n[0]-tip[0]*self.n[2]))/(1-load*self.n[2])
                cmd=seed+self.feedforward(seed,load)+bias*(1-u if phase=='lower' else 0.)
                nodes.append(dict(q=seed.tolist(),cmd=cmd.tolist(),gap=realgap,height=float(tip@self.up),pitch=actualpitch,cop=float(cop),load=load,phase=phase))
        cmd=np.array([n['cmd'] for n in nodes])
        diff=np.max(abs(np.diff(cmd,axis=0)),axis=1)
        # Piecewise arc-length timing leaves actuator and balance-feedback authority.
        times=np.r_[0,np.cumsum(np.maximum(diff/.125,[.06 if n["phase"]=="lower" else .14 for n in nodes[1:]]))]
        return nodes,times
