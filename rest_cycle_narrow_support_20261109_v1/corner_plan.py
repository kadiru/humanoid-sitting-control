"""Accepted-witness corner perch; measured two-edge contact region and rolling unload."""
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
from scipy.spatial import ConvexHull
from scipy.optimize import least_squares,brentq
from humanoid_sitting.shared_frame import transform,apply
from pivot import Pivot,Mechanics
from corner_schedule import schedule
from rolling import Rolling

class LowestMechanics(Mechanics):
    def support_points(self,root,fk):
        p=apply(root,self.mesh['pelvis']);return np.tile(p[int(np.argmin(p@self.n+self.d))],(2,1))

class CornerPerch(Pivot):
    def __init__(self,model,inertia,shapes,frame,previous,source,corner):
        front=0.;self.patch_sensor=corner;self.edge_error=0.
        from shapely.geometry import Polygon
        self.observed=Polygon(corner['vertices_uv'],corner['holes_uv'])
        self.corner=ConvexHull(np.asarray(corner['vertices_uv'])).equations
        super().__init__(model,inertia,shapes,frame,previous,source,front)
        self.front=front
        self.mechanics=LowestMechanics(model,inertia,shapes,self.q0,frame['gravity'],self.previous,self.n,self.d,self.cloud,'pelvis',self.marker,self.target,'sitting')
        self.mechanics.patch=self.corner.copy();self.mechanics.e=np.asarray(corner['e']);self.mechanics.b=np.asarray(corner['b'])
        self.initial=self.mechanics.state(self.mechanics.x0,0.)
        self.vertices=self.mechanics.mesh['pelvis']
        h=ConvexHull(self.vertices);self.edges=np.unique(np.sort(np.concatenate([h.simplices[:,[0,1]],h.simplices[:,[1,2]],h.simplices[:,[2,0]]]),axis=1),axis=0)
        self.angles=Rotation.from_matrix(self.endR).as_euler('xyz')
        self.evidence=dict(association=dict(source_witness='known_old/16 accepted',target='complete lower2mm pelvis cap in nonconvex directly observed clipped-corner component;22mm reserve after6mm erosion',target_load='full-wrench equilibrium with current foot CoP center; source orientation retained'),stages=[])
    def contact_pose(self,pitch,x):
        a=self.angles.copy();a[1]=pitch;rot=Rotation.from_euler('xyz',a).as_matrix()
        height=float((self.vertices@rot.T@self.n).min())
        z=-(height+self.d+self.n[0]*x)/self.n[2]
        return transform(rot,[x,0.,z])
    def cap_front(self,root):
        p=apply(root,self.vertices);height=p@self.n+self.d;cut=height.min()+.002
        values=list(p[height<=cut,0]);a,b=self.edges.T
        cross=(height[a]-cut)*(height[b]-cut)<0
        aa,bb=a[cross],b[cross]
        u=(cut-height[aa])/(height[bb]-height[aa])
        values.extend((p[aa,0]+u*(p[bb,0]-p[aa,0])).tolist())
        return float(max(values))
    def corner_pose(self):
        rot=self.endR;points=self.vertices@rot.T;height=points@self.n;cut=height.min()+.002
        cap=list(points[height<=cut]);a,b=self.edges.T;cross=(height[a]-cut)*(height[b]-cut)<0
        aa,bb=a[cross],b[cross];u=(cut-height[aa])/(height[bb]-height[aa])
        cap.extend(points[aa]+u[:,None]*(points[bb]-points[aa]));cap=np.asarray(cap)
        from shapely.geometry import Polygon
        e,b=self.mechanics.e,self.mechanics.b;uv=np.c_[cap@e,cap@b]
        center=(uv.min(0)+uv.max(0))/2;radius=float(np.linalg.norm(uv-center,axis=1).max())
        # Circle contains complete convex lower cap. Erode the measured region
        # by its radius plus22mm; retain holes/concavities and6mmpriorerosion.
        feasible=self.observed.buffer(-(radius+.022),resolution=32)
        self.evidence['contact_region']=dict(cap_radius_m=radius,cap_center_uv=center.tolist(),observed_area_m2=float(self.observed.area),eroded_area_m2=float(feasible.area))
        if feasible.is_empty:raise ValueError('CORNER_REGION_TOO_SMALL_FOR_COMPLETE_CONTACT_CAP_AND_RESERVE')
        polys=list(feasible.geoms) if feasible.geom_type=='MultiPolygon' else [feasible]
        if len(polys)>32:raise ValueError('FINITE_CORNER_COMPONENT_CAP')
        candidates=[]
        for p in polys:
            chain=np.asarray(p.exterior.coords)
            for left,right in zip(chain,chain[1:]):
                candidates.append(left)
                l=(left[0]-center[0])*e+(left[1]-center[1])*b-(self.d+height.min())*self.n
                r=(right[0]-center[0])*e+(right[1]-center[1])*b-(self.d+height.min())*self.n
                if l[1]*r[1]<0:candidates.append(left+(right-left)*(-l[1]/(r[1]-l[1])))
        candidates=np.asarray(candidates)
        translations=(candidates[:,0,None]-center[0])*e+(candidates[:,1,None]-center[1])*b-(self.d+height.min())*self.n
        in_box=(translations[:,0]>=-.5)&(translations[:,0]<=.15)&(abs(translations[:,1])<=.2)
        if not in_box.any():raise ValueError('OBSERVED_CORNER_OUTSIDE_CONSTRUCTION_DOMAIN')
        valid=np.flatnonzero(in_box);forward=translations[valid,0].max();near=valid[translations[valid,0]>=forward-1e-8]
        chosen=near[np.argmin(abs(translations[near,1]))];translation=translations[chosen]

        actual=Polygon(uv[ConvexHull(uv).vertices]+candidates[chosen]-center)
        if not self.observed.covers(actual) or actual.distance(self.observed.boundary)<.022-1e-8:raise ValueError('CIRCLE_EROSION_CONTACT_COVERAGE_FAILED')
        self.evidence['contact_region'].update(target_center_uv=candidates[chosen].tolist(),actual_cap_boundary_distance_m=float(actual.distance(self.observed.boundary)),root=translation.tolist())
        return transform(rot,translation)

    def construct(self):
        # Register the accepted posture to the actual two-edge support region.
        # Solve only the twelve leg coordinates; no rejected endpoint is a seed.
        restroot=self.corner_pose();restq=self.fit_feet(self.restq,restroot,'accepted_witness_corner_registration')
        pitch=float(self.angles[1]);rest_load=self.load_for_cop(restq,restroot)
        self.evidence['registered_rest']=dict(load=rest_load,pitch=pitch,root=restroot.tolist(),q=restq.tolist(),edge_reserve_m=.022+self.edge_error)
        if not .10<=rest_load<=.8:raise ValueError(('CORNER_SUPPORT_LOAD_OUTSIDE_RANGE',rest_load))
        roll=Rolling(self.vertices,self.n,self.angles,pitch,np.deg2rad(38))
        self.evidence['rolling_geometry']=roll.audit(restroot[:3,3],self.d)
        if not self.evidence['rolling_geometry']['passed']:raise ValueError('ROLLING_GEOMETRY_PROOF_FAILED')
        def rolling_pose(theta):return transform(roll.rotation(theta),roll.translation(theta,restroot[:3,3]))
        root_evaluations=0
        def load(theta):
            nonlocal root_evaluations
            root_evaluations+=1
            rt=rolling_pose(theta);q=self.fit_feet(restq,rt,'rolling_load_bracket_'+str(root_evaluations))
            return self.load_for_cop(q,rt)
        left,right=load(pitch),load(np.deg2rad(38))
        self.evidence['unload_bracket']=dict(start_load=left,limit_load=right)
        if left<=0 or right>=0:raise ValueError('ROLLING_UNLOAD_NOT_BRACKETED')
        end=brentq(load,pitch,np.deg2rad(38),xtol=1e-9,maxiter=24)
        angles=sorted(set(np.linspace(pitch,end,9).tolist()+[a for a,b,c in roll.segments if pitch<a<end]))
        pivot=[];q=restq.copy()
        for k,theta in enumerate(angles):
            rt=rolling_pose(theta);q=self.fit_feet(q,rt,'rolling_unload_'+str(k));fraction=self.load_for_cop(q,rt)
            if abs(fraction)<1e-5:fraction=0.
            if not 0<=fraction<=rest_load+1e-5:raise ValueError(('ROLLING_LOAD_NONMONOTONE',k,fraction))
            pivot.append(self.node(q,rt,fraction,'landing'))
        br=rt.copy();normal=[]
        for k,distance in enumerate(np.linspace(0,.012,5)):
            rt=br.copy();rt[:3,3]+=distance*self.n
            q=self.fit_feet(q,rt,'normal_retraction_'+str(k));normal.append(self.node(q,rt,0.,'landing',True))
        hover=rt.copy();hq=q.copy();lift=[normal[-1]]
        interpolation=Slerp([0,1],Rotation.from_matrix(np.stack([hover[:3,:3],self.R0])))
        for k,u in enumerate(np.linspace(0,1,17)[1:],1):
            rt=transform(interpolation(u).as_matrix(),(1-u)*hover[:3,3]);qr=(1-u)*hq+u*self.q0
            target_com=(1-u)*self.copx+u*self.initial['com'][0]
            def pose_lift(y):
                qi=qr.copy();qi[self.legs]=y[:-1];rr=rt.copy();rr[0,3]=y[-1];return qi,rr
            def residual_lift(y):
                qi,rr=pose_lift(y);return np.r_[self.feet_residual(qi,rr),self.state(qi,rr)['com'][0]-target_com]
            fit=least_squares(residual_lift,np.r_[q[self.legs],rt[0,3]],bounds=(np.r_[self.lo,-.5],np.r_[self.hi,.15]),max_nfev=60,xtol=1e-9,ftol=1e-9,gtol=1e-9)
            err=float(abs(fit.fun).max());self.evidence['stages'].append(dict(stage='balanced_lift_'+str(k),success=bool(fit.success),nfev=int(fit.nfev),max_residual=err))
            if not fit.success or err>2e-6:raise ValueError(('BALANCED_LIFT_FAILED',k,err))
            q,rt=pose_lift(fit.x);lift.append(self.node(q,rt,0.,'approach'))
        if np.max(abs(np.asarray(lift[-1]['q'])-self.q0))>2e-5 or np.max(abs(np.asarray(lift[-1]['root'])-self.mechanics.root0))>2e-5:raise ValueError('RETURN_NOT_ORIGINAL_STANDING')
        lift[-1].update(q=self.q0.tolist(),root=self.mechanics.root0.tolist(),command=self.previous.tolist())
        approach=list(reversed(lift))+list(reversed(normal))[1:]+list(reversed(pivot))[1:]
        returning=pivot+normal[1:]+lift[1:]
        forward=schedule(approach,self.mechanics.lower,self.mechanics.upper);backward=schedule(returning,self.mechanics.lower,self.mechanics.upper)
        return dict(nodes=approach,return_nodes=returning,forward=forward,backward=backward,
            nominal_rest_settle_s=forward['seconds']+1.5,nominal_return_settle_s=backward['seconds']+2.5,
            rest_load=rest_load,rest_model_effort_ratio=approach[-1]['model_effort_ratio'],association=self.evidence['association'],
            rolling=self.evidence['rolling_geometry'],runtime_load_observer_admitted=False,physical_success=False)
