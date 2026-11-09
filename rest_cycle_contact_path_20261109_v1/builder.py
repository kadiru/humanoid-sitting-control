"""Fresh standing-to-contact continuation using an accepted physical witness.

The witness supplies contact/hold orientations, never an old planned path or
failed iterate. Current rear-camera geometry supplies the support plane. Feet
are closed at every control; support motion follows the exact rolling curve.
"""
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
from humanoid_sitting.shared_frame import apply,transform,level_rotation
from static_model import FullRest
from rolling import Rolling
from continuation import FixedFeet
from control_path import assemble,rolling_root,bind_cycle_stages


class Mechanics(FullRest):
    support_bodies=('pelvis','pelvis')
    def __init__(self,*a):
        self.support_normals=np.tile(np.asarray(a[6]),(2,1))
        super().__init__(*a)
    def support_points(self,root,fk):
        p=apply(root,self.mesh['pelvis'])
        return np.tile(p[int(np.argmin(p@self.n+self.d))],(2,1))


class ContactPath:
    APPROACH=380
    NORMAL=100
    ROLL=160

    def __init__(self,model,inertia,shapes,frame,previous,patch,reference):
        self.model=model;self.q0=np.asarray(frame['q'],float);self.previous=np.asarray(previous,float)
        self.R0=level_rotation(np.asarray(frame['gravity'],float).copy())
        geometry=frame['geometry'];self.n=self.R0@geometry[2:5];self.n/=np.linalg.norm(self.n)
        self.d=float(geometry[5])
        if not geometry[11] or geometry[12]>.21 or self.n[2]<=.97:raise ValueError('QUALIFIED_CAMERA_PLANE_REQUIRED')
        self.vertices=np.concatenate([s['vertices'] for s in shapes if s['body']=='pelvis'])
        alignment=Rotation.align_vectors(self.n[None],np.asarray(reference['normal'])[None])[0].as_matrix()
        rest=alignment@np.asarray(reference['held_rotation'])
        touch=alignment@np.asarray(reference['contact_rotation'])
        self.angles=Rotation.from_matrix(rest).as_euler('xyz')
        self.rest_pitch=float(self.angles[1]);self.touch_pitch=float(Rotation.from_matrix(touch).as_euler('xyz')[1])
        if not -.15<=self.rest_pitch<self.touch_pitch<=np.deg2rad(38):
            raise ValueError('ACCEPTED_WITNESS_DOES_NOT_DEFINE_FORWARD_UNLOADED_CONTACT_ORIENTATION')
        # Keep the accepted hold roll/yaw throughout contact; only the witnessed
        # contact pitch is transferred. No previous planned endpoint is loaded.
        self.rolling=Rolling(self.vertices,self.n,self.angles,self.rest_pitch,self.touch_pitch)
        self.touch_rotation=self.rolling.rotation(self.touch_pitch)
        cloud=np.asarray(frame['selected_points'])@self.R0.T
        self.mechanics=Mechanics(model,inertia,shapes,self.q0,frame['gravity'],previous,self.n,self.d,
                                cloud,'pelvis',self.vertices[0],np.zeros(3),'sitting')
        self.mechanics.e=np.asarray(patch['e']);self.mechanics.b=np.asarray(patch['b'])
        from scipy.spatial import ConvexHull
        from shapely.geometry import Polygon
        self.mechanics.patch=ConvexHull(np.asarray(patch['vertices_uv'])).equations
        self.observed=Polygon(patch['vertices_uv'],patch['holes_uv'])
        hull=ConvexHull(self.vertices)
        self.edges=np.unique(np.sort(np.concatenate([hull.simplices[:,[0,1]],hull.simplices[:,[1,2]],hull.simplices[:,[2,0]]]),axis=1),axis=0)
        self.closure=FixedFeet(model,inertia,self.q0,self.mechanics.root0,self.mechanics.lower,self.mechanics.upper)
        self.initial=self.mechanics.state(self.mechanics.x0,0.)
        self.initial_gap=float((apply(self.mechanics.root0,self.vertices)@self.n+self.d).min())
        if self.initial_gap<=.012:raise ValueError('STANDING_START_HAS_NO_HOVER_SEPARATION')
        self.nodes=[];self.trace=[];self.stage='initialization'
        self.association=dict(source='accepted physical known_old/16 only',reference=reference,
            target='current camera plane and measured nonconvex patch; initial two physical foot anchors',
            mechanism='fixed-foot and COM continuation at every control, exact rolling after contact, same-state torque commands',
            phase_controls=[self.APPROACH,self.NORMAL,self.ROLL],old_plan_or_failed_iterate_loaded=False)

    def x(self,q,root):
        return np.r_[q[:23],root[:3,3],Rotation.from_matrix(self.R0.T@root[:3,:3]).as_rotvec(),0.]

    def node(self,q,root,load,phase,normal=False):
        if not 0<=load<=.8:raise ValueError(('SUPPORT_FRACTION_OUTSIDE_ORIGINAL_RANGE',float(load)))
        state=self.mechanics.state(self.x(q,root),load)
        if any(not np.isfinite(np.asarray(state[k])).all() for k in ('foot_error','tau','base_residual','com','foot_forces')):
            raise ValueError('NONFINITE_MECHANICS')
        gap=float((apply(root,self.vertices)@self.n+self.d).min())
        if max(abs(state['foot_error']))>2e-5 or gap < -1e-6:raise ValueError('FOOT_OR_SUPPORT_PLANE_FAILURE')
        if root[2,2]<np.cos(np.deg2rad(38)):raise ValueError('ORIGINAL_TILT_LIMIT')
        margins=[]
        for eq,point,force in zip(self.mechanics.foot_polygons,state['foot_points'],state['foot_forces']):
            margins.extend(-(eq[:,:2]@point[:2]+eq[:,2])-.01)
            margins.extend([force[2]/self.mechanics.W-.05,(.5*force[2]-np.linalg.norm(force[:2]))/self.mechanics.W])
        if min(margins)<-1e-8:raise ValueError(('ORIGINAL_FOOT_BALANCE_MARGIN',float(min(margins))))
        blend=float(np.clip((gap-.012)/(self.initial_gap-.012),0,1))
        command=self.previous.copy();command[:23]=q[:23]+state['tau']/self.mechanics.K+blend*self.mechanics.initial_bias
        if not self.nodes:command=self.previous.copy()
        if self.nodes:
            prior=np.asarray(self.nodes[-1]['command']);delta=command-prior
            desired=prior[:23]+10*delta[:23]
            if max(abs(delta))>.003+1e-12:raise ValueError(('ORIGINAL_COMMAND_SPEED_EXCEEDED',float(max(abs(delta)))))
            if np.any(desired<self.mechanics.lower+.03) or np.any(desired>self.mechanics.upper-.03):
                raise ValueError('ORIGINAL_INVERSE_FILTER_BOX_EXCEEDED')
        node=dict(q=q.tolist(),root=root.tolist(),command=command.tolist(),load=float(load),gap=gap,
                  marker_gap=float(state['point']@self.n+self.d),model_cop=state['cop'].tolist(),
                  model_tau=state['tau'].tolist(),model_effort_ratio=float(state['tau']@state['tau']/self.mechanics.standing_effort),
                  phase=phase,normal=normal)
        self.nodes.append(node)
        return state

    def construct(self):
        q=self.q0.copy();root=self.mechanics.root0.copy()
        self.node(q,root,0.,'approach')
        rotations=Slerp([0,1],Rotation.from_matrix(np.stack([self.R0,self.touch_rotation])))
        copx=float(np.mean([a[0,3] for a in self.closure.anchors]))
        direction=np.array([1.,0.,-self.n[0]/self.n[2]])
        self.stage='balanced_approach'
        for k in range(1,self.APPROACH+1):
            u=k/self.APPROACH;R=rotations(u).as_matrix()
            gap=(1-u)*self.initial_gap+u*.012
            z=(gap-self.d-min(self.vertices@R.T@self.n)-self.n[0]*root[0,3])/self.n[2]
            root=transform(R,[root[0,3],0.,z])
            q,root,info=self.closure.solve(q,root,direction=direction,
                target_com_x=(1-u)*self.initial['com'][0]+u*copx)
            self.trace.append(dict(stage=self.stage,index=k,**info));self.node(q,root,0.,'approach')
        hover=root.copy();self.stage='normal_contact'
        for k in range(1,self.NORMAL+1):
            root=hover.copy();root[:3,3]-=self.n*(.012*k/self.NORMAL)
            q,root,info=self.closure.solve(q,root)
            self.trace.append(dict(stage=self.stage,index=k,**info));touch=self.node(q,root,0.,'landing',True)
        copx=float(touch['com'][0])
        touch_position=root[:3,3].copy()
        self.stage='rolling_load_transfer'
        for k in range(1,self.ROLL+1):
            theta=self.touch_pitch+(self.rest_pitch-self.touch_pitch)*k/self.ROLL
            root=rolling_root(self.rolling,theta,self.touch_pitch,touch_position)
            q,root,info=self.closure.solve(q,root)
            state=self.mechanics.state(self.x(q,root),0.)
            point=state['point'];denominator=(point[2]-self.mechanics.floor)*self.n[0]+(copx-point[0])*self.n[2]
            if abs(denominator)<1e-6:raise ValueError('DEGENERATE_SUPPORT_LEVER')
            load=(copx-state['com'][0])/denominator
            self.trace.append(dict(stage=self.stage,index=k,**info));self.node(q,root,load,'landing')
        if self.nodes[-1]['load']<.05:raise ValueError('INSUFFICIENT_MODELED_WEIGHT_TRANSFER')
        if self.nodes[-1]['model_effort_ratio']>=1:raise ValueError('NO_MODELED_EFFORT_REDUCTION')
        result=assemble(self.nodes,self.mechanics.lower,self.mechanics.upper,self.association)
        bind_cycle_stages(result,self.APPROACH,self.APPROACH+self.NORMAL)
        result['closure_resources']=dict(FK_evaluations=self.closure.calls,linear_system_solves=self.closure.linear_solves,
                                         terminal_optimizer_calls=0,maximum_newton_steps_per_control=8)
        return result
