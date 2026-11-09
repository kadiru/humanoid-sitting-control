"""Construct contact on the evaluator's existing pelvis/proximal-thigh bodies."""
import numpy as np
from humanoid_sitting.shared_frame import apply
from seat_plan import SeatPlan as BasePlan
from contact_surface import associate


class SeatPlan(BasePlan):
    def __init__(self,model,inertia,shapes,q,gravity,geometry,points,previous,intent):
        value,local,self.patch_report,self.cloud_samples,self.height_field,_=associate(model,q,gravity,geometry,points,intent)
        super().__init__(model,inertia,shapes,q,gravity,value,local,previous)
        self.refined_geometry=value
        self.rest_mesh={'pelvis':self.mesh['pelvis']}
        for body in ['left_hip_yaw_link','right_hip_yaw_link']:
            v=self.mesh[body]
            self.rest_mesh[body]=v[v[:,2]>=.5*(v[:,2].min()+v[:,2].max())]
        self.contact_target=None
        initial=self.state(self.z0);self.start_tip=float(initial[5]@self.e);self.start_gap=float(initial[4].min())

    def support(self,q,root,fk=None):
        fk=self.model.fk(q) if fk is None else fk
        points=np.concatenate([apply(root@fk[body],v) for body,v in self.rest_mesh.items()])
        heights=points@self.n+self.d
        return points,points[heights.argmin()],heights

    def state(self,z):
        q,root,fk,pelvis,heights,tip,com=super().state(z)
        if hasattr(self,'rest_mesh'):
            points,tip,heights=self.support(q,root,fk)
            return q,root,fk,points,heights,tip,com
        return q,root,fk,pelvis,heights,tip,com

    def residual(self,z,gap,tangent,balance):
        result=super().residual(z,gap,tangent,balance)
        q,root,fk,points,heights,tip,com=self.state(z)
        if self.contact_target is None:
            patch=points[heights<heights.min()+.002]
            uv=np.c_[patch@self.e,patch@self.b]
            inset=-(uv@self.hull.equations[:,:2].T+self.hull.equations[:,2])
            result[7]=inset.min()-.022
        else:
            s=max(self.progress,self.arm_reach)
            # A normal landing supplies its explicit final tangent unchanged.
            progress=max(0.,(s-.25)/.75)
            goal=self.contact_target if getattr(self,'landing_active',False) else (1-progress)*self.start_tip+progress*self.contact_target
            result[7]=tip@self.e-goal
        return result

    def fit(self,seed,gap,tangent,balance):
        if self.contact_target is not None and not getattr(self,'landing_active',False):
            rise=min(1.,max(self.progress,self.arm_reach)/.25)
            gap=(1-rise)*self.start_gap+rise*.012
        z=super().fit(seed,gap,tangent,balance)
        if self.contact_target is None:self.contact_target=float(self.state(z)[5]@self.e)
        return z

    def solve(self):
        result=super().solve();result['target_tangent']=self.contact_target
        result['target_rule']='existing_legal_pelvis_or_proximal_thigh_patch_with22mmobserved_inset'
        result['support_bodies']=list(self.rest_mesh)
        return result
