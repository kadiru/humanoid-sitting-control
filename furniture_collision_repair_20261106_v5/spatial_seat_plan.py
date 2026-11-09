"""Fresh sensor-associated pose construction with spatial housing witnesses."""
import numpy as np
from humanoid_sitting.shared_frame import apply
from seat_plan import SeatPlan as BasePlan
from contact_surface import associate


class SeatPlan(BasePlan):
    def __init__(self,model,inertia,shapes,q,gravity,geometry,points,previous,intent):
        value,local,self.patch_report,self.observed_cloud,self.observed_height,self.central_target=associate(model,q,gravity,geometry,points,intent)
        super().__init__(model,inertia,shapes,q,gravity,value,local,previous)
        self.refined_geometry=value
        self.start_tangent=float(self.state(self.z0)[5]@self.e)

    def residual(self,z,gap,tangent,balance):
        if self.central_target is not None:
            # The startup tangent remains the original measured pose. The
            # endpoint/path target is the associated central sensor triangle.
            s=max(self.progress,self.arm_reach)
            tangent=(1-s)*self.start_tangent+s*(self.central_target@self.e)
        original=super().residual(z,gap,tangent,balance)
        q,root,fk,*_=self.state(z);deficits=[]
        for body in ['left_hip_pitch_link','right_hip_pitch_link']:
            vertices=apply(root@fk[body],self.mesh[body])
            surface=np.asarray(self.observed_height(vertices[:,:2]));inside=np.isfinite(surface)
            # Outside the measured domain is never certified safe by this
            # heuristic. The full original mesh audit remains mandatory.
            minimum=float(np.min(vertices[inside,2]-surface[inside])) if inside.any() else 1.
            required=.008+.015*max(self.progress,self.arm_reach)
            deficits.append(min(0.,minimum-required))
        return np.r_[original,deficits]

    def solve(self):
        result=super().solve()
        if self.central_target is not None:
            result['target_tangent']=float(self.central_target@self.e)
            result['target_rule']='observed_central_surface_triangle'
        result['housing_construction']='spatial_observed_height_field; original full-mesh audit is the acceptance test'
        return result
