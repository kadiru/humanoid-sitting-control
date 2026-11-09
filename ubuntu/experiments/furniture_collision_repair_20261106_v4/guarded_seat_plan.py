"""Local camera support plus explicit hip-housing clearance during landing."""
import numpy as np
from humanoid_sitting.shared_frame import apply
from seat_plan import SeatPlan as BasePlan
from local_patch import refine


def housing_reserve(progress,plane_error):
    # Original6mm collision expansion, the entire15mm later stroke, and2mm
    # construction reserve. Plane error increases the requirement, never loosens it.
    return .006+.015*progress+plane_error+.002


class SeatPlan(BasePlan):
    def __init__(self,model,inertia,shapes,q,gravity,geometry,points,previous):
        value,local,self.patch_report=refine(model,q,gravity,geometry,points)
        super().__init__(model,inertia,shapes,q,gravity,value,local,previous)
        self.refined_geometry=value
        self.housing=['left_hip_pitch_link','right_hip_pitch_link']

    def residual(self,z,gap,tangent,balance):
        original=super().residual(z,gap,tangent,balance)
        q,root,fk,*_=self.state(z)
        minimum=np.array([(apply(root@fk[body],self.mesh[body])@self.n+self.d).min() for body in self.housing])
        required=housing_reserve(max(self.progress,self.arm_reach),self.patch_report['maximum_plane_residual_m'])
        return np.r_[original,np.minimum(minimum-required,0.)]
