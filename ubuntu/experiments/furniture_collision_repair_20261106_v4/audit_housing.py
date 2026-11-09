"""Independent recomputation of housing separation at every approach/landing node."""
import numpy as np
from humanoid_sitting.shared_frame import apply


def audit(plan,construction,path):
    approach=[];landing=[]
    for i,node in enumerate(construction['nodes']):
        rt=np.array(node['root']);fk=plan.model.fk(node['q'])
        required=.008+.015*i/(len(construction['nodes'])-1)+plan.patch_report['maximum_plane_residual_m']
        for body in ['left_hip_pitch_link','right_hip_pitch_link']:
            gap=float((apply(rt@fk[body],plan.mesh[body])@plan.n+plan.d).min())
            approach.append(gap-required)
    for node in path['landing']:
        rt=np.array(node['root']);fk=plan.model.fk(node['q'])
        for body in ['left_hip_pitch_link','right_hip_pitch_link']:
            gap=float((apply(rt@fk[body],plan.mesh[body])@plan.n+plan.d).min())
            landing.append(gap-.006-plan.patch_report['maximum_plane_residual_m'])
    return dict(passed=bool(min(approach)>=-1e-4 and min(landing)>=0),
                minimum_approach_extra_margin_m=min(approach),minimum_landing_extra_margin_m=min(landing),
                original6mm_mesh_clearance_still_required=True,physical_success=False)
