"""Independent synthetic witness for support-feature torque discontinuity."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
vertices=np.array([[-.05,0.,-.1],[-.05,0.,.1],[.05,0.,-.1],[.05,0.,.1]])
normal=np.array([1.,0.,0.]);force=normal*100.
fixed=vertices[1].copy();dynamic_moments=[];fixed_moments=[];legal=[]
for theta in [-1e-7,1e-7]:
    R=Rotation.from_euler('y',theta).as_matrix();world=vertices@R.T
    # Original dynamic support convention: centroid of tied exposed vertices.
    projection=world@normal;chosen=world[projection<=projection.min()+1e-10].mean(0)
    dynamic_moments.append(float(np.cross(chosen,force)[1]))
    point=R@fixed;fixed_moments.append(float(np.cross(point,force)[1]))
    translated=world-(point@normal)*normal
    legal.append(bool(np.min(translated@normal)>=-1e-12))
jump=abs(dynamic_moments[1]-dynamic_moments[0]);smooth=abs(fixed_moments[1]-fixed_moments[0])
tests=dict(dynamic_torque_jump=jump>19.9,fixed_material_continuity=smooth<2e-6,
    wrong_material_orientation_rejected=legal==[True,False])
result=dict(passed=all(tests.values()),tests=tests,dynamic_torque_jump_Nm=jump,
    fixed_material_torque_change_Nm=smooth,experiment_inputs_read=False,physics_steps=0,
    limitation='Demonstrates formulation discontinuity, not the exclusive cause of a saved solver failure')
with Path(__file__).with_name('synthetic_audit.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result));assert result['passed']
