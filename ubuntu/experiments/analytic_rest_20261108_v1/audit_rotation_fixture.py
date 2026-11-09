"""Independent SO(3) derivative fixtures, no study data."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from analytic_model import right_jacobian,skew
errors=[];inverse_errors=[];h=1e-6
for v in (np.zeros(3),np.array([.12,-.22,.18]),np.array([.6,-.4,.2])):
    R=Rotation.from_rotvec(v).as_matrix();J=right_jacobian(v);Ji=right_jacobian(v,inverse=True)
    for k in range(3):
        e=np.eye(3)[k]*h
        numeric=(Rotation.from_rotvec(v+e).as_matrix()-Rotation.from_rotvec(v-e).as_matrix())/(2*h)
        errors.append(float(np.max(abs(numeric-R@skew(J[:,k])))))
        a=Rotation.from_matrix(R@Rotation.from_rotvec(-e).as_matrix()).as_rotvec()
        b=Rotation.from_matrix(R@Rotation.from_rotvec(e).as_matrix()).as_rotvec()
        inverse_errors.append(float(np.max(abs((b-a)/(2*h)-Ji[:,k]))))
result=dict(passed=max(errors+inverse_errors)<1e-8,maximum_rotation_error=max(errors),maximum_log_error=max(inverse_errors),experiment_inputs_read=False,physics_steps=0)
with Path(__file__).with_name('synthetic_audit.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result));assert result['passed']
