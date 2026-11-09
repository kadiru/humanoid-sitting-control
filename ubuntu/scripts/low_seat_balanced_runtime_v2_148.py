"""Pre-training force-objective revision of132; no saved sensors in runtime.

Only the planned force allocation changes. Current-input pose planner126,
measured guard130, follower128, filter/rate, measured-view selection, frozen
parent and all physical criteria remain unchanged. Static loads are assumed;
dynamic balance and stable pauses still require physical validation.
"""
import low_seat_fixed_runtime_132 as previous
from low_seat_fixed_foot_plan_126 import build_plan as original_build_plan
from low_seat_balanced_force_v2_145 import reallocate

def build_plan(model,meta,inertia,cal,q,gravity,geometry,reference,command,endpoint):
    plan=original_build_plan(model,meta,inertia,cal,q,gravity,geometry,reference,command,endpoint)
    inputs=dict(q=q,command=command)
    return reallocate(model,inertia,plan,inputs)

def install(harness,landing_config,config):
    previous.build_plan=build_plan
    previous.install(harness,landing_config,config)
