"""Independent setup/endpoint schedule recomputation, no solver invocation."""
import numpy as np
from humanoid_sitting.shared_frame import apply


def audit(plan, construction):
    endpoint=construction['nodes'][-1]
    q=np.asarray(endpoint['q']);rt=np.asarray(endpoint['root']);fk=plan.model.fk(q)
    com=np.average([apply(rt@fk[b],c) for b,c in
        zip(plan.inertia['body_names'],plan.inertia['local_com'])],axis=0,weights=plan.inertia['mass'])
    margin=min(com[0]-plan.foot_bounds[0],plan.foot_bounds[1]-com[0])
    start=np.asarray(construction['nodes'][0]['command']);end=np.asarray(endpoint['command'])
    length=float(np.max(abs(end-start)));rate=.003*.99999/.02
    total=length/rate+1.;count=int(np.ceil(total/.02))
    limits={j['name']:np.deg2rad(j['limits']) for j in plan.model.joints if j['name'] in plan.model.names}
    previous=start.copy();peak=0.;minimum=float('inf');mismatch=0.
    for i in range(count):
        t=min((i+1)*.02,total)
        if t<1.:distance=.5*rate*t*t
        elif t>total-1.:distance=length-.5*rate*(total-t)**2
        else:distance=rate*(t-.5)
        target=start+(end-start)*distance/length
        desired=previous+(target-previous)/.1
        applied=previous+np.clip(.1*(desired-previous),-.003,.003)
        mismatch=max(mismatch,float(abs(applied-target).max()))
        peak=max(peak,float(abs(applied-previous).max()))
        minimum=min(minimum,min(min(desired[j]-limits[plan.model.names[j]][0],limits[plan.model.names[j]][1]-desired[j]) for j in range(23)))
        previous=applied
    tests=dict(endpoint_balance_margin=margin>=.04-2e-6,
        actuator_schedule_time=count*.02<=10.4,actuator_schedule_exact=mismatch<1e-10,
        actuator_schedule_rate=peak<=.003+1e-12,actuator_schedule_margin=minimum>=.03,
        endpoint_fingers_fixed=bool(np.array_equal(start[23:],end[23:])),
        path_fingers_fixed=all(np.array_equal(np.asarray(n['command'])[23:],start[23:]) for n in construction['nodes']))
    return dict(passed=all(tests.values()),tests=tests,endpoint_balance_margin_m=float(margin),
        direct_schedule_seconds=count*.02,direct_schedule_increment_rad=peak,direct_schedule_margin_rad=minimum,
        full_path_witness=False,current_camera_validated=False,physical_success=False)
