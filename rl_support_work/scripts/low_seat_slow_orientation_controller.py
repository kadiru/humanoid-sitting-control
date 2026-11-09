"""Modest, slower orientation before normal landing on low steep seats.

The preceding large rotation exceeded the inherited command slew limit. This
candidate keeps that limit and narrows the orientation change. It is a physical
research probe, not a validated clearance or stability guarantee.
"""
import numpy as np
import low_seat_normal_timing_controller as normal
import low_seat_load_controller as load

def plan(*args,inertial,config):
    # Use the measured-plane planner's audit for selection, never truth labels.
    c,q,a=normal.plan(*args,inertial=inertial,config=config)
    n=np.asarray(a['normal']);height=(n[0]*.4-a['offset'])/n[2]
    slope=np.rad2deg(np.arctan2(n[0],n[2]))
    active=bool(a.get('landing_repair_applied') and height<.39 and slope>14.)
    if active:
        changed=dict(config,orientation_fraction=.6,landing_fraction=1.,pitch=.55,steep_pitch=.55)
        c,q,a=normal.plan(*args,inertial=inertial,config=changed)
    a['slow_orientation_probe']=dict(applied=active,target_pitch_rad=.55 if active else a['pitch'],
        rotation_seconds=3. if active else 0.,landing_seconds=2. if active else 3.,
        scope='Measured-height<39cm and positive slope>14deg; original physical and command slew limits unchanged')
    return c,q,a

def install(harness,config):
    load.install(harness,config)
    bridge=harness.parent_harness.traction.bridge
    bridge.progressive.rolling_plan=lambda *a,**kw:plan(*a,**kw,inertial=bridge.inertial,config=config)
