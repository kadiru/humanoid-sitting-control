"""Install a separately selected sensor-only positive-slope reference.

Selection and configuration must be saved before simulation. No test labels
are inspected here; the runtime uses only kinematics, IMU, measured plane and
the explicit reference motion. Physical labels remain diagnostic outputs.
"""
import json
from pathlib import Path
import numpy as np
from positive_slope_planner import plan

def install(harness,path):
    selected=json.loads(Path(path).read_text());config=selected['planner'];bridge=harness.traction.bridge
    harness.traction.progressive.rolling_plan=lambda *a,**kw:plan(*a,**kw,inertial=bridge.inertial,**config)
    duration=selected.get('duration_s',5.)
    if duration!=5.:
        import positive_timed_reference as timing
        timing.DURATION=duration
        bridge.BridgeEnv._positive_original_split_reference=lambda self:bridge.progressive.original.SplitEnv.reference(self)
        bridge.BridgeEnv.reference=timing.timed_reference
    return selected
