"""Match the SAME fresh-start/indexed-reset sequence in both harnesses.

The earlier audit incorrectly compared the original first episode against the
new second episode. Its sources, status and outputs are retained unchanged.
"""
import traceback
import numpy as np
from low_positive_common import *
from run_transfer_assessment import run,status
from verify_seat_load_study import starts
state=DATA/'feedback_zero_v2_status.json';folder=DATA/'feedback_zero_01'
try:
    status(state,'matched_original_two_episodes')
    cmd=old_command(folder,0,'old_parent_two_episodes','calibrate',seed=62111,start=8.,updates=2,bank='training')
    setbank(cmd,DATA/'banks/probe');run(folder,'old_parent_two_episodes',cmd)
    first=folder/'old_parent_two_episodes';second=folder/'new_zero'
    assert starts(first,second)['exact'];A=np.load(first/'traction_startup.npz');B=np.load(second/'traction_startup.npz')
    for k in A.files:assert np.array_equal(A[k],B[k]),('startup414',k)
    A=np.load(first/'trace.npz');B=np.load(second/'trace.npz')
    for k in A.files:assert np.array_equal(A[k],B[k],equal_nan=True),('full_global_warmup_indexed',k)
    A=np.load(first/'every_tick.npz');B=np.load(second/'every_tick.npz')
    for k in A.files:assert np.array_equal(A[k],B[k],equal_nan=True),('every_tick',k)
    A=np.load(first/'episode_decisions.npz');B=np.load(second/'sensor_inputs.npz')
    for k,key in [('parent_observation','parent_observation'),('parent_action','parent_action'),('handover_step','handover')]:assert np.array_equal(A[k],B[key]),k
    status(state,'passed',same_GPU=True,exact_global_and_indexed_and_warmup_traces=True,exact_actual414_parent_inputs_and_actions=True,scope='Harness regression; known physical failures remain. Fresh-start and indexed resets have different settling histories in both original and new harness.')
except Exception as e:status(state,'failed',error=repr(e),traceback=traceback.format_exc());raise
