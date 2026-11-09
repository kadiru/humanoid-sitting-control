"""Compare original-parent harness with new zero-feedback execution exactly."""
import argparse,time,traceback
from pathlib import Path
import numpy as np
from low_positive_common import *
from run_transfer_assessment import run,status
from positive_slope_validation_v4 import physical
from verify_seat_load_study import starts

p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,default=0);a=p.parse_args()
state=DATA/'feedback_zero_status.json';folder=DATA/'feedback_zero_01'
try:
    while read(DATA/'probe_status_timing_mild.json')['phase']=='running':time.sleep(10)
    folder.mkdir(exist_ok=False);status(state,'baseline')
    bank=DATA/'banks/probe'
    cmd=old_command(folder,a.gpu,'old_parent','zero',seed=62111,start=8.,bank='training');setbank(cmd,bank)
    run(folder,'old_parent',cmd)
    status(state,'zero_global_indexed')
    cmd=feedback_command(folder,a.gpu,'new_zero','zero',bank,OLD/'selected_reference_v2.json',kind='original',updates=2)
    run(folder,'new_zero',cmd)
    first=folder/'old_parent';second=folder/'new_zero'
    assert starts(first,second)['exact']
    A=np.load(first/'traction_startup.npz');B=np.load(second/'traction_startup.npz')
    for k in A.files:assert np.array_equal(A[k],B[k]),('startup414',k)
    A=np.load(first/'trace.npz');B=np.load(second/'trace.npz')
    for k in A.files:
        if k=='t':continue
        assert np.array_equal(A[k],B[k][:200],equal_nan=True),('global',k)
        assert np.array_equal(A[k],B[k][-200:],equal_nan=True),('indexed',k)
    A=np.load(first/'episode_decisions.npz');B=np.load(second/'sensor_inputs.npz')
    for i in range(2):
        assert np.array_equal(A['parent_observation'][0],B['parent_observation'][i]),('parent_inputs',i)
        assert np.array_equal(A['parent_action'][0],B['parent_action'][i]),('parent_actions',i)
    outcome=physical(second,episodes=2,require_clean=False)
    status(state,'passed',exact_global_and_indexed_trace=True,exact_actual414_parent=True,physical=outcome,scope='Harness equivalence only; known physical failures retained, not repair admission')
except Exception as e:status(state,'failed',error=repr(e),traceback=traceback.format_exc());raise
