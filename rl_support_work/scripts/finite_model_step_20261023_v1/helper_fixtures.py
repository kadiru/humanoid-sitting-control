"""Synthetic current-origin identity tests only; no saved scientific inputs."""
import json
import numpy as np
from common import bind_current,identical

def must_reject(fn):
    try:fn()
    except AssertionError:return
    raise AssertionError('Expected identity rejection')

def main():
    reference={'features':np.array([[1.,2.]]),'actualY':np.array([[3.]]),'J':np.array([[[4.]]]),'active':np.array([0])}
    state={'features':np.array([[5.,0.]]),'actualY':np.array([[6.]]),'J':np.array([[[7.]]])}
    af,ay,aj=(state[k].copy() for k in ('features','actualY','J'))
    z=bind_current(reference,state,af,ay,aj)
    assert z['features'][0,0]==5 and z['actualY'][0,0]==6 and z['J'][0,0,0]==7
    assert z['active'] is reference['active'] and reference['features'][0,0]==1
    checks=['current_origin_replaces_initial_f_Y_J','structural_metadata_preserved_and_reference_unmodified']
    must_reject(lambda:bind_current(reference,state,reference['features'],ay,aj));checks.append('old_initial_features_rejected')
    must_reject(lambda:bind_current(reference,state,af,ay+1,aj));checks.append('mismatched_accepted_Y_rejected')
    signed=af.copy();signed[0,1]=-0.;must_reject(lambda:bind_current(reference,state,signed,ay,aj));checks.append('signed_zero_identity_mismatch_rejected')
    altered={k:v.copy() for k,v in state.items()};altered['J'][0,0,0]=np.inf
    must_reject(lambda:bind_current(reference,altered,af,ay,altered['J']));checks.append('nonfinite_matching_J_rejected')
    must_reject(lambda:identical(af.astype(np.float32),af));checks.append('dtype_identity_mismatch_rejected')
    print(json.dumps({'passed':True,'checks':checks,'scientific_data_loaded':False,'model_calls':0}))
if __name__=='__main__':main()
