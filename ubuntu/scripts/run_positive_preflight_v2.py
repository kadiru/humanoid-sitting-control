"""Independent clean-domain admission; no automatic main launch here."""
import argparse,time,traceback
import numpy as np
from positive_slope_common_v2 import *
from positive_slope_validation_v2 import physical,matched,replay,actors
from verify_seat_load_study import equal,load,evaluation
from run_transfer_assessment import run,status

def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,choices=[0,1],required=True);a=p.parse_args();seed=SEEDS[a.gpu]
    folder=DATA/'preflight_v2';folder.mkdir(exist_ok=True);state=DATA/f'preflight_v2_status_gpu{a.gpu}.json'
    try:
        if a.gpu==0:
            status(state,'matched_repaired_reference');run(folder,'reference',repaired_baseline(folder,0,'reference'))
            physical(folder/'reference');evaluation(folder/'reference',18)
            status(state,'calibration_and_zero_action_reset');run(folder,'calibrate',command(folder,0,'calibrate','calibrate',updates=2))
            physical(folder/'calibrate',2)
        else:
            status(state,'matched_frozen_parent');run(folder,'parent',repaired_baseline(folder,1,'parent','59117'))
            physical(folder/'parent');evaluation(folder/'parent',18)
            status(state,'independent_parent_indexed_reset')
            cmd=repaired_baseline(folder,1,'parent_reset','59117');cmd[2]='../scripts/run_positive_parent_v2.py'
            setarg(cmd,'--episode-mode','explore');setarg(cmd,'--episode-updates',2)
            run(folder,'parent_reset',cmd);physical(folder/'parent_reset',2)
        status(state,'waiting_for_matched_calibration')
        while not all((folder/x/y).exists() for x,y in [('calibrate','early_contract.json'),('parent_reset','episode_contract.json')]):
            other=DATA/f'preflight_v2_status_gpu{1-a.gpu}.json'
            assert not other.exists() or read(other)['phase']!='failed',read(other)
            time.sleep(10)
        zero=np.load(folder/'calibrate/trace.npz');parent=np.load(folder/'parent/trace.npz');reset=np.load(folder/'parent_reset/trace.npz')
        for k in parent.files:
            assert np.array_equal(zero[k][:200],parent[k]),('first zero episode',k)
            assert np.array_equal(zero[k],reset[k]),('indexed zero replay',k)
        assert equal(load(checkpoint(59117)),load(folder/'calibrate/loaded_parent.pt'))
        write(folder/f'zero_validation_{seed}.json',dict(passed=True,all_trace_arrays_exact=True,global_and_indexed_reset_match=True,parent_frozen_exact=True))
        for label,mode,updates in [(f'explore_{seed}','explore',2),(f'smoke_{seed}','train',2)]:
            status(state,label);run(folder,label,command(folder,a.gpu,label,mode,seed=seed,updates=updates))
            physical(folder/label,2)
        cp=folder/f'smoke_{seed}/final.pt';label=f'smoke_eval_{seed}'
        status(state,'matched_smoke_evaluation');run(folder,label,command(folder,a.gpu,label,'evaluate',seed=seed,cp=cp))
        physical(folder/label);evaluation(folder/label,18);matched(folder/'reference',folder/label,9.)
        validation=dict(physical=physical(folder/label),optimizer=replay(folder/f'smoke_{seed}',2),actor=actors(folder/label,cp))
        write(folder/f'validation_{seed}.json',validation);check(read(DATA/'prior_hashes.json'))
        status(state,'complete',seed=seed,validation=validation)
    except Exception as e:status(state,'failed',error=repr(e),traceback=traceback.format_exc());raise
if __name__=='__main__':main()
