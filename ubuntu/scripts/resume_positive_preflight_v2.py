"""Preserved cross-GPU replay recovery; compare identical harnesses on same GPU."""
import argparse,time,traceback
import numpy as np
from positive_slope_common_v2 import *
from positive_slope_validation_v2 import physical,matched,replay,actors
from verify_seat_load_study import equal,load,evaluation
from run_transfer_assessment import run,status

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,choices=[0,1],required=True);a=ap.parse_args();seed=SEEDS[a.gpu]
    pre=DATA/'preflight_v2';recovery=DATA/'reset_recovery_01';state=DATA/f'preflight_v2_status_gpu{a.gpu}.json'
    try:
        status(state,'same_gpu_zero_reset_replay',gpu=a.gpu)
        if a.gpu==0:
            label='parent_gpu0';cmd=repaired_baseline(recovery,0,label,'59117');setarg(cmd,'--episode-mode','explore');setarg(cmd,'--episode-updates',2)
            run(recovery,label,cmd);physical(recovery/label,2)
            left,right=pre/'calibrate',recovery/label
        else:
            label='calibrate_gpu1';run(recovery,label,command(recovery,1,label,'calibrate',seed=61111,updates=2));physical(recovery/label,2)
            left,right=recovery/label,pre/'parent_reset'
        x=np.load(left/'trace.npz');y=np.load(right/'trace.npz')
        for k in x.files:assert np.array_equal(x[k],y[k]),('same GPU zero replay',a.gpu,k,float(np.max(abs(x[k]-y[k]))))
        x=np.load(left/'episode_decisions.npz');y=np.load(right/'episode_decisions.npz')
        assert np.array_equal(x['parent_observation'],y['observation']) and np.array_equal(x['parent_action'],y['action'])
        base=np.load(pre/'parent/trace.npz');z=np.load(pre/'calibrate/trace.npz')
        for k in base.files:assert np.array_equal(z[k][:200],base[k]),('original first episode',k)
        write(pre/f'zero_validation_{seed}.json',dict(passed=True,all_trace_arrays_exact=True,comparison_hardware='same physical GPU',gpu=a.gpu,global_and_indexed_reset_match=True,parent_frozen_exact=True,left=str(left),right=str(right),cross_gpu_differences_preserved=str(recovery/'manifest.json')))
        for label,mode,updates in [(f'explore_{seed}','explore',2),(f'smoke_{seed}','train',2)]:
            status(state,label);run(pre,label,command(pre,a.gpu,label,mode,seed=seed,updates=updates));physical(pre/label,2)
        cp=pre/f'smoke_{seed}/final.pt';label=f'smoke_eval_{seed}'
        status(state,'matched_smoke_evaluation');run(pre,label,command(pre,a.gpu,label,'evaluate',seed=seed,cp=cp));physical(pre/label);evaluation(pre/label,18);matched(pre/'reference',pre/label,9.)
        validation=dict(physical=physical(pre/label),optimizer=replay(pre/f'smoke_{seed}',2),actor=actors(pre/label,cp));write(pre/f'validation_{seed}.json',validation);check(read(DATA/'prior_hashes.json'));status(state,'complete',seed=seed,validation=validation)
    except Exception as e:status(state,'failed',error=repr(e),traceback=traceback.format_exc());raise
if __name__=='__main__':main()
