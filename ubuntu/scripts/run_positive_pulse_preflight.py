"""New physical preflight for a compact early pulse; previous failures stay intact."""
import argparse,traceback
import numpy as np
from positive_slope_common_v4 import *
from positive_slope_validation_v4 import physical,matched,replay,actors
from run_positive_bounded_preflight import tick_checks
from verify_seat_load_study import evaluation
from run_transfer_assessment import run,status


def zero_check(folder,gpu):
    reference=DATA/('preflight_v2/calibrate' if gpu==0 else 'reset_recovery_01/calibrate_gpu1')
    for filename in ['trace.npz','episode_decisions.npz','traction_startup.npz']:
        with np.load(folder/filename) as a,np.load(reference/filename) as b:
            assert set(a.files)==set(b.files)
            for k in a.files:assert np.array_equal(a[k],b[k]),(filename,k)
    assert read(folder/'early_calibration.json')==read(reference/'early_calibration.json')
    return dict(passed=True,all_trace_arrays_exact=True,actual414_parent_inputs_actions_exact=True,
                global_and_indexed_reset_match=True,same_gpu=gpu,reference=str(reference),new_run=str(folder))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,choices=[0,1],required=True);a=ap.parse_args()
    seed=SEEDS[a.gpu];pre=DATA/'preflight_pulse_04';pre.mkdir(exist_ok=True)
    state=DATA/f'preflight_pulse_04_status_gpu{a.gpu}.json'
    try:
        assert not (DATA/'ppo_01').exists()
        archive=read(DATA/'early_span_001_persistent_archive_04/revision.json');check(archive['preserved_outputs'])
        for name in ['reference','parent']:physical(DATA/'preflight_v2'/name);evaluation(DATA/'preflight_v2'/name,18)
        label=f'zero_pulse_{seed}';status(state,label,seed=seed)
        run(pre,label,command(pre,a.gpu,label,'calibrate',seed=61111,updates=2))
        physical(pre/label,2);tick_checks(pre/label,2)
        proof=zero_check(pre/label,a.gpu);write(pre/f'zero_validation_{seed}.json',proof)
        checks={}
        for label,mode in [(f'explore_{seed}','explore'),(f'smoke_{seed}','train')]:
            status(state,label,seed=seed,span=SPAN,pulse_duration=RAMP)
            run(pre,label,command(pre,a.gpu,label,mode,seed=seed,updates=2))
            checks[label]=dict(physical=physical(pre/label,2),ticks=tick_checks(pre/label,2))
        cp=pre/f'smoke_{seed}/final.pt';label=f'smoke_eval_{seed}'
        status(state,'matched_smoke_evaluation',seed=seed)
        run(pre,label,command(pre,a.gpu,label,'evaluate',seed=seed,cp=cp))
        physical(pre/label);evaluation(pre/label,18);matched(DATA/'preflight_v2/reference',pre/label,9.)
        validation=dict(physical=physical(pre/label),ticks=tick_checks(pre/label,1),optimizer=replay(pre/f'smoke_{seed}',2),actor=actors(pre/label,cp),preceding=checks)
        check(read(DATA/'prior_hashes.json'));check(archive['preserved_outputs'])
        write(pre/f'validation_{seed}.json',validation);status(state,'complete',seed=seed,validation=validation)
    except Exception as e:status(state,'failed',error=repr(e),traceback=traceback.format_exc());raise


if __name__=='__main__':main()
