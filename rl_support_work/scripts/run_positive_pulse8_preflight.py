"""Admit 8-second intervention only after both physical timing probes pass."""
import argparse,time,traceback
import numpy as np
from positive_slope_common_v5 import *
from positive_slope_validation_v4 import physical,matched,replay,actors
from run_positive_bounded_preflight import tick_checks
from verify_seat_load_study import evaluation
from run_transfer_assessment import run,status


def zero_check8(folder,gpu):
    reference=DATA/('preflight_v2/calibrate' if gpu==0 else 'reset_recovery_01/calibrate_gpu1')
    for filename in ['trace.npz','traction_startup.npz']:
        with np.load(folder/filename) as a,np.load(reference/filename) as b:
            assert set(a.files)==set(b.files)
            for k in a.files:assert np.array_equal(a[k],b[k]),(filename,k)
    with np.load(folder/'episode_decisions.npz') as a,np.load(reference/'episode_decisions.npz') as b,np.load(reference/'stage_observations.npz') as stages:
        for k in ['parent_observation','parent_action','handover_step']:assert np.array_equal(a[k],b[k]),k
        assert np.array_equal(a['early_observation'],stages['8.0'])
        assert not a['early_action'].any()
    assert read(folder/'early_calibration.json')==read(reference/'early_calibration.json')
    return dict(passed=True,full_trace_exact=True,actual414_parent_inputs_actions_exact=True,early_inputs_match_saved_stage8=True,
                same_gpu=gpu,global_and_indexed_reset_exact=True,new_run=str(folder),reference=str(reference))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,choices=[0,1],required=True);a=ap.parse_args()
    seed=SEEDS[a.gpu];pre=DATA/'preflight_pulse8_05';pre.mkdir(exist_ok=True);state=DATA/f'preflight_pulse8_05_status_gpu{a.gpu}.json'
    try:
        assert not (DATA/'ppo_01').exists();status(state,'waiting_for_both_timing_probes')
        probe=DATA/'pulse_timing_probe_05'
        while True:
            states=[read(probe/f'status_{s}.json') if (probe/f'status_{s}.json').exists() else {} for s in SEEDS]
            assert not any(x.get('phase')=='failed' or (x.get('phase')=='complete' and not x.get('passed')) for x in states),states
            if all(x.get('phase')=='complete' and x.get('passed') for x in states):break
            time.sleep(15)
        checks={}
        for s in SEEDS:
            folder=probe/f'explore8_{s}';checks[str(s)]=dict(physical=physical(folder,2),ticks=tick_checks(folder,2),startup=matched(DATA/'preflight_v2/reference',folder,8.))
        label=f'zero_pulse_{seed}';status(state,label,seed=seed,start=8.,release=9.)
        run(pre,label,command(pre,a.gpu,label,'calibrate',seed=61111,start=8.,updates=2))
        physical(pre/label,2);tick_checks(pre/label,2);proof=zero_check8(pre/label,a.gpu);write(pre/f'zero_validation_{seed}.json',proof)
        label=f'smoke_{seed}';status(state,label,seed=seed,start=8.)
        run(pre,label,command(pre,a.gpu,label,'train',seed=seed,start=8.,updates=2))
        checks['smoke']=dict(physical=physical(pre/label,2),ticks=tick_checks(pre/label,2))
        cp=pre/f'smoke_{seed}/final.pt';label=f'smoke_eval_{seed}';status(state,label,seed=seed)
        run(pre,label,command(pre,a.gpu,label,'evaluate',seed=seed,start=8.,cp=cp))
        physical(pre/label);evaluation(pre/label,18);matched(DATA/'preflight_v2/reference',pre/label,8.)
        validation=dict(physical=physical(pre/label),ticks=tick_checks(pre/label,1),optimizer=replay(pre/f'smoke_{seed}',2),actor=actors(pre/label,cp),preceding=checks)
        check(read(DATA/'prior_hashes.json'));check(read(DATA/'pulse9_archive_05/revision.json')['preserved_outputs'])
        write(pre/f'validation_{seed}.json',validation);status(state,'complete',seed=seed,validation=validation)
    except Exception as e:status(state,'failed',error=repr(e),traceback=traceback.format_exc());raise


if __name__=='__main__':main()
