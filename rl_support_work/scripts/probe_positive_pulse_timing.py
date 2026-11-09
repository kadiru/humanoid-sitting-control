"""Development-only earlier-pulse probes; no optimizer or heldout simulation."""
import argparse,traceback
from positive_slope_common_v4 import *
from positive_slope_validation_v4 import physical,matched
from run_positive_bounded_preflight import tick_checks
from run_transfer_assessment import run,status


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,choices=[0,1],required=True);a=ap.parse_args()
    root=DATA/'pulse_timing_probe_05';root.mkdir(exist_ok=True);seed=SEEDS[a.gpu]
    state=root/f'status_{seed}.json';label=f'explore8_{seed}'
    try:
        assert not (DATA/'ppo_01').exists()
        status(state,'running',seed=seed,start=8.,release=9.,optimizer_updates=0)
        run(root,label,command(root,a.gpu,label,'explore',seed=seed,start=8.,updates=2))
        outcome=physical(root/label,2,require_clean=False);ticks=tick_checks(root/label,2)
        startup=matched(DATA/'preflight_v2/reference',root/label,8.)
        write(root/f'validation_{seed}.json',dict(physical=outcome,ticks=ticks,startup=startup,passed=outcome['clean']==36,optimizer_updates=0))
        status(state,'complete',seed=seed,passed=outcome['clean']==36,outcome=outcome)
    except Exception as e:status(state,'failed',error=repr(e),traceback=traceback.format_exc());raise


if __name__=='__main__':main()
