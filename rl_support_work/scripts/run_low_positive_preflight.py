"""One persistent preflight per GPU; never starts main or heldout evaluations."""
import argparse,time,traceback,shutil
import numpy as np
from low_positive_common import *
from run_transfer_assessment import run,status
from validate_low_positive_feedback_v2 import validate
from positive_slope_common_v5 import write
from positive_slope_validation_v4 import matched
p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);p.add_argument('--seed',type=int,required=True);a=p.parse_args()
state=DATA/f'preflight_status_{a.seed}.json';folder=DATA/'preflight_01'/f'seed_{a.seed}'
try:
    status(state,'waiting_matched_zero_regression',gpu=a.gpu,seed=a.seed)
    while True:
        if (DATA/'feedback_zero_v2_status.json').exists():
            z=read(DATA/'feedback_zero_v2_status.json')
            if z['phase']=='failed':raise RuntimeError(z)
            if z['phase']=='passed':break
        time.sleep(10)
    folder.mkdir(parents=True,exist_ok=False);bank=DATA/'banks/training';repair=DATA/'learning_reference.json'
    def command(label,mode,updates=1,cal=None,cp=None):
        cmd=feedback_command(folder,a.gpu,label,mode,bank,repair,kind='original',seed=a.seed,start=10.,updates=updates,cal=cal,cp=cp)
        cmd[2]='../scripts/run_low_positive_feedback_v2.py';return cmd
    status(state,'calibration',gpu=a.gpu,seed=a.seed);run(folder,'calibration',command('calibration','calibrate'))
    cal=folder/'calibration/feedback_calibration.json';write(folder/'calibration_validation.json',validate(folder/'calibration'))
    status(state,'exploration',gpu=a.gpu,seed=a.seed);run(folder,'exploration',command('exploration','explore',cal=cal))
    write(folder/'exploration_validation.json',validate(folder/'exploration'));matched(folder/'calibration',folder/'exploration',start=10.)
    status(state,'smoke',gpu=a.gpu,seed=a.seed);run(folder,'smoke',command('smoke','train',updates=2,cal=cal))
    smoke=validate(folder/'smoke');write(folder/'smoke_validation.json',smoke);matched(folder/'calibration',folder/'smoke',start=10.)
    status(state,'smoke_evaluation',gpu=a.gpu,seed=a.seed);run(folder,'smoke_evaluation',command('smoke_evaluation','evaluate',cal=cal,cp=folder/'smoke/final.pt'))
    outcome=validate(folder/'smoke_evaluation');write(folder/'smoke_evaluation_validation.json',outcome);matched(folder/'calibration',folder/'smoke_evaluation',start=10.)
    b=read(folder/'calibration/results.json')['episodes'];r=read(folder/'smoke_evaluation/results.json')['episodes']
    clean=lambda x:x['success'] and not any(x['failures'].values()) and not x['slip_ticks'] and not x['illegal_ticks']
    lost=[i for i,(x,y) in enumerate(zip(b,r)) if clean(x) and not clean(y)]
    new_physical=[i for i,(x,y) in enumerate(zip(b,r)) if not any(x['failures'].values()) and any(y['failures'].values())]
    status(state,'passed',gpu=a.gpu,seed=a.seed,validation=outcome,smoke_lost_baseline_clean_cases=lost,smoke_added_physical_failure_cases=new_physical,
        qualification='Software/optimizer preflight, not a no-slip success claim. Known failing cases participate in training. Deterministic smoke regressions remain reported; final policy selection requires preserving baseline clean outcomes.')
except Exception as e:status(state,'failed',error=repr(e),traceback=traceback.format_exc());raise
