"""Resume preserved preflight after a numerical audit-target correction."""
import argparse,time,traceback
from low_positive_common import *
from positive_slope_common_v5 import write
from run_transfer_assessment import run,status
from validate_low_positive_feedback_v3 import validate
from positive_slope_validation_v4 import matched
p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);p.add_argument('--seed',type=int,required=True);a=p.parse_args()
state=DATA/f'preflight_recovery_status_{a.seed}.json';folder=DATA/'preflight_01'/f'seed_{a.seed}'
try:
    status(state,'waiting_original_worker',seed=a.seed,gpu=a.gpu)
    while read(DATA/f'preflight_status_{a.seed}.json')['phase'] not in ['failed','passed']:time.sleep(10)
    bank=DATA/'banks/training';repair=DATA/'learning_reference.json';cal=folder/'calibration/feedback_calibration.json'
    def command(label,mode,updates=1,cp=None):
        cmd=feedback_command(folder,a.gpu,label,mode,bank,repair,kind='original',seed=a.seed,start=10.,updates=updates,cal=cal,cp=cp);cmd[2]='../scripts/run_low_positive_feedback_v2.py';return cmd
    for label in ['calibration','exploration']:
        result=validate(folder/label);write(folder/(label+'_validation_v3.json'),result)
    matched(folder/'calibration',folder/'exploration',start=10.)
    status(state,'smoke',seed=a.seed,gpu=a.gpu)
    if not (folder/'smoke').exists():run(folder,'smoke',command('smoke','train',updates=2))
    assert (folder/'smoke/final.pt').exists(),'Partial smoke output requires inspection, never overwrite/restart silently'
    write(folder/'smoke_validation_v3.json',validate(folder/'smoke'));matched(folder/'calibration',folder/'smoke',start=10.)
    status(state,'smoke_evaluation',seed=a.seed,gpu=a.gpu)
    if not (folder/'smoke_evaluation').exists():run(folder,'smoke_evaluation',command('smoke_evaluation','evaluate',cp=folder/'smoke/final.pt'))
    result=validate(folder/'smoke_evaluation');write(folder/'smoke_evaluation_validation_v3.json',result);matched(folder/'calibration',folder/'smoke_evaluation',start=10.)
    b=read(folder/'calibration/results.json')['episodes'];r=read(folder/'smoke_evaluation/results.json')['episodes']
    clean=lambda x:x['success'] and not any(x['failures'].values()) and not x['slip_ticks'] and not x['illegal_ticks']
    status(state,'passed',seed=a.seed,gpu=a.gpu,validation=result,smoke_lost_baseline_clean_cases=[i for i,(x,y) in enumerate(zip(b,r)) if clean(x) and not clean(y)],smoke_added_physical_failure_cases=[i for i,(x,y) in enumerate(zip(b,r)) if not any(x['failures'].values()) and any(y['failures'].values())],audit_repair='Use recorded previous applied offset for actor-memory identity; all numerical tolerances unchanged. Separate independent slew/action reconstruction retained.',qualification='Software/optimizer validation; known physical failures remain training targets and must be reported.')
except Exception as e:status(state,'failed',error=repr(e),traceback=traceback.format_exc());raise
