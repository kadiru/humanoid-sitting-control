import argparse,traceback
from low_positive_common import *
from run_transfer_assessment import run,status
from positive_slope_validation_v4 import physical
p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);p.add_argument('--variant',required=True);a=p.parse_args()
state=DATA/f'probe_status_{a.variant}.json'
try:
    folder=DATA/a.variant;folder.mkdir(exist_ok=False)
    cmd=old_command(folder,a.gpu,'parent','zero',seed=62101,start=8.,bank='training');setbank(cmd,DATA/'banks/probe')
    cmd[2]='../scripts/run_low_positive_rolling_probe.py';cmd+=['--low-positive-config',str(DATA/(a.variant+'.json'))]
    status(state,'running',gpu=a.gpu,variant=a.variant);run(folder,'parent',cmd)
    status(state,'complete',physical=physical(folder/'parent',require_clean=False))
except Exception as e:status(state,'failed',error=repr(e),traceback=traceback.format_exc());raise
