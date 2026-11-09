"""A single preserved timing test, no PPO or heldout evaluation."""
import argparse,traceback
from pathlib import Path
from positive_slope_common_v5 import command,setarg,read
from run_transfer_assessment import run,status
from positive_slope_validation_v4 import physical
DATA=Path('/data/humanoid-sitting-perception/low_positive_transfer_01')
p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);p.add_argument('--variant',required=True);a=p.parse_args()
state=DATA/f'probe_status_{a.variant}.json'
try:
    folder=DATA/a.variant;folder.mkdir(exist_ok=False)
    cmd=command(folder,a.gpu,'parent','zero',seed=62101,start=8.,bank='training')
    cmd[2]='../scripts/run_low_positive_timing_probe.py';setarg(cmd,'--bank',DATA/'banks/probe');setarg(cmd,'--num-envs',12)
    cmd+=['--low-positive-config',str(DATA/(a.variant+'.json'))]
    status(state,'running',variant=a.variant,gpu=a.gpu);run(folder,'parent',cmd)
    status(state,'complete',physical=physical(folder/'parent',require_clean=False))
except Exception as e:status(state,'failed',error=repr(e),traceback=traceback.format_exc());raise
