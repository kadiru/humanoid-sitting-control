"""One persistent physical probe per GPU, no optimizer or holdout access."""
import argparse,sys,time,traceback,json
from pathlib import Path
from positive_slope_common_v5 import command,setarg
from run_transfer_assessment import run,status
DATA=Path('/data/humanoid-sitting-perception/low_positive_transfer_01')
p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);p.add_argument('--variant',required=True);a=p.parse_args()
state=DATA/f'probe_status_{a.variant}.json'
try:
    folder=DATA/a.variant;folder.mkdir(exist_ok=False)
    cmd=command(folder,a.gpu,'parent','zero',seed=62101,start=8.,bank='training')
    cmd[2]='../scripts/run_low_positive_probe.py';setarg(cmd,'--bank',DATA/'banks/probe');setarg(cmd,'--num-envs',12)
    cmd+=['--low-positive-config',str(DATA/(a.variant+'.json'))]
    # Early action is zero; frozen59117 remains on the conditional11/15s handoff.
    status(state,'running',variant=a.variant,gpu=a.gpu)
    run(folder,'parent',cmd)
    rows=json.loads((folder/'parent/results.json').read_text())['episodes']
    status(state,'complete',trials=len(rows),stable=sum(x['success'] for x in rows),clean=sum(x['success'] and not any(x['failures'].values()) and not x['slip_ticks'] and not x['illegal_ticks'] for x in rows))
except Exception as e:status(state,'failed',error=repr(e),traceback=traceback.format_exc());raise
