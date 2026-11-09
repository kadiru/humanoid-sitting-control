"""Sequential baseline and isolated physical probe; no automatic training."""
import argparse,json,traceback
from pathlib import Path
from low_positive_common import feedback_command
from run_transfer_assessment import run,status
ROOT=Path('/data/humanoid-sitting-perception/low_seat_contact_01')
OLD=ROOT.parent/'low_positive_transfer_01'
p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);p.add_argument('--variant',required=True)
a=p.parse_args();folder=ROOT/a.variant;folder.mkdir(exist_ok=False);state=ROOT/f'status_{a.variant}.json'
try:
    for label in ['baseline','candidate']:
        cmd=feedback_command(folder,a.gpu,label,'zero',ROOT/'banks/probe',OLD/'learning_reference.json',kind='original',seed=63101,start=10.)
        cmd[2]='../scripts/run_low_positive_feedback_v2.py'
        if label=='candidate':
            cmd[2]='../scripts/run_low_seat_contact_probe.py'
            cmd+=['--landing-config',str(ROOT/(a.variant+'.json'))]
        status(state,'running',stage=label,gpu=a.gpu)
        run(folder,label,cmd)
    rows=json.loads((folder/'candidate/results.json').read_text())['episodes']
    status(state,'complete',trials=len(rows),stable=sum(x['success'] for x in rows),
        clean=sum(x['success'] and not any(x['failures'].values()) and not x['slip_ticks'] and not x['illegal_ticks'] for x in rows),
        slip=sum(x['slip_ticks']>0 for x in rows),physical=sum(any(x['failures'].values()) for x in rows),training_started=False)
except Exception as e:
    status(state,'failed',error=repr(e),traceback=traceback.format_exc());raise
