"""New normal-approach probe, reusing a preserved exact same-GPU baseline."""
import argparse,json,hashlib,traceback
from pathlib import Path
from low_positive_common import feedback_command
from run_transfer_assessment import run,status
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');OLD=D.parent/'low_positive_transfer_01'
p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);p.add_argument('--variant',required=True);a=p.parse_args()
folder=D/a.variant;folder.mkdir(exist_ok=False);state=D/f'status_{a.variant}.json'
try:
    original=D/('damped_original_01' if a.gpu==0 else 'clearance_damped_01')/'baseline'
    assert (original/'feedback_contract.json').exists()
    (folder/'baseline').symlink_to(original,target_is_directory=True)
    (folder/'baseline_reuse.json').write_text(json.dumps(dict(source=str(original),physical_gpu=a.gpu,
        simulated_again=False,hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in original.iterdir() if p.is_file()}),indent=2))
    cmd=feedback_command(folder,a.gpu,'candidate','zero',D/'banks/probe',OLD/'learning_reference.json',kind='original',seed=63101,start=10.)
    cmd[2]='../scripts/run_low_seat_timing_probe.py';cmd+=['--landing-config',str(D/(a.variant+'.json'))]
    status(state,'running',gpu=a.gpu,baseline_reused=str(original));run(folder,'candidate',cmd)
    rows=json.loads((folder/'candidate/results.json').read_text())['episodes']
    status(state,'complete',trials=len(rows),stable=sum(x['success'] for x in rows),
        clean=sum(x['success'] and not any(x['failures'].values()) and not x['slip_ticks'] and not x['illegal_ticks'] for x in rows),
        slip=sum(x['slip_ticks']>0 for x in rows),physical=sum(any(x['failures'].values()) for x in rows),training_started=False)
except Exception as e:
    status(state,'failed',error=repr(e),traceback=traceback.format_exc());raise
