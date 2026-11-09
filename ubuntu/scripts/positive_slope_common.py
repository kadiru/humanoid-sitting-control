"""Separate positive-slope study; completed studies are read-only."""
import hashlib,json
from pathlib import Path
from posture_credit_common import command as posture_command
from earlier_control_common import command as earlier_command

ROOT=Path('/home/kadir/humanoid-sitting')
DATA=Path('/data/humanoid-sitting-perception/positive_slope_01')
OLD=DATA.parent/'earlier_control_01'
SEEDS=[61111,61117]
STAGES=[9.,8.,6.]
SPAN=.004
RAMP=1.
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2)+'\n');tmp.replace(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def check(h):
    for p,v in h.items():assert sha(p)==v,p
def checkpoint(seed):
    if seed in [59111,59117]:return DATA.parent/f'posture_credit_01/ppo_01/seed_{seed}/train/final.pt'
    assert seed in [60111,60117]
    return OLD/f'ppo_01/seed_{seed}/block_03/train/final.pt'
def setarg(cmd,key,value):cmd[cmd.index(key)+1]=str(value)
def bank_args(cmd,bank):
    p=DATA/'banks'/bank;setarg(cmd,'--bank',p);setarg(cmd,'--num-envs',2*len(read(p/'geometry.json')))
    if '--reset-library' in cmd:
        i=cmd.index('--reset-library');del cmd[i:i+2]
    return cmd
def baseline(folder,gpu,label,kind='reference',bank='development'):
    if kind=='reference':cmd=posture_command(folder,gpu,label,'zero')
    elif int(kind) in [59111,59117]:cmd=posture_command(folder,gpu,label,'evaluate',checkpoint=checkpoint(int(kind)))
    else:
        seed=int(kind);cmd=earlier_command(folder,gpu,label,'evaluate',seed=seed,checkpoint=checkpoint(seed))
    return bank_args(cmd,bank)

def repaired_baseline(folder,gpu,label,kind='reference',bank='training'):
    cmd=baseline(folder,gpu,label,kind,bank);cmd[2]='../scripts/run_positive_selected_baseline.py'
    cmd+=['--positive-reference',str(DATA/'selected_reference.json')]
    return cmd

def command(folder,gpu,label,mode,seed=61111,start=9.,cp=None,updates=1,bank='training'):
    cmd=repaired_baseline(folder,gpu,label,'59117',bank)
    cmd[2]='../scripts/run_positive_slope_ppo.py'
    cmd+=['--early-mode',mode,'--early-start',str(start),'--early-seed',str(seed),
          '--early-span',str(SPAN),'--early-ramp',str(RAMP),'--early-updates',str(updates)]
    if mode!='calibrate':cmd+=['--early-calibration',str(DATA/'preflight/calibrate/early_calibration.json')]
    if cp:cmd+=['--early-checkpoint',str(cp)]
    return cmd
