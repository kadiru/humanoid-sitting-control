"""New experiment only: frozen resting policy plus an earlier bounded actor."""
import hashlib,json
from pathlib import Path
from posture_credit_common import command as parent_command
ROOT=Path('/home/kadir/humanoid-sitting')
DATA=Path('/data/humanoid-sitting-perception/earlier_control_01')
OLD=DATA.parent/'posture_credit_01'
SEEDS=[60111,60117]
PARENTS={60111:59111,60117:59117}
STAGES=[10.,9.,8.,6.]
EARLY_SPAN=.001
PREFLIGHT=DATA/'preflight_bounded_02'
def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def check(h):
    for p,v in h.items():assert sha(p)==v,p
def parent_checkpoint(seed):return OLD/f'ppo_01/seed_{PARENTS[seed]}/train/final.pt'
def baseline(folder,gpu,label,kind,bank='development',seed=60111):
    cp=None if kind=='reference' else parent_checkpoint(seed)
    cmd=parent_command(folder,gpu,label,'zero' if cp is None else 'evaluate',checkpoint=cp)
    if bank!='development':
        path=OLD/'reserved_bank_01' if bank=='retention' else DATA/'wide_bank_01'
        cmd[cmd.index('--bank')+1]=str(path);cmd[cmd.index('--num-envs')+1]='16' if bank=='retention' else '32'
        if '--reset-library' in cmd:
            i=cmd.index('--reset-library');del cmd[i:i+2]
    return cmd
def command(folder,gpu,label,mode,seed=60111,start=10.,checkpoint=None,updates=1,bank='development'):
    cmd=baseline(folder,gpu,label,'parent',bank,seed);cmd[2]='../scripts/run_earlier_control_ppo.py'
    cmd+=['--early-mode',mode,'--early-start',str(start),'--early-seed',str(seed),'--early-updates',str(updates),'--early-span',str(EARLY_SPAN)]
    if mode!='calibrate':cmd+=['--early-calibration',str(DATA/f'preflight/calibrate_{seed}/early_calibration.json')]
    if checkpoint:cmd+=['--early-checkpoint',str(checkpoint)]
    return cmd
