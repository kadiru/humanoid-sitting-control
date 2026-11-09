"""Commands and immutable paths for the separate posture-credit study."""
import json,hashlib
from pathlib import Path
from run_traction_study import command as old_command
ROOT=Path('/home/kadir/humanoid-sitting')
DATA=Path('/data/humanoid-sitting-perception/posture_credit_01')
OLD=DATA.parent/'traction_transfer_01'
SEEDS=[59111,59117]
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def command(folder,gpu,label,mode,seed=59101,checkpoint=None,updates=16,holdout=False):
    cmd=old_command(folder,gpu,label,'evaluate' if checkpoint else 'reference',seed=seed,checkpoint=checkpoint)
    cmd[2]='../scripts/run_posture_episode_ppo.py'
    cmd+=['--episode-mode',mode,'--episode-updates',str(updates),'--episode-span','.008']
    if mode!='calibrate':cmd+=['--episode-calibration',str(DATA/'preflight/calibrate/episode_calibration.json')]
    if holdout:
        cmd[cmd.index('--bank')+1]=str(DATA/'reserved_bank_01');cmd[cmd.index('--num-envs')+1]='16'
        if '--reset-library' in cmd:
            i=cmd.index('--reset-library');del cmd[i:i+2]
    return cmd
def check_hashes(hashes):
    for p,h in hashes.items():assert sha(p)==h,p
