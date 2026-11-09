"""Command construction for the isolated low-seat study."""
from pathlib import Path
from positive_slope_common_v5 import command as old_command,setarg,read
ROOT=Path('/home/kadir/humanoid-sitting')
DATA=Path('/data/humanoid-sitting-perception/low_positive_transfer_01')
OLD=DATA.parent/'positive_slope_01'

def setbank(cmd,bank):
    bank=Path(bank);setarg(cmd,'--bank',bank);setarg(cmd,'--num-envs',2*len(read(bank/'geometry.json')))
    return cmd

def feedback_command(folder,gpu,label,mode,bank,repair,kind='timing',seed=62111,start=10.,updates=1,cal=None,cp=None):
    cmd=old_command(folder,gpu,label,'zero',seed=seed,start=8.,bank='training')
    cmd[2]='../scripts/run_low_positive_feedback.py';setbank(cmd,bank)
    cmd+=['--feedback-mode',mode,'--feedback-start',str(start),'--feedback-seed',str(seed),
          '--feedback-updates',str(updates),'--repair-config',str(repair),'--repair-kind',kind]
    if cal:cmd+=['--feedback-calibration',str(cal)]
    if cp:cmd+=['--feedback-checkpoint',str(cp)]
    return cmd
