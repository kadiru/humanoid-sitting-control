"""Wait for both exact-zero/optimizer preflights, freeze once, launch once."""
import time,subprocess,traceback
from low_positive_common import *
from run_transfer_assessment import status
from freeze_low_positive_study import main as freeze
state=DATA/'launch_status.json'
try:
    status(state,'waiting_preflight')
    while True:
        rows=[read(DATA/f'preflight_recovery_status_{seed}.json') if (DATA/f'preflight_recovery_status_{seed}.json').exists() else {} for seed in [62111,62117]]
        if any(r.get('phase')=='failed' for r in rows):raise RuntimeError(rows)
        if all(r.get('phase')=='passed' for r in rows):break
        time.sleep(20)
    status(state,'freezing');freeze()
    for gpu,seed in enumerate([62111,62117]):
        cmd=f'cd {ROOT} && PYTHONPATH=scripts:sitting_task/src IsaacLab/.venv/bin/python scripts/run_low_positive_study.py --gpu {gpu} --seed {seed} > {DATA}/worker_{seed}.log 2>&1'
        subprocess.run(['tmux','new-session','-d','-s',f'low_positive_ppo_{seed}',cmd],check=True)
    cmd=f'cd {ROOT} && PYTHONPATH=scripts:sitting_task/src IsaacLab/.venv/bin/python scripts/finalize_low_positive_study.py > {DATA}/finalizer.log 2>&1'
    subprocess.run(['tmux','new-session','-d','-s','low_positive_finalize',cmd],check=True)
    status(state,'launched',seeds=[62111,62117],budget=917000,no_policy_promotion=True)
    status(DATA/'status.json','main_training',main_training_started=True,holdout_started=False,seeds=[62111,62117],budget=917000,all_training_start_s=10.,continuous_residual_window_s=[10.,20.],manual_repairs_rejected=True)
except Exception as e:status(state,'failed',error=repr(e),traceback=traceback.format_exc());raise
