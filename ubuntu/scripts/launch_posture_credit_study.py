"""Wait for both physical gates, freeze, then launch only authorized budgets."""
import subprocess,time,traceback
from posture_credit_common import *
from run_transfer_assessment import status
state=DATA/'launch_status.json'
try:
 status(state,'waiting_for_both_preflights')
 while True:
  phases=[read(DATA/f'preflight_status_gpu{i}.json')['phase'] for i in [0,1]]
  assert 'failed' not in phases,phases
  if phases==['complete','complete']:break
  time.sleep(15)
 subprocess.run([str(ROOT/'IsaacLab/.venv/bin/python'),'scripts/freeze_posture_credit_study.py'],cwd=ROOT,check=True)
 for gpu,seed in enumerate(SEEDS):
  shell=f'cd {ROOT} && PYTHONPATH=sitting_task/src:scripts IsaacLab/.venv/bin/python scripts/run_posture_credit_study.py --gpu {gpu} > {DATA}/ppo_01/worker_{seed}.log 2>&1'
  subprocess.run(['tmux','new-session','-d','-s',f'posture_credit_ppo_{seed}',shell],check=True)
 status(state,'main_training_launched',seeds=SEEDS,total_transitions=368100)
 status(DATA/'status.json','ppo_running',seeds=SEEDS)
except Exception as e:status(state,'failed',error=repr(e),traceback=traceback.format_exc());raise
