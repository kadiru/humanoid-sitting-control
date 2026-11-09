"""Persistent launch gate; never bypass failed physical or optimizer admission."""
import subprocess,time,traceback,shlex
from earlier_control_common import *
from run_transfer_assessment import status
def main():
    state=DATA/'launch_status.json'
    try:
        status(state,'waiting_for_preflight')
        while True:
            states=[read(DATA/f'preflight_status_gpu{i}.json') for i in range(2)]
            assert not any(x['phase']=='failed' for x in states),states
            if all(x['phase']=='complete' for x in states):break
            time.sleep(20)
        status(state,'independent_admission')
        subprocess.run([str(ROOT/'IsaacLab/.venv/bin/python'),str(ROOT/'scripts/freeze_earlier_control_study.py')],check=True,cwd=ROOT)
        for gpu,seed in enumerate(SEEDS):
            cmd=f'cd {ROOT} && PYTHONPATH=sitting_task/src:scripts IsaacLab/.venv/bin/python scripts/run_earlier_control_study.py --gpu {gpu} > {DATA}/ppo_01/worker_{seed}.log 2>&1'
            subprocess.run(['tmux','new-session','-d','-s',f'earlier_control_ppo_{seed}',cmd],check=True)
        status(state,'launched',seeds=SEEDS);status(DATA/'status.json','ppo_running',seeds=SEEDS)
    except Exception as e:status(state,'failed',error=repr(e),traceback=traceback.format_exc());raise
if __name__=='__main__':main()
