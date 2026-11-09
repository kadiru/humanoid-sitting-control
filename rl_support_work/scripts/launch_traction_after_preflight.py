"""Gate main PPO on completed preflight; never relax a failed check or retry tuning."""
import time,subprocess,shlex,traceback
from run_traction_study import DATA,STUDY,SEEDS,read
from run_transfer_assessment import run,status
state=DATA/'launch_status.json'
try:
    while True:
        phases=[read(DATA/f'smoke_status_{s}.json').get('phase') if (DATA/f'smoke_status_{s}.json').exists() else 'pending' for s in SEEDS]
        assert 'failed' not in phases,phases
        if all(p=='complete' for p in phases):break
        status(state,'waiting_for_optimizer_smokes',workers=dict(zip(SEEDS,phases)));time.sleep(20)
    status(state,'validating_smokes')
    run(DATA,'preflight_with_smokes_01',['IsaacLab/.venv/bin/python','scripts/validate_traction_preflight.py','--with-smokes','--output',str(DATA/'preflight_with_smokes_01.json')])
    status(state,'freezing_protocol');run(DATA,'protocol_freeze_01',['IsaacLab/.venv/bin/python','scripts/freeze_traction_study.py'])
    assert read(STUDY/'protocol.json')['admitted']
    for gpu,seed in enumerate(SEEDS):
        cmd=['IsaacLab/.venv/bin/python','scripts/run_traction_study.py','--gpu',str(gpu),'--seed',str(seed)]
        shell='cd /home/kadir/humanoid-sitting && PYTHONPATH=sitting_task/src:scripts '+shlex.join(cmd)+' > '+str(DATA/f'worker_{seed}.log')+' 2>&1'
        subprocess.run(['tmux','new-session','-d','-s',f'traction_ppo_{seed}','bash','-lc',shell],check=True)
    shell='cd /home/kadir/humanoid-sitting && PYTHONPATH=sitting_task/src:scripts IsaacLab/.venv/bin/python scripts/finalize_traction_study.py > '+str(DATA/'finalizer.log')+' 2>&1'
    subprocess.run(['tmux','new-session','-d','-s','traction_finalize_01','bash','-lc',shell],check=True)
    status(state,'main_training_launched',seeds=SEEDS,total_physics_transitions=368640)
    status(DATA/'status.json','ppo_running',seeds=SEEDS,protocol=str(STUDY/'protocol.json'),main_training_started=True)
except Exception as e:status(state,'failed',error=repr(e),traceback=traceback.format_exc());raise
