"""Persistent repair/preflight gate; never launches main after a failed check."""
import subprocess,shlex,time,traceback
from positive_slope_common_v4 import *
from run_transfer_assessment import status

def tmux(name,script,arguments,log):
    cmd=['env','PYTHONPATH=scripts:sitting_task/src',str(ROOT/'IsaacLab/.venv/bin/python'),str(ROOT/'scripts'/script),*arguments]
    shell='cd '+shlex.quote(str(ROOT))+' && '+shlex.join(cmd)+' > '+shlex.quote(str(log))+' 2>&1'
    subprocess.run(['tmux','new-session','-d','-s',name,shell],check=True)

def main():
    state=DATA/'launch_v2_status.json'
    try:
        status(state,'waiting_for_repaired_reference_and_parent')
        while not all((DATA/f'repair_06/{k}/episode_contract.json').exists() for k in ['reference','parent']):
            for k in ['reference','parent']:
                log=DATA/f'repair_06/{k}.log';assert not log.exists() or 'Traceback (most recent call last)' not in log.read_text(),log
            time.sleep(15)
        from verify_seat_load_study import evaluation
        from positive_slope_validation_v4 import physical,matched
        outcomes={}
        for k in ['reference','parent']:
            folder=DATA/f'repair_06/{k}';s,rows,g=evaluation(folder,24);physical(folder,require_clean=False);matched(DATA/'assessment/reference',folder,10.)
            admitted=[x for x in rows if not ((g[x['geometry_index']]['height']<.38 and g[x['geometry_index']]['pitch_deg']>=15) or (abs(g[x['geometry_index']]['height']-.4)<1e-6 and g[x['geometry_index']]['pitch_deg']>=20))]
            assert len(admitted)==18 and all(x['success'] and not any(x['failures'].values()) and not x['slip_ticks'] and not x['illegal_ticks'] for x in admitted),(k,[(x['case'],x['success'],x['slip_ticks'],x['illegal_ticks'],x['failures']) for x in admitted])
            outcomes[k]=s
        write(DATA/'repair_06/admitted_validation.json',dict(passed=True,outcomes=outcomes,admitted=18,stress=6,strict_margin_not_claimed=True))
        status(state,'waiting_for_pulse_preflight')
        while True:
            states=[read(DATA/f'preflight_pulse_04_status_gpu{g}.json') if (DATA/f'preflight_pulse_04_status_gpu{g}.json').exists() else {} for g in [0,1]]
            assert not any(x.get('phase')=='failed' for x in states),states
            if all(x.get('phase')=='complete' for x in states):break
            time.sleep(15)
        status(state,'independent_admission_and_freeze')
        subprocess.run([str(ROOT/'IsaacLab/.venv/bin/python'),str(ROOT/'scripts/freeze_positive_slope_study_v4.py')],check=True)
        protocol=read(DATA/'ppo_01/protocol.json');assert protocol['admitted'];check(protocol['hashes']);check(protocol['prior_hashes'])
        for gpu,seed in enumerate(SEEDS):tmux(f'positive_slope_ppo_{seed}','run_positive_slope_study_v4.py',['--gpu',str(gpu)],DATA/f'worker_{seed}.log')
        tmux('positive_slope_finalize','finalize_positive_slope_study_v4.py',[],DATA/'finalizer.log')
        status(state,'main_launched',seeds=SEEDS,transitions=439200,protocol_sha256=sha(DATA/'ppo_01/protocol.json'))
    except Exception as e:status(state,'failed',error=repr(e),traceback=traceback.format_exc());raise
if __name__=='__main__':main()
