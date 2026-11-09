"""Wait, audit twice, then export the full comparison and complete short video."""
import subprocess,time,traceback
from earlier_control_common import *
from run_transfer_assessment import status
def main():
    study=DATA/'ppo_01';state=DATA/'finalization_status.json';python=str(ROOT/'IsaacLab/.venv/bin/python')
    try:
        status(state,'waiting_for_runs')
        while True:
            states=[read(study/f'status_{s}.json') if (study/f'status_{s}.json').exists() else {} for s in SEEDS]
            for x in states:assert x.get('phase')!='failed',x
            if (DATA/'launch_status.json').exists():assert read(DATA/'launch_status.json')['phase']!='failed'
            if all(x.get('phase')=='complete' for x in states):break
            time.sleep(20)
        for label in ['verified_01','verified_02']:
            status(state,'verifying',output=label)
            subprocess.run([python,'scripts/verify_earlier_control_study.py','--study',str(study),'--output',str(study/label)],cwd=ROOT,check=True)
        assert read(study/'verified_01/verification.json')==read(study/'verified_02/verification.json')
        status(state,'exporting_report_and_video')
        subprocess.run([python,'scripts/report_earlier_control_study.py','--study',str(study),'--output',str(DATA/'final_export_01')],cwd=ROOT,check=True)
        status(state,'complete',export=str(DATA/'final_export_01'));status(DATA/'status.json','complete_pending_independent_review',export=str(DATA/'final_export_01'))
    except Exception as e:status(state,'failed',error=repr(e),traceback=traceback.format_exc());raise
if __name__=='__main__':main()
