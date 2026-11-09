"""Persistent finalization only; never starts or extends training."""
import subprocess,time,traceback
from positive_slope_common_v3 import *
from run_transfer_assessment import status

def main():
    study=DATA/'ppo_01';state=DATA/'finalization_status.json'
    try:
        status(state,'waiting_for_both_workers')
        while True:
            states=[read(study/f'status_{s}.json') if (study/f'status_{s}.json').exists() else {} for s in SEEDS]
            assert not any(x.get('phase')=='failed' for x in states),states
            if all(x.get('phase')=='complete' for x in states):break
            time.sleep(20)
        python=str(ROOT/'IsaacLab/.venv/bin/python')
        for name in ['verified_01','verified_02']:
            status(state,name);subprocess.run([python,str(ROOT/'scripts/verify_positive_slope_study_v3.py'),'--study',str(study),'--output',str(study/name)],check=True)
        assert read(study/'verified_01/verification.json')==read(study/'verified_02/verification.json')
        status(state,'report_and_video');subprocess.run([python,str(ROOT/'scripts/report_positive_slope_study.py'),'--study',str(study),'--verification',str(study/'verified_02/verification.json'),'--output',str(DATA/'final_export_01')],check=True)
        status(DATA/'status.json','comparison_complete_awaiting_local_review',main_training_complete=True,holdout_complete=True,export=str(DATA/'final_export_01'))
        status(state,'complete',export=str(DATA/'final_export_01'),video=read(DATA/'final_export_01/video/verification.json'),local_copy_and_visual_review_pending=True)
    except Exception as e:status(state,'failed',error=repr(e),traceback=traceback.format_exc());raise
if __name__=='__main__':main()
