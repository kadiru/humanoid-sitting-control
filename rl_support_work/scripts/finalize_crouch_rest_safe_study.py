"""Persistent read-only completion, duplicate verification and report export."""
import os,subprocess,time,traceback
from run_crouch_rest_safe_study import ROOT,DATA,STUDY,SEEDS,read,status

def main():
    state=DATA/'finalization_status.json';env=dict(os.environ,PYTHONPATH='sitting_task/src:scripts')
    try:
        status(state,'waiting_for_workers')
        while True:
            states=[read(STUDY/f'status_{s}.json') if (STUDY/f'status_{s}.json').exists() else {} for s in SEEDS]
            assert not any(s.get('phase')=='failed' for s in states),states
            if all(s.get('phase')=='complete' for s in states):break
            time.sleep(20)
        for name in ['verified_01','verified_02']:
            status(state,'verifying',output=name)
            with (STUDY/(name+'.log')).open('x') as f:
                subprocess.run([str(ROOT/'IsaacLab/.venv/bin/python'),'scripts/verify_crouch_rest_safe_study.py','--study',str(STUDY),'--output',str(STUDY/name)],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        assert (STUDY/'verified_01/verification.json').read_bytes()==(STUDY/'verified_02/verification.json').read_bytes()
        status(state,'exporting')
        with (DATA/'final_export_01.log').open('x') as f:
            subprocess.run([str(ROOT/'IsaacLab/.venv/bin/python'),'scripts/report_crouch_rest_safe_study.py','--study',str(STUDY),'--verification',str(STUDY/'verified_02'),'--output',str(DATA/'final_export_01')],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        status(state,'verified_export_ready',export=str(DATA/'final_export_01'));status(DATA/'status.json','verified_export_ready',export=str(DATA/'final_export_01'))
    except Exception as e:
        status(state,'failed',error=repr(e),traceback=traceback.format_exc());raise
if __name__=='__main__':main()
