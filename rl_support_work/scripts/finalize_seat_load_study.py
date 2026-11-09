"""Wait for the two bounded workers, independently verify twice and package."""
import json,subprocess,time,traceback
from pathlib import Path
from run_transfer_assessment import write,status

ROOT=Path('/home/kadir/humanoid-sitting');DATA=Path('/data/humanoid-sitting-perception/seat_load_transfer_01');STUDY=DATA/'ppo_01';PY=ROOT/'IsaacLab/.venv/bin/python'

def main():
    try:
        while True:
            states=[json.loads((STUDY/f'status_{s}.json').read_text()) for s in [56811,56817]]
            assert not any(s['phase']=='failed' for s in states),states
            if all(s['phase']=='complete' for s in states):break
            time.sleep(20)
        status(DATA/'status.json','independent_verification',training_started=True)
        for name in ['verified_01','verified_02']:
            cmd=[str(PY),'scripts/verify_seat_load_study.py','--study',str(STUDY),'--output',str(STUDY/name)]
            with (STUDY/(name+'.log')).open('x') as f:subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
        a=json.loads((STUDY/'verified_01/verification.json').read_text());b=json.loads((STUDY/'verified_02/verification.json').read_text());assert a==b
        cmd=[str(PY),'scripts/report_seat_load_study.py','--study',str(STUDY),'--verification',str(STUDY/'verified_02'),'--output',str(DATA/'final_export_01')]
        with (DATA/'final_export_01.log').open('x') as f:subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
        status(DATA/'status.json','verified_export_ready',export=str(DATA/'final_export_01'),verification=str(STUDY/'verified_02'),copies_identical=True,policy_promoted=False)
    except Exception as e:
        status(DATA/'status.json','finalization_failed',error=repr(e),traceback=traceback.format_exc());raise

if __name__=='__main__':main()
