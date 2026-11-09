"""Wait for the two frozen workers, verify twice and export; never add training."""
import subprocess,time,traceback
from pathlib import Path
from run_progressive_seat_study import ROOT,DATA,SEEDS,read,status

def main():
 study=DATA/'ppo_01';state=DATA/'finalization_status.json'
 try:
  status(state,'waiting_for_workers')
  while True:
   states=[read(study/f'status_{s}.json') if (study/f'status_{s}.json').exists() else {} for s in SEEDS]
   assert not any(s.get('phase')=='failed' for s in states),states
   if all(s.get('phase')=='complete' for s in states):break
   time.sleep(20)
  for n in ['verified_01','verified_02']:
   status(state,'verifying',output=n)
   with (study/(n+'.log')).open('x') as f:subprocess.run([str(ROOT/'IsaacLab/.venv/bin/python'),'scripts/verify_progressive_seat_study.py','--study',str(study),'--output',str(study/n)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
  assert (study/'verified_01/verification.json').read_bytes()==(study/'verified_02/verification.json').read_bytes()
  status(state,'exporting')
  with (DATA/'final_export_01.log').open('x') as f:subprocess.run([str(ROOT/'IsaacLab/.venv/bin/python'),'scripts/report_progressive_seat_study.py','--study',str(study),'--verification',str(study/'verified_02'),'--output',str(DATA/'final_export_01')],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
  status(state,'verified_export_ready',export=str(DATA/'final_export_01'));status(DATA/'status.json','verified_export_ready',export=str(DATA/'final_export_01'))
 except Exception as e:status(state,'failed',error=repr(e),traceback=traceback.format_exc());raise
if __name__=='__main__':main()
