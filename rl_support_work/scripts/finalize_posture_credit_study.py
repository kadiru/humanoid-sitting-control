"""Persistent verification/export only; never extends training."""
import time,traceback
from posture_credit_common import *
from run_transfer_assessment import run,status
study=DATA/'ppo_01';state=DATA/'finalization_status.json'
try:
 while True:
  phases=[read(study/f'status_{s}.json')['phase'] if (study/f'status_{s}.json').exists() else 'pending' for s in SEEDS]
  assert 'failed' not in phases,phases
  if (DATA/'launch_status.json').exists():assert read(DATA/'launch_status.json')['phase']!='failed'
  if phases==['complete','complete']:break
  status(state,'waiting_for_workers',workers=dict(zip(SEEDS,phases)));time.sleep(30)
 for label in ['verified_01','verified_02']:
  status(state,'verifying',active=label);run(study,label,['IsaacLab/.venv/bin/python','scripts/verify_posture_credit_study.py','--study',str(study),'--output',str(study/label)])
 assert read(study/'verified_01/verification.json')==read(study/'verified_02/verification.json')
 status(state,'exporting');run(DATA,'final_export_01',['IsaacLab/.venv/bin/python','scripts/report_posture_credit_study.py','--study',str(study),'--verified',str(study/'verified_02'),'--output',str(DATA/'final_export_01')]);status(state,'complete',export=str(DATA/'final_export_01'))
except Exception as e:status(state,'failed',error=repr(e),traceback=traceback.format_exc());raise
