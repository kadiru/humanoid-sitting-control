"""Persistent report worker; does not train, tune or change research definitions."""
import time,traceback
from run_traction_study import DATA,STUDY,SEEDS,read
from run_transfer_assessment import run,status
state=DATA/'finalization_status.json'
try:
    while True:
        phases=[read(STUDY/f'status_{s}.json').get('phase') if (STUDY/f'status_{s}.json').exists() else 'pending' for s in SEEDS]
        assert 'failed' not in phases,phases
        if all(p=='complete' for p in phases):break
        status(state,'waiting_for_workers',workers=dict(zip(SEEDS,phases)));time.sleep(30)
    for label in ['verified_01','verified_02']:
        status(state,'verifying',active=label)
        run(STUDY,label,['IsaacLab/.venv/bin/python','scripts/verify_traction_study.py','--study',str(STUDY),'--output',str(STUDY/label)])
    assert read(STUDY/'verified_01/verification.json')==read(STUDY/'verified_02/verification.json')
    status(state,'exporting');run(DATA,'final_export_01',['IsaacLab/.venv/bin/python','scripts/report_traction_study.py','--study',str(STUDY),'--verified',str(STUDY/'verified_02'),'--output',str(DATA/'final_export_01')]);status(state,'complete',export=str(DATA/'final_export_01'))
except Exception as e:status(state,'failed',error=repr(e),traceback=traceback.format_exc());raise
