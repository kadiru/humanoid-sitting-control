"""Persistent read-only verification/export after the existing worker finishes."""
import json,subprocess,time,traceback
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path('/home/kadir/humanoid-sitting')
DATA=Path('/data/humanoid-sitting-perception/contact_effort_repair_01')
STUDY=DATA/'sitting_ppo_01'
PY=ROOT/'IsaacLab/.venv/bin/python'
def status(phase,**kw):
    (DATA/'sitting_report_status.json').write_text(json.dumps(dict(phase=phase,utc=datetime.now(timezone.utc).isoformat(),**kw),indent=2))
def run(label,cmd):
    with (DATA/(label+'.log')).open('x') as f:
        subprocess.run([str(x) for x in cmd],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
try:
    status('waiting_for_existing_study')
    while True:
        s=json.loads((STUDY/'status.json').read_text())
        if s['phase']=='verified_complete':break
        if s['phase']=='failed':raise RuntimeError(s)
        time.sleep(10)
    status('independent_verification')
    run('sitting_independent_verify_02',[PY,ROOT/'scripts/run_repaired_sitting_comparison.py','--verify',STUDY,'--output',STUDY/'verified_02'])
    one=json.loads((STUDY/'verified_01/verification.json').read_text());two=json.loads((STUDY/'verified_02/verification.json').read_text());assert one==two
    status('building_report')
    run('sitting_export_01',[PY,ROOT/'scripts/report_repaired_sitting.py','--study',STUDY,'--verification',STUDY/'verified_02/verification.json','--output',DATA/'sitting_export_01'])
    status('exported_pending_visual_review',output=str(DATA/'sitting_export_01'),independent_verification_matches=True)
except Exception as e:
    status('failed',error=repr(e),traceback=traceback.format_exc());raise
