"""Wait for these two physical probes, audit once; never launch training."""
from pathlib import Path
import sys,time,json,subprocess,traceback,datetime
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')
names=['normal_task_mild_05','normal_task_firm_05'];status=D/'task_audit_status_05.json'
def write(phase,**extra):
    status.write_text(json.dumps(dict(phase=phase,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**extra),indent=2))
try:
    write('waiting',main_training_started=False,fresh_holdout_started=False)
    while True:
        states=[json.loads((D/('status_'+n+'.json')).read_text()) for n in names]
        if any(s['phase']=='failed' for s in states):raise RuntimeError(states)
        if all(s['phase']=='complete' for s in states):break
        time.sleep(20)
    write('auditing',main_training_started=False,fresh_holdout_started=False)
    for n in names:
        for script,folder,result in [('audit_low_seat_contact_probe_v2.py',D/n,'independent_audit_v2.json'),
                                     ('audit_low_seat_sensor_control.py',D/n/'candidate','sensor_control_audit.json'),
                                     ('audit_low_seat_task_probe.py',D/n/'candidate','task_control_audit.json')]:
            if not (folder/result).exists():subprocess.run([sys.executable,'scripts/'+script,'--folder',str(folder)],check=True)
    write('complete',outcome_audits={n:str(D/n/'independent_audit_v2.json') for n in names},
          main_training_started=False,fresh_holdout_started=False,physical_admission_not_implied=True)
except Exception as exc:
    write('failed',error=repr(exc),traceback=traceback.format_exc());raise
