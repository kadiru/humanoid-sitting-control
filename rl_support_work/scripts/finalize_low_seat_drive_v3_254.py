"""Offline audits/video/report once, after completed246. No physics followup."""
from pathlib import Path
import datetime,json,subprocess,sys,traceback
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');R=Path('/home/kadir/humanoid-sitting');S=D/'drive_finalization_recovery_v3_status_254.json'
def status(**kw):S.write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw),indent=2))
def main():
 assert not S.exists();assert json.loads((D/'drive_process_exit_246.json').read_text())['returncode']==0
 assert json.loads((D/'drive_diagnostic_246/completion_before_close.json').read_text())['completed']
 try:
  for name,log in [('verify_low_seat_all_attempts_v3_251.py','all_attempts_video_audit_v3_251.log'),('report_low_seat_drive_252.py','progress_report_252.log')]:
   status(phase='running',job=name,repair_complete=False)
   with (D/log).open('x') as f:subprocess.run([sys.executable,str(R/'scripts'/name)],cwd=R,stdout=f,stderr=subprocess.STDOUT,check=True)
  status(phase='complete',repair_complete=False,physical_jobs_running=False,main_training_started=False,fresh_holdout_started=False)
 except BaseException:status(phase='failed',traceback=traceback.format_exc(),repair_complete=False);raise
if __name__=='__main__':main()
