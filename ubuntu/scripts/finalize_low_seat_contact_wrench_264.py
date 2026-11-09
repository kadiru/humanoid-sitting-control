"""Wait once for259, then independently audit before diagnosis/video. No physics."""
from pathlib import Path
import datetime,json,subprocess,sys,time,os
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');S=D/'contact_wrench_finalization_status_264.json'
def state(phase,**kw):S.write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),phase=phase,**kw),indent=2))
def main():
 assert not S.exists();state('waiting_for_physics')
 while not (D/'contact_wrench_process_exit_259.json').exists():time.sleep(10)
 exitstatus=json.loads((D/'contact_wrench_process_exit_259.json').read_text());assert exitstatus['returncode']==0,exitstatus
 for name in ['audit_low_seat_contact_wrench_260.py','diagnose_low_seat_contact_wrench_261.py','render_low_seat_all_attempts_262.py','verify_low_seat_all_attempts_263.py']:
  state('running',script=name);subprocess.run([sys.executable,'scripts/'+name],check=True)
 state('complete',physical_admission=False,repair_complete=False)
if __name__=='__main__':
 try:main()
 except BaseException as e:state('failed',error=repr(e));raise
