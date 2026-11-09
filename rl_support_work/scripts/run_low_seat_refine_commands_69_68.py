from pathlib import Path
import json,hashlib,datetime,subprocess,time
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');R=Path('/home/kadir/humanoid-sitting');status=D/'refine_commands_status_69_68.json'
def write(phase,**kw):
 r=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),phase=phase,physics_ticks=0,optimizer_updates=0,holdout_used=False,**kw);status.write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)
try:
 assert not (D/'coupled_step_refined_69').exists() and not (D/'contact_command_screen_68').exists()
 write('waiting_for_existing_search67');started=time.monotonic()
 while not (D/'coupled_step_timing_screen_67/results.json').exists():
  assert time.monotonic()-started<1800,'Existing search67 did not finish within bounded wait'
  assert subprocess.run(['tmux','has-session','-t','low_seat_coupled_step_67'],capture_output=True).returncode==0,'Search67 exited without saved results'
  time.sleep(5)
 for source,phase in [('refine_low_seat_coupled_step_69.py','dense_refinement69'),('check_low_seat_contact_commands_68.py','static_command68')]:
  p=R/'scripts'/source;write(phase,source_sha256=hashlib.sha256(p.read_bytes()).hexdigest());subprocess.run([str(R/'IsaacLab/.venv/bin/python'),str(p)],check=True,cwd=R)
 write('complete',physical_admission=False)
except Exception as e:
 write('failed',error=repr(e));raise
