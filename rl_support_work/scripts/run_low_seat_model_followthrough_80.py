from pathlib import Path
import json,hashlib,datetime,subprocess,time
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');R=Path('/home/kadir/humanoid-sitting');status=D/'model_followthrough_status_80.json'
def update(phase,**kw):
 r=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),phase=phase,physics_ticks=0,optimizer_updates=0,holdout_used=False,physical_admission=False,**kw);status.write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)
def run(script):
 p=R/'scripts'/script;subprocess.run([str(R/'IsaacLab/.venv/bin/python'),str(p)],cwd=R,check=True)
try:
 for n in ['actuator_path_refinement_audit_79','landing_coupling_path_80','landing_coupling_path_audit_81']:assert not (D/n).exists(),n
 update('waiting_for_existing_refinement77');start=time.monotonic()
 while not (D/'actuator_contact_path_refinement_77/results.json').exists():
  assert time.monotonic()-start<1800,'Bounded wait expired'
  assert subprocess.run(['tmux','has-session','-t','low_seat_path_refinement_77'],capture_output=True).returncode==0,'77 exited without a result'
  time.sleep(10)
 update('independent_audit79');run('audit_low_seat_actuator_path_79.py');a=json.loads((D/'actuator_contact_path_refinement_77/results.json').read_text())['dense']
 eligible=a['model_geometry_passed'] and a['model_command_passed'] and not a['command_schedule_passed'] and a['parameters'][1]>=.429
 if not eligible:update('not_admitted',reason='77 did not establish the declared timing-only/active-parameter-bound prerequisite. No revision80 search ran.')
 else:
  p=R/'scripts/refine_low_seat_landing_coupling_80.py';record=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='New PRE-TRAINING bounded offline planning revision80. Audited77 reaches landing-crouch fraction0.43 while timing still fails. Extend ONLY that planning-variable upper bound to0.55; same3 variables, up to40coarse evaluations/10iterations plus one105-knot check. All physical/support/foot/clearance/terminal/joint/effort/rate constraints, arms, gains and sensors unchanged. No physics/PPO/holdout. Endpoint/pre-contact model results never establish sensed seat support.',source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),prerequisite77=a['parameters'],physics_ticks=0,optimizer_updates=0,holdout_used=False)
  f=D/'landing_coupling_revision_80.json';assert not f.exists();f.write_text(json.dumps(record,indent=2));update('bounded_landing_coupling80',source_sha256=record['source_sha256']);run(p.name);update('independent_audit81');run('audit_low_seat_landing_coupling_81.py');update('complete',next='Inspect all geometry/command/timing gates and actual pre-contact versus sensed-load requirements before any physical launch; PPO and fresh holdout remain closed.')
except Exception as e:update('failed',error=repr(e));raise
