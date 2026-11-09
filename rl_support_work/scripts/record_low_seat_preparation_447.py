"""Preserve prepared444–447; no new candidate, final report or task closure."""
from pathlib import Path
import json,hashlib,datetime,subprocess,shutil
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');R=Path('/home/kadir/humanoid-sitting')
read=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def main():
 assert not (D/'preparation_handoff_record_447.json').exists();checks=[]
 for name in ['prior_hashes.json','completed_probes_preservation_443.json','supplemental_preservation_443.json']:
  m=read(D/name)
  for p,h in m.items():assert sha(p)==h,p
  checks.append(dict(manifest=name,files=len(m),passed=True))
 folders=['clock_transcription_preparation_444','clock_transcription_preparation_audit_445','clock_row_preparation_446','clock_row_preparation_v2_446','clock_row_audit_447']
 for name in folders:
  for p,h in read(D/name/'SHA256.json').items():assert sha(D/name/p)==h,p
 assert read(D/'clock_row_audit_447/independent_audit.json')['passed']
 assert read(D/'clock_row_audit_exit_447.json')['actual_exit']==0
 workers=[]
 for line in subprocess.check_output(['ps','-eo','pid,args'],text=True).splitlines():
  a=line.split()
  if len(a)>2 and a[1].endswith('IsaacLab/.venv/bin/python') and a[2].startswith('scripts/') and Path(a[2]).name!=Path(__file__).name:workers.append(line)
 assert not workers,workers
 for source,name in [(D/'status.json','status_before_clock_transcription_447.json'),(R/'docs/LOW_SEAT_CONTACT_STATUS.md','handoff_before_clock_transcription_447.md')]:
  assert not (D/name).exists();shutil.copy2(source,D/name)
 mf=read(D/'completed_probes_preservation_443.json');mf.update(read(D/'supplemental_preservation_443.json'))
 for name in folders:
  for p in (D/name).rglob('*'):
   if p.is_file():mf[str(p)]=sha(p)
 scripts=['low_seat_clock_transcription_444.py','prepare_low_seat_clock_transcription_444.py','audit_low_seat_clock_transcription_445.py','low_seat_clock_rows_446.py','prepare_low_seat_clock_rows_446.py','low_seat_clock_rows_v2_446.py','finalize_low_seat_clock_rows_v2_446.py','audit_low_seat_clock_rows_447.py','record_low_seat_preparation_447.py']
 for n in scripts:mf[str(R/'scripts'/n)]=sha(R/'scripts'/n)
 names=['clock_transcription_preparation_launch_444.json','clock_transcription_preparation_444.log','clock_transcription_preparation_exit_444.json','clock_transcription_audit_launch_445.json','clock_transcription_audit_445.log','clock_transcription_audit_exit_445.json','clock_row_preparation_launch_446.json','clock_row_preparation_446.log','clock_row_preparation_exit_446.json','clock_row_correction_launch_v2_446.json','clock_row_finalize_v2_446.log','clock_row_finalize_exit_v2_446.json','clock_row_audit_447.log','clock_row_audit_exit_447.json','clock_transcription_local_hashes_447.json','status_before_clock_transcription_447.json','handoff_before_clock_transcription_447.md','handoff_prepared_447.txt']
 for n in names:mf[str(D/n)]=sha(D/n)
 manifest=D/'clock_transcription_evidence_preservation_447.json';manifest.write_text(json.dumps(mf,indent=2))
 for p,h in mf.items():assert sha(p)==h,p
 now=datetime.datetime.now(datetime.timezone.utc).isoformat();notes=(D/'handoff_prepared_447.txt').read_text();status=read(D/'status.json')
 status.update(utc=now,updated_utc=now,phase='offline_clock_collocation_prepared_no_candidate_447',audit_jobs_running=False,model_jobs_running=False,physical_jobs_running=False,latest_active_run=None,latest_active_snapshot=None,latest_handoff=str(D/'handoff_prepared_447.txt'),latest_preservation=str(manifest),completed_evidence_manifest=str(manifest),completed_evidence_count=len(mf),last_sealed_closure=str(D/'closure_record_443.json'),latest_preparation=str(D/'clock_row_preparation_v2_446'),latest_preparation_audit=str(D/'clock_row_audit_447/independent_audit.json'),latest_preparation_audit_passed=True,latest_preparation_scope='Whole-clock mechanics, original/dense rows and implicit map audited. New bounded whole-path solver/evaluation wrapper NOT implemented/frozen/launched. No new candidate;latest complete candidate440 stays rejected.',latest_round_constructed_model_records=0,latest_round_saved_primitive_replays=1467,latest_round_new_derivative_fixture_columns=144,latest_round_new_root_fixture_states=12,latest_round_new_full20s_trials=0,latest_round_initial_settling_ticks=0,latest_round_new_active_ticks=0,latest_round_new_diagnostic_prefixes=0,latest_round_new_aborted_case_attempts=0,latest_round_optimizer_updates=0,latest_round_actual_forward_filter_ticks=0,latest_round_command_ticks_per_device=0,latest_round_offline_milp_calls=0,latest_round_offline_nlp_calls=0,latest_round_offline_nonlinear_iterations=0,latest_round_offline_objective_evaluations=0,latest_round_offline_jacobian_evaluations=0,latest_round_offline_nonlinear_trial_evaluations=0,latest_round_offline_accepted_steps=0,main_training_started=False,ppo_started=False,fresh_holdout_started=False,holdout_started=False,policy_promoted=False,task_complete=False,repair_complete=False,notes=notes)
 (D/'status.json').write_text(json.dumps(status,indent=2));doc=R/'docs/LOW_SEAT_CONTACT_STATUS.md';doc.write_text(notes+'\n\nEarlier preserved entries follow.\n\n'+doc.read_text())
 record=dict(utc=now,scope=__doc__,completed='444local/meshprep,445savedalgebra,446globalrows/12newrootfixtures,446v2saved-only5extraneous-edge removal,447independentrowaudit',prepared_model_only=True,prior_checks=checks,protected_files=len(mf),manifest=str(manifest),manifest_sha256=sha(manifest),workers=workers,actual_exits={'444':'67d482/session73061/exit0','445':'1d0ee8/exit0','446':'f822b5/session99651/exit0','446v2':'daa6e5/session20828/exit0','447':'bfba71/session61241/exit0'},local_hash_review=read(D/'clock_transcription_local_hashes_447.json'),new_complete_candidate=False,trajectory_optimizer_calls=0,actual_forward_filter_ticks=0,physics_ticks=0,policy_updates=0,last_sealed_closure=str(D/'closure_record_443.json'),current_reviewed_report=str(D/'progress_report_442/findings.html'),video307_unchanged=True,best_clean='64/72',protected_clean=64,unresolved_cases=8,repair_complete=False,task_complete=False,automation_status='ACTIVE')
 (D/'preparation_handoff_record_447.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2),flush=True)
if __name__=='__main__':main()
