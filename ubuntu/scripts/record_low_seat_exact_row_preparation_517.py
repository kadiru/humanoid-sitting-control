"""Preserve independently proved real row quotients and the failed synthetic fixture; no new closure."""
from pathlib import Path
import json,hashlib,datetime,shutil
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');R=Path('/home/kadir/humanoid-sitting')
read=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def main():
 assert not (D/'preparation_handoff_record_517.json').exists();checks=[]
 for name in ['prior_hashes.json','completed_probes_preservation_515.json','supplemental_preservation_515.json']:
  m=read(D/name)
  for p,h in m.items():assert sha(p)==h,p
  checks.append(dict(manifest=name,files=len(m),passed=True))
 for launch,key in [('exact_row_quotient_launch_516.json','source_input_sha256'),('exact_row_quotient_launch_v2_516.json','source_input_sha256'),('exact_row_quotient_staged_launch_516.json','preserved_source_output_sha256')]:
  for p,h in read(D/launch)[key].items():assert sha(p)==h,p
 exits={}
 for n,want in [('exact_row_quotient_compile_exit_516.json',0),('exact_row_quotient_freeze_exit_516.json',0),('exact_row_quotient_compile_exit_v2_516.json',0),('exact_row_quotient_freeze_exit_v2_516.json',0),('exact_row_quotient_preparation_exit_516.json',1),('exact_row_quotient_staged_compile_exit_516.json',0),('exact_row_quotient_finalization_exit_516.json',0),('exact_row_quotient_audit_exit_517.json',0)]:
  exits[n]=read(D/n);assert exits[n]['actual_exit']==want
 folders=['exact_row_quotient_preparation_516','exact_row_quotient_audit_517']
 for folder in folders:
  for p,h in read(D/folder/'SHA256.json').items():assert sha(D/folder/p)==h,p
 prep=read(D/folders[0]/'preparation.json');audit=read(D/folders[1]/'independent_audit.json');review=read(D/'exact_row_quotient_local_verification_517.json');scope=read(D/'exact_row_quotient_scope_517.json')
 assert not prep['passed'] and prep['saved_full_systems_complete'] and audit['passed'] and review['all_hashes_verified']
 assert not audit['synthetic_fixture_passed'] and not audit['synthetic_fixture_repeated']
 workers=[]
 for p in Path('/proc').glob('[0-9]*/cmdline'):
  try:a=p.read_bytes().decode().split('\0')
  except (OSError,UnicodeError):continue
  if a and 'python' in a[0] and any((x.startswith('scripts/') or x.startswith(str(R/'scripts')+'/')) and Path(x).name!=Path(__file__).name for x in a[1:]):workers.append(a)
 assert not workers,workers
 for p,n in [(D/'status.json','status_before_exact_row_517.json'),(R/'docs/LOW_SEAT_CONTACT_STATUS.md','handoff_before_exact_row_517.md')]:
  assert not (D/n).exists();shutil.copy2(p,D/n)
 mf=read(D/'completed_probes_preservation_515.json');mf.update(read(D/'supplemental_preservation_515.json'))
 for folder in folders:
  for p in (D/folder).rglob('*'):
   if p.is_file():mf[str(p)]=sha(p)
 for pattern in ['*exact_row*516*.py','*exact_row*517*.py']:
  for p in (R/'scripts').glob(pattern):mf[str(p)]=sha(p)
 for pattern in ['exact_row_quotient_*.json','exact_row_quotient_*.log']:
  for p in D.glob(pattern):
   if p.is_file() and p.name!='exact_row_quotient_preservation_517.log':mf[str(p)]=sha(p)
 for n in ['handoff_prepared_517.txt','status_before_exact_row_517.json','handoff_before_exact_row_517.md']:mf[str(D/n)]=sha(D/n)
 manifest=D/'exact_row_quotient_evidence_preservation_517.json';manifest.write_text(json.dumps(mf,indent=2))
 for p,h in mf.items():assert sha(p)==h,p
 now=datetime.datetime.now(datetime.timezone.utc).isoformat();notes=(D/'handoff_prepared_517.txt').read_text();s=read(D/'status.json')
 for k in list(s):
  if k.startswith('latest_round_'):
   if isinstance(s[k],bool):s[k]=False
   elif isinstance(s[k],(int,float)):s[k]=0
   else:assert k=='latest_round_new_root_fixture_scope',k
 s.update(utc=now,updated_utc=now,phase='offline_exact_row516_real_system_audit517_complete_no_new_solver',latest_handoff=str(D/'handoff_prepared_517.txt'),latest_preparation=str(D/folders[0]),latest_preparation_audit=str(D/folders[1]/'independent_audit.json'),latest_preparation_preservation=str(manifest),latest_preservation=str(manifest),completed_evidence_manifest=str(manifest),completed_evidence_count=len(mf),latest_preparation_handoff_record=str(D/'preparation_handoff_record_517.json'),latest_preparation_handoff=str(D/'preparation_handoff_record_517.json'),last_sealed_closure=str(D/'closure_record_515.json'),latest_closure=str(D/'closure_record_515.json'),latest_attempt=str(D/'separated_path_512'),latest_complete_construction=str(D/'interior_path_484'),latest_audit_status='517 independent REAL-system row implications complete; original516 producer actual1 at unsaved synthetic expected-ID assertion, excluded from pass. No new actual-state audit;513 sealed473 inheritance remains.',audit_jobs_running=False,model_jobs_running=False,physical_jobs_running=False,latest_active_run=None,latest_active_snapshot=None,
  latest_round_new_exact_row_quotient_systems=4,latest_round_new_exact_implication_row_checks=scope['total_real_row_checks'],latest_round_new_exact_implication_coefficient_products=scope['exact_coefficient_cross_products'],latest_round_primary_proved_omitted_rows=200007,latest_round_witness_proved_omitted_rows=218962,latest_round_new_arithmetic_fixture_attempts=1,latest_round_failed_arithmetic_fixture_assertions=1,latest_round_failed_preparation_executions=1,latest_round_saved_only_metadata_completions=1,latest_round_new_root_fixture_scope='ZERO model/root/AD/geometry/work/map/LP/physics. Four real saved-system quotients proved; synthetic expected-ID assertion failed, unsaved output unavailable and not rerun.',
  new_solver_exists=False,new_solver_scope='No518 executable bounded algorithm or candidate.512 COMPLETE/read-only;516/517 exact implication preparation only, with preserved producer failure and successful independent REAL-system proof.',new_executable_whole_path_wrapper_exists=False,new_constructed_model_path_exists=False,new_candidate_exists=False,latest_preparation_scope='Exact positive-ray row quotient in both490/504 systems, union representatives;452522primary/618101witness rows. All omitted real rows exactly implied; full future original/transformed numerical checks mandatory. Failed synthetic fixture excluded.',latest_new_path_scope='No changed actualX/Y/J, canonical auxiliary or drops; no new LP, map or actual path. Storage reduction is not solver performance or feasibility.',main_training_started=False,ppo_started=False,fresh_holdout_started=False,holdout_started=False,policy_promoted=False,task_complete=False,repair_complete=False,automation_status='ACTIVE',notes=notes)
 (D/'status.json').write_text(json.dumps(s,indent=2));doc=R/'docs/LOW_SEAT_CONTACT_STATUS.md';doc.write_text(notes+'\n\nEarlier preserved entries follow.\n\n'+doc.read_text())
 record=dict(utc=now,scope=__doc__,preparation_only=True,new_closure=False,last_sealed_closure=str(D/'closure_record_515.json'),prior_checks=checks,protected_files=len(mf),manifest=str(manifest),manifest_sha256=sha(manifest),workers=workers,actual_exits=exits,preparation=prep,independent_audit=audit,local_verification=review,saved_scope=scope,original_producer_actual_exit=1,real_system_implications_passed=True,synthetic_fixture_passed=False,synthetic_fixture_output_available=False,synthetic_fixture_repeated=False,completed_groupings_repeated=False,new_solver_exists=False,new_complete_path=False,model_calls=0,root_calls=0,lp_calls=0,finite_maps=0,physics_ticks=0,policy_updates=0,protected_clean=64,unresolved_cases=8,repair_complete=False,task_complete=False,automation_status='ACTIVE')
 (D/'preparation_handoff_record_517.json').write_text(json.dumps(record,indent=2))
 handoff=[D/'status.json',R/'docs/LOW_SEAT_CONTACT_PROTOCOL.md',doc,D/'handoff_prepared_517.txt',manifest,D/'preparation_handoff_record_517.json',D/folders[0]/'preparation.json',D/folders[1]/'independent_audit.json',D/'exact_row_quotient_scope_517.json',D/'exact_row_quotient_local_verification_517.json']
 (D/'final_preparation_manifest_517.json').write_text(json.dumps({str(p):sha(p) for p in handoff},indent=2))
 print(json.dumps({k:record[k] for k in ['utc','prior_checks','protected_files','manifest','manifest_sha256','workers','original_producer_actual_exit','real_system_implications_passed','synthetic_fixture_passed','new_closure','task_complete','automation_status']},indent=2),flush=True)
if __name__=='__main__':main()
