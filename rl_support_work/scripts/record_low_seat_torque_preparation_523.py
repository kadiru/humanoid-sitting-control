"""Preserve completed522/523 preparation; last SEALED closure521 remains unchanged."""
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
 assert not (D/'preparation_handoff_record_523.json').exists();checks=[]
 for name in ['prior_hashes.json','completed_probes_preservation_521.json','supplemental_preservation_521.json']:
  m=read(D/name)
  for p,h in m.items():assert sha(p)==h,p
  checks.append(dict(manifest=name,files=len(m),passed=True))
 for p,h in read(D/'torque_projection_launch_522.json')['source_input_sha256'].items():assert sha(p)==h,p
 exits={}
 for name in ['torque_projection_compile_exit_522.json','torque_projection_freeze_exit_522.json','torque_projection_preparation_exit_522.json','torque_projection_audit_exit_523.json']:
  exits[name]=read(D/name);assert exits[name]['actual_exit']==0
 folders=['torque_projection_preparation_522','torque_projection_audit_523']
 for name in folders:
  for p,h in read(D/name/'SHA256.json').items():assert sha(D/name/p)==h,p
 prep=read(D/folders[0]/'preparation.json');audit=read(D/folders[1]/'independent_audit.json');review=read(D/'torque_projection_local_verification_523.json');scope=read(D/'torque_projection_scope_523.json')
 assert prep['passed'] and audit['passed'] and review['all_hashes_verified'];assert audit['remaining_direction_coordinates']==56950 and not audit['prior517_proof_repeated']
 workers=[]
 for p in Path('/proc').glob('[0-9]*/cmdline'):
  try:a=p.read_bytes().decode().split('\0')
  except (OSError,UnicodeError):continue
  if a and 'python' in a[0] and any((x.startswith('scripts/') or x.startswith(str(R/'scripts')+'/')) and Path(x).name!=Path(__file__).name for x in a[1:]):workers.append(a)
 assert not workers,workers
 for p,n in [(D/'status.json','status_before_torque_projection_523.json'),(R/'docs/LOW_SEAT_CONTACT_STATUS.md','handoff_before_torque_projection_523.md')]:
  assert not (D/n).exists();shutil.copy2(p,D/n)
 mf=read(D/'completed_probes_preservation_521.json');mf.update(read(D/'supplemental_preservation_521.json'))
 for name in folders:
  for p in (D/name).rglob('*'):
   if p.is_file():mf[str(p)]=sha(p)
 for pattern in ['*torque_projection*522*.py','*torque_projection*523*.py','*torque_preparation*523*.py']:
  for p in (R/'scripts').glob(pattern):mf[str(p)]=sha(p)
 for pattern in ['torque_projection_*.json','torque_projection_*.log']:
  for p in D.glob(pattern):
   if p.is_file() and p.name!='torque_projection_preservation_523.log':mf[str(p)]=sha(p)
 for n in ['handoff_prepared_523.txt','status_before_torque_projection_523.json','handoff_before_torque_projection_523.md']:mf[str(D/n)]=sha(D/n)
 manifest=D/'torque_projection_evidence_preservation_523.json';manifest.write_text(json.dumps(mf,indent=2))
 for p,h in mf.items():assert sha(p)==h,p
 now=datetime.datetime.now(datetime.timezone.utc).isoformat();notes=(D/'handoff_prepared_523.txt').read_text();s=read(D/'status.json')
 for k in list(s):
  if k.startswith('latest_round_'):
   if isinstance(s[k],bool):s[k]=False
   elif isinstance(s[k],(int,float)):s[k]=0
   else:assert k=='latest_round_new_root_fixture_scope',k
 s.update(utc=now,updated_utc=now,phase='offline_torque_projection522_audit523_complete_no_new_solver',latest_handoff=str(D/'handoff_prepared_523.txt'),latest_preparation=str(D/folders[0]),latest_preparation_audit=str(D/folders[1]/'independent_audit.json'),latest_preparation_preservation=str(manifest),latest_preservation=str(manifest),completed_evidence_manifest=str(manifest),completed_evidence_count=len(mf),latest_preparation_handoff_record=str(D/'preparation_handoff_record_523.json'),latest_preparation_handoff=str(D/'preparation_handoff_record_523.json'),last_sealed_closure=str(D/'closure_record_521.json'),latest_closure=str(D/'closure_record_521.json'),latest_attempt=str(D/'quotient_path_518'),latest_complete_construction=str(D/'interior_path_484'),latest_audit_status='523 independent exact existential-pair arithmetic complete; no new actual-state audit.519 sealed473 inheritance remains.',audit_jobs_running=False,model_jobs_running=False,physical_jobs_running=False,latest_active_run=None,latest_active_snapshot=None,
 latest_round_new_existential_projection_systems=2,latest_round_new_eliminated_auxiliary_torque_coordinates=21840,latest_round_new_exact_projection_pair_checks=scope['total_pair_checks'],latest_round_new_exact_projection_coefficients=scope['total_generated_coefficients'],latest_round_new_projection_arithmetic_examples=2,latest_round_new_root_fixture_scope='ZERO model/root/AD/geometry/finite-work/map/LP/physics. Two NEW saved-system existential projections and two small interval arithmetic examples only. Prior516/517 quotient/proof inherited, not repeated.',
 new_solver_exists=False,new_solver_scope='No524 executable algorithm/resource budget/actual candidate.518 COMPLETE;522/523 is exact torque-auxiliary elimination preparation only.',new_executable_whole_path_wrapper_exists=False,new_constructed_model_path_exists=False,new_candidate_exists=False,latest_preparation_scope='Eliminate21840 native auxiliary torque increments by exhaustive exact lower/upper pair projection,56950retained direction coordinates. Both consistency signs preserved existentially; source full witness checks mandatory. Binary64 exact lift not guaranteed.',latest_new_path_scope='No actualX/Y/J/canonical auxiliary/drop state changed. No LP or finite map. Primary2147925nonzeros; witness2697204nonzeros, larger than source witness. Counts do not establish speed or feasibility.',main_training_started=False,ppo_started=False,fresh_holdout_started=False,holdout_started=False,policy_promoted=False,task_complete=False,repair_complete=False,automation_status='ACTIVE',notes=notes)
 (D/'status.json').write_text(json.dumps(s,indent=2));doc=R/'docs/LOW_SEAT_CONTACT_STATUS.md';doc.write_text(notes+'\n\nEarlier preserved entries follow.\n\n'+doc.read_text())
 record=dict(utc=now,scope=__doc__,preparation_only=True,new_closure=False,last_sealed_closure=str(D/'closure_record_521.json'),prior_checks=checks,protected_files=len(mf),manifest=str(manifest),manifest_sha256=sha(manifest),workers=workers,actual_exits=exits,preparation=prep,independent_audit=audit,local_verification=review,saved_scope=scope,old_grouping_or_proof_repeated=False,new_solver_exists=False,new_complete_path=False,model_calls=0,root_calls=0,lp_calls=0,finite_maps=0,physics_ticks=0,policy_updates=0,protected_clean=64,unresolved_cases=8,repair_complete=False,task_complete=False,automation_status='ACTIVE')
 (D/'preparation_handoff_record_523.json').write_text(json.dumps(record,indent=2))
 handoff=[D/'status.json',R/'docs/LOW_SEAT_CONTACT_PROTOCOL.md',doc,D/'handoff_prepared_523.txt',manifest,D/'preparation_handoff_record_523.json',D/folders[0]/'preparation.json',D/folders[1]/'independent_audit.json',D/'torque_projection_scope_523.json',D/'torque_projection_local_verification_523.json']
 (D/'final_preparation_manifest_523.json').write_text(json.dumps({str(p):sha(p) for p in handoff},indent=2))
 print(json.dumps({k:record[k] for k in ['utc','prior_checks','protected_files','manifest','manifest_sha256','workers','new_closure','task_complete','automation_status']},indent=2),flush=True)
if __name__=='__main__':main()
