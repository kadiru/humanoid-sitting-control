"""Preserve negative536/537 preparation; no reduction, LP or actual path, last sealed535 unchanged."""
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
 assert not (D/'preparation_handoff_record_537.json').exists();checks=[]
 for name in ['prior_hashes.json','completed_probes_preservation_535.json','supplemental_preservation_535.json']:
  m=read(D/name)
  for p,h in m.items():assert sha(p)==h,p
  checks.append(dict(manifest=name,files=len(m),passed=True))
 for name in ['command_forest_launch_536.json','command_forest_audit_launch_537.json']:
  for p,h in read(D/name)['source_input_sha256'].items():assert sha(p)==h,p
 exits={}
 for name in ['command_forest_compile_exit_536.json','command_forest_freeze_exit_536.json','command_forest_preparation_exit_536.json','command_forest_audit_compile_exit_537.json','command_forest_audit_exit_537.json','command_forest_index_exit_537.json']:
  exits[name]=read(D/name);assert exits[name]['actual_exit']==0
 folders=['command_forest_preparation_536','command_forest_audit_537']
 for name in folders:
  for p,h in read(D/name/'SHA256.json').items():assert sha(D/name/p)==h,p
 prep=read(D/folders[0]/'preparation.json');audit=read(D/folders[1]/'independent_audit.json');review=read(D/'command_forest_local_verification_537.json');scope=read(D/'command_forest_scope_537.json')
 assert prep['passed'] and audit['passed'] and review['all_hashes_verified']
 assert scope['new_proved_implications']==scope['new_removed_rows']==scope['lp_calls']==0 and scope['new_negative_path_decisions']==19840 and not scope['successful_reduction']
 workers=[]
 for p in Path('/proc').glob('[0-9]*/cmdline'):
  try:a=p.read_bytes().decode().split('\0')
  except (OSError,UnicodeError):continue
  if a and 'python' in a[0] and any((x.startswith('scripts/') or x.startswith(str(R/'scripts')+'/')) and Path(x).name!=Path(__file__).name for x in a[1:]):workers.append(a)
 assert not workers,workers
 for p,n in [(D/'status.json','status_before_command_forest_537.json'),(R/'docs/LOW_SEAT_CONTACT_STATUS.md','handoff_before_command_forest_537.md')]:
  assert not (D/n).exists();shutil.copy2(p,D/n)
 mf=read(D/'completed_probes_preservation_535.json');mf.update(read(D/'supplemental_preservation_535.json'))
 for name in folders:
  for p in (D/name).rglob('*'):
   if p.is_file():mf[str(p)]=sha(p)
 for pattern in ['*command_forest*536*.py','*command_forest*537*.py','record_low_seat_command_preparation_537.py']:
  for p in (R/'scripts').glob(pattern):mf[str(p)]=sha(p)
 for pattern in ['command_forest_*.json','command_forest_*.log']:
  for p in D.glob(pattern):
   if p.is_file() and p.name!='command_forest_preservation_537.log':mf[str(p)]=sha(p)
 for n in ['handoff_prepared_537.txt','status_before_command_forest_537.json','handoff_before_command_forest_537.md']:mf[str(D/n)]=sha(D/n)
 manifest=D/'command_forest_evidence_preservation_537.json';manifest.write_text(json.dumps(mf,indent=2))
 for p,h in mf.items():assert sha(p)==h,p
 now=datetime.datetime.now(datetime.timezone.utc).isoformat();notes=(D/'handoff_prepared_537.txt').read_text();s=read(D/'status.json')
 for k in list(s):
  if k.startswith('latest_round_'):
   if isinstance(s[k],bool):s[k]=False
   elif isinstance(s[k],(int,float)):s[k]=0
   else:assert k=='latest_round_new_root_fixture_scope',k
 s.update(utc=now,updated_utc=now,phase='offline_command_forest536_audit537_complete_no_reduction',latest_handoff=str(D/'handoff_prepared_537.txt'),latest_preparation=str(D/folders[0]),latest_preparation_audit=str(D/folders[1]/'independent_audit.json'),latest_preparation_preservation=str(manifest),latest_preservation=str(manifest),completed_evidence_manifest=str(manifest),completed_evidence_count=len(mf),latest_preparation_handoff_record=str(D/'preparation_handoff_record_537.json'),latest_preparation_handoff=str(D/'handoff_prepared_537.txt'),last_sealed_closure=str(D/'closure_record_535.json'),latest_closure=str(D/'closure_record_535.json'),latest_attempt=str(D/'schur_path_532'),latest_complete_construction=str(D/'interior_path_484'),latest_audit_status='537 independently checked all19840 negative exact forest decisions,68672 path-cap decisions, unchanged full source copies. No row reduction or new actual-state audit.533 sealed473 inheritance remains.',audit_jobs_running=False,model_jobs_running=False,physical_jobs_running=False,latest_active_run=None,latest_active_snapshot=None,
 latest_round_new_command_graph_systems=2,latest_round_new_exact_command_path_decisions=19840,latest_round_new_negative_command_path_decisions=19840,latest_round_command_path_cap_decisions=68672,latest_round_new_proved_command_implications=0,latest_round_new_removed_rows=0,latest_round_new_root_fixture_scope='ZERO LP/model/root/AD/geometry/finite-work/Gram/factor/finite-map/physics.19840 NEW exact multirow command-path implication attempts all failed the selected endpoint certificate;68672 paths retained untested under128-edge cap. No old scientific calculation repeated.',
 new_solver_exists=False,new_solver_scope='No new538 algorithm/wrapper/budget/candidate. Completed532 solver remains read-only;536/537 produced NO reduction and does not justify rerunning532/524 on identical matrices.',new_executable_whole_path_wrapper_exists=False,new_constructed_model_path_exists=False,new_candidate_exists=False,latest_preparation_scope='Distinct exact command-forest multirow implication hypothesis tested on two saved522 projected numeric systems; no rows proved redundant. All matrices/costs/bounds/equalities remain unchanged.',latest_new_path_scope='ActualX/Y/J/canonical auxiliary/drops unchanged. No LP solve, model state, physical motion or effort result. Failed chosen implications do not prove global nonredundancy/infeasibility.',main_training_started=False,ppo_started=False,fresh_holdout_started=False,holdout_started=False,policy_promoted=False,task_complete=False,repair_complete=False,automation_status='ACTIVE',notes=notes)
 assert s['latest_round_new_integer_normal_gap_incidence_coordinates']==0
 (D/'status.json').write_text(json.dumps(s,indent=2));doc=R/'docs/LOW_SEAT_CONTACT_STATUS.md';doc.write_text(notes+'\n\nEarlier preserved entries follow.\n\n'+doc.read_text())
 record=dict(utc=now,scope=__doc__,preparation_only=True,new_closure=False,last_sealed_closure=str(D/'closure_record_535.json'),prior_checks=checks,protected_files=len(mf),manifest=str(manifest),manifest_sha256=sha(manifest),workers=workers,actual_exits=exits,preparation=prep,independent_audit=audit,local_verification=review,saved_scope=scope,old_scientific_calculations_repeated=False,execution_recovery=False,unexecuted_original_audit_preserved=True,new_reduction=False,new_solver_exists=False,new_complete_path=False,model_calls=0,root_calls=0,lp_calls=0,finite_maps=0,physics_ticks=0,policy_updates=0,protected_clean=64,unresolved_cases=8,repair_complete=False,task_complete=False,automation_status='ACTIVE')
 (D/'preparation_handoff_record_537.json').write_text(json.dumps(record,indent=2))
 handoff=[D/'status.json',R/'docs/LOW_SEAT_CONTACT_PROTOCOL.md',doc,D/'handoff_prepared_537.txt',manifest,D/'preparation_handoff_record_537.json',D/folders[0]/'preparation.json',D/folders[1]/'independent_audit.json',D/'command_forest_scope_537.json',D/'command_forest_local_verification_537.json']
 (D/'final_preparation_manifest_537.json').write_text(json.dumps({str(p):sha(p) for p in handoff},indent=2))
 print(json.dumps({k:record[k] for k in ['utc','prior_checks','protected_files','manifest','manifest_sha256','workers','new_closure','new_reduction','task_complete','automation_status']},indent=2),flush=True)
if __name__=='__main__':main()
