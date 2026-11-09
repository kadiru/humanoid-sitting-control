"""Preserve580/581 numerical input preparation; closure577 and actual path unchanged."""
from pathlib import Path
import json,hashlib,datetime,shutil
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');R=Path('/home/kadir/humanoid-sitting');read=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def main():
 assert not (D/'preparation_handoff_record_581.json').exists();checks=[]
 for name in ['prior_hashes.json','affine_output_evidence_preservation_579.json','affine_output_supplemental_579.json']:
  m=read(D/name)
  for p,h in m.items():assert sha(p)==h,p
  checks.append(dict(manifest=name,files=len(m),passed=True))
 for p,h in read(D/'graph_encoding_launch_580.json')['source_input_sha256'].items():assert sha(p)==h,p
 exits={}
 for n in ['graph_encoding_compile_exit_580.json','graph_encoding_freeze_exit_580.json','graph_encoding_preparation_exit_580.json','graph_encoding_audit_exit_581.json','graph_encoding_index_exit_581.json','graph_encoding_preservation_compile_exit_581.json']:
  exits[n]=read(D/n);assert exits[n]['actual_exit']==0
 folders=['graph_encoding_preparation_580','graph_encoding_audit_581']
 for folder in folders:
  for p,h in read(D/folder/'SHA256.json').items():assert sha(D/folder/p)==h,p
 prep=read(D/folders[0]/'preparation.json');audit=read(D/folders[1]/'independent_audit.json');review=read(D/'graph_encoding_local_verification_581.json');scope=read(D/'graph_encoding_scope_581.json');q=scope['primary']
 assert prep['passed'] and audit['passed'] and review['all_hashes_verified'] and scope['numerical_lp_calls']==0 and scope['primal_vectors_constructed']==0 and not scope['complete_numerical_algorithm_exists']
 assert q['columns']==31016 and q['rows']==265967 and q['real_encoded_set_inner_subset'] and q['native_outer_box_encodings']==31015
 workers=[]
 for p in Path('/proc').glob('[0-9]*/cmdline'):
  try:a=p.read_bytes().decode().split('\0')
  except (OSError,UnicodeError):continue
  if a and 'python' in a[0] and any((x.startswith('scripts/') or x.startswith(str(R/'scripts')+'/')) and Path(x).name!=Path(__file__).name for x in a[1:]):workers.append(a)
 assert not workers,workers
 notes=(D/'handoff_prepared_581.txt').read_text();s=read(D/'status.json')
 for k in list(s):
  if k.startswith('latest_round_'):
   if isinstance(s[k],bool):s[k]=False
   elif isinstance(s[k],(int,float)):s[k]=0
   else:assert k=='latest_round_new_root_fixture_scope',k
 for p,name in [(D/'status.json','status_before_graph_encoding_581.json'),(R/'docs/LOW_SEAT_CONTACT_STATUS.md','handoff_before_graph_encoding_581.md')]:
  assert not (D/name).exists();shutil.copy2(p,D/name)
 mf=read(D/'affine_output_evidence_preservation_579.json');mf.update(read(D/'affine_output_supplemental_579.json'))
 for folder in folders:
  for p in (D/folder).rglob('*'):
   if p.is_file():mf[str(p)]=sha(p)
 for pattern in ['*graph_encoding*580*.py','*graph_encoding*581*.py']:
  for p in (R/'scripts').glob(pattern):mf[str(p)]=sha(p)
 for pattern in ['graph_encoding_*.json','graph_encoding_*.log']:
  for p in D.glob(pattern):
   if p.is_file() and p.name!='graph_encoding_preservation_581.log':mf[str(p)]=sha(p)
 for name in ['handoff_prepared_581.txt','status_before_graph_encoding_581.json','handoff_before_graph_encoding_581.md']:mf[str(D/name)]=sha(D/name)
 manifest=D/'graph_encoding_evidence_preservation_581.json';assert not manifest.exists();manifest.write_text(json.dumps(mf,indent=2))
 for p,h in mf.items():assert sha(p)==h,p
 now=datetime.datetime.now(datetime.timezone.utc).isoformat()
 s.update(utc=now,updated_utc=now,phase='offline_graph_encoding580_audit581_preparation_complete',latest_handoff=str(D/'handoff_prepared_581.txt'),latest_preparation=str(D/folders[0]),latest_preparation_audit=str(D/folders[1]/'independent_audit.json'),latest_preparation_preservation=str(manifest),latest_preservation=str(manifest),completed_evidence_manifest=str(manifest),completed_evidence_count=len(mf),latest_preparation_handoff_record=str(D/'preparation_handoff_record_581.json'),latest_preparation_handoff=str(D/'handoff_prepared_581.txt'),last_sealed_closure=str(D/'closure_record_577.json'),latest_closure=str(D/'closure_record_577.json'),latest_attempt=str(D/'zero_face_path_574'),latest_complete_construction=str(D/'interior_path_484'),latest_audit_status='581 independent coefficient nearest-even cells,integer error support,compensated RHS floor and outerbox/copy audit COMPLETE. No primalwitness/graphpointlift/actualstateaudit.575 sealed473 geometry/work inheritance remains.',audit_jobs_running=False,model_jobs_running=False,physical_jobs_running=False,latest_active_run=None,latest_active_snapshot=None,
 latest_round_new_coefficient_encodings=q['nearest_coefficient_encodings'],latest_round_new_error_support_products=q['exact_error_support_products'],latest_round_new_RHS_floor_encodings=q['compensated_RHS_floor_encodings'],latest_round_new_outer_box_encodings=q['native_outer_box_encodings'],latest_round_new_root_fixture_scope='NEW compensated binary64 representation of578 exact rational graph zero-face. Explicit OUTER power2 coordinate box encloses existing546 exact endpoints; exact coefficient errors supported overbox and RHS roundedDOWN. Sufficient REAL INNER SUBSET,not equivalence. No endpoint propagation/oldproof replay/LP/linear-solve/primalgraphlift/analyticdualconstruction/model/root/AD/geometry/finiteencoding/physics. Feasibility and performance unknown.',
 new_solver_exists=False,new_solver_scope='580 rounded input plus581 independent representation audit only. No582 completealgorithm/child/wholepathwrapper/budget/sourcefreeze/return-and-actualstateauditor/primalwitness.574 remains completed read-only.',new_executable_whole_path_wrapper_exists=False,new_constructed_model_path_exists=False,new_candidate_exists=False,latest_preparation_scope=scope['scope'],latest_new_path_scope='Actual472 INITIAL X/Y/J plus490canonicalaux/exactdrops unchanged. No primal numerical witness,motion or effort result.',main_training_started=False,ppo_started=False,fresh_holdout_started=False,holdout_started=False,policy_promoted=False,task_complete=False,repair_complete=False,automation_status='ACTIVE',notes=notes)
 (D/'status.json').write_text(json.dumps(s,indent=2));doc=R/'docs/LOW_SEAT_CONTACT_STATUS.md';doc.write_text(notes+'\n\nEarlier preserved entries follow.\n\n'+doc.read_text())
 record=dict(utc=now,scope=__doc__,preparation_only=True,new_closure=False,last_sealed_closure=str(D/'closure_record_577.json'),prior_checks=checks,protected_files=len(mf),manifest=str(manifest),manifest_sha256=sha(manifest),workers=workers,actual_exits=exits,preparation=prep,independent_audit=audit,local_verification=review,saved_scope=scope,completed_scientific_calculations_repeated=False,scientific_execution_recovery=False,new_complete_numerical_algorithm=False,new_solver_exists=False,new_complete_path=False,numerical_lp_calls=0,linear_system_solves=0,model_calls=0,root_calls=0,finite_maps=0,physics_ticks=0,policy_updates=0,primal_vectors_constructed=0,analytic_dual_vectors_constructed=0,graph_point_lifts=0,rounded_numeric_input_exists=True,zero_face_feasibility_known=False,original_optimum_zero_established=False,protected_clean=64,unresolved_cases=8,repair_complete=False,task_complete=False,automation_status='ACTIVE')
 (D/'preparation_handoff_record_581.json').write_text(json.dumps(record,indent=2))
 handoff=[D/'status.json',R/'docs/LOW_SEAT_CONTACT_PROTOCOL.md',doc,D/'handoff_prepared_581.txt',manifest,D/'preparation_handoff_record_581.json',D/folders[0]/'preparation.json',D/folders[1]/'independent_audit.json',D/'graph_encoding_scope_581.json',D/'graph_encoding_local_verification_581.json']
 (D/'final_preparation_manifest_581.json').write_text(json.dumps({str(p):sha(p) for p in handoff},indent=2));print(json.dumps({k:record[k] for k in ['utc','prior_checks','protected_files','manifest','manifest_sha256','workers','new_closure','task_complete','automation_status']},indent=2),flush=True)
if __name__=='__main__':main()
