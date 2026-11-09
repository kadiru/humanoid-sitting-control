"""Preserve578/579 exact rational graph elimination; closure577 and actual path unchanged."""
from pathlib import Path
import json,hashlib,datetime,shutil
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');R=Path('/home/kadir/humanoid-sitting');read=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def main():
 assert not (D/'preparation_handoff_record_579.json').exists();checks=[]
 for name in ['prior_hashes.json','completed_probes_preservation_577.json','supplemental_preservation_577.json']:
  m=read(D/name)
  for p,h in m.items():assert sha(p)==h,p
  checks.append(dict(manifest=name,files=len(m),passed=True))
 for p,h in read(D/'affine_output_launch_578.json')['source_input_sha256'].items():assert sha(p)==h,p
 exits={}
 for n in ['affine_output_compile_exit_578.json','affine_output_freeze_exit_578.json','affine_output_preparation_exit_578.json','affine_output_audit_exit_579.json','affine_output_index_exit_579.json','affine_output_preservation_compile_exit_579.json']:
  exits[n]=read(D/n);assert exits[n]['actual_exit']==0
 folders=['affine_output_preparation_578','affine_output_audit_579']
 for folder in folders:
  for p,h in read(D/folder/'SHA256.json').items():assert sha(D/folder/p)==h,p
 prep=read(D/folders[0]/'preparation.json');audit=read(D/folders[1]/'independent_audit.json');review=read(D/'affine_output_local_verification_579.json');scope=read(D/'affine_output_scope_579.json');q=scope['primary']
 assert prep['passed'] and audit['passed'] and review['all_hashes_verified'] and scope['numerical_lp_calls']==0 and scope['rounded_coefficients_created']==0 and scope['primal_vectors_constructed']==0
 assert q['eliminated_auxiliary_columns']==47775 and q['reduced_columns']==31016 and q['rows_discarded']==0
 workers=[]
 for p in Path('/proc').glob('[0-9]*/cmdline'):
  try:a=p.read_bytes().decode().split('\0')
  except (OSError,UnicodeError):continue
  if a and 'python' in a[0] and any((x.startswith('scripts/') or x.startswith(str(R/'scripts')+'/')) and Path(x).name!=Path(__file__).name for x in a[1:]):workers.append(a)
 assert not workers,workers
 notes=(D/'handoff_prepared_579.txt').read_text();s=read(D/'status.json')
 for k in list(s):
  if k.startswith('latest_round_'):
   if isinstance(s[k],bool):s[k]=False
   elif isinstance(s[k],(int,float)):s[k]=0
   else:assert k=='latest_round_new_root_fixture_scope',k
 for p,name in [(D/'status.json','status_before_affine_output_579.json'),(R/'docs/LOW_SEAT_CONTACT_STATUS.md','handoff_before_affine_output_579.md')]:
  assert not (D/name).exists();shutil.copy2(p,D/name)
 mf=read(D/'completed_probes_preservation_577.json');mf.update(read(D/'supplemental_preservation_577.json'))
 for folder in folders:
  for p in (D/folder).rglob('*'):
   if p.is_file():mf[str(p)]=sha(p)
 for pattern in ['*affine_output*578*.py','*affine_output*579*.py']:
  for p in (R/'scripts').glob(pattern):mf[str(p)]=sha(p)
 for pattern in ['affine_output_*.json','affine_output_*.log']:
  for p in D.glob(pattern):
   if p.is_file() and p.name!='affine_output_preservation_579.log':mf[str(p)]=sha(p)
 for name in ['handoff_prepared_579.txt','status_before_affine_output_579.json','handoff_before_affine_output_579.md']:mf[str(D/name)]=sha(D/name)
 manifest=D/'affine_output_evidence_preservation_579.json';assert not manifest.exists();manifest.write_text(json.dumps(mf,indent=2))
 for p,h in mf.items():assert sha(p)==h,p
 now=datetime.datetime.now(datetime.timezone.utc).isoformat()
 s.update(utc=now,updated_utc=now,phase='offline_affine_output578_audit579_preparation_complete',latest_handoff=str(D/'handoff_prepared_579.txt'),latest_preparation=str(D/folders[0]),latest_preparation_audit=str(D/folders[1]/'independent_audit.json'),latest_preparation_preservation=str(manifest),latest_preservation=str(manifest),completed_evidence_manifest=str(manifest),completed_evidence_count=len(mf),latest_preparation_handoff_record=str(D/'preparation_handoff_record_579.json'),latest_preparation_handoff=str(D/'handoff_prepared_579.txt'),last_sealed_closure=str(D/'closure_record_577.json'),latest_closure=str(D/'closure_record_577.json'),latest_attempt=str(D/'zero_face_path_574'),latest_complete_construction=str(D/'interior_path_484'),latest_audit_status='579 independent exact graph/incidence/copy and reverse equality-row coefficient/RHS reconstruction complete. No primalwitness/numericalmatrix/actualstateaudit.575 sealed473 geometry/work inheritance remains.',audit_jobs_running=False,model_jobs_running=False,physical_jobs_running=False,latest_active_run=None,latest_active_snapshot=None,
 latest_round_new_exact_affine_output_pivots=47775,latest_round_new_exact_affine_output_coefficient_products=q['exact_coefficient_products'],latest_round_new_exact_affine_output_RHS_products=q['exact_RHS_products'],latest_round_new_exact_reverse_coefficient_products=scope['independent_coefficient_products'],latest_round_new_exact_reverse_RHS_products=scope['independent_RHS_products'],latest_round_new_root_fixture_scope='NEW exact rational elimination of47775 auxiliary command/torque/COM outputs on572 zero face,31016retainedvariables inclALLphysicalX/scalar. No inverse/factor/15root/roundednumericmatrix/LP/linear-systemsolve/primalpoint/analyticdualproof/model/root/AD/geometry/finiteencoding/physics. Proper/emptyface andfinite-extension limits remain; originaloptimumzero/feasibility unknown.',
 new_solver_exists=False,new_solver_scope='578 exact rational graph reduction only. No580 rounded numerical input/completealgorithm/child/wholepathwrapper/budget/sourcefreeze/primalwitness.574 remains completed read-only.',new_executable_whole_path_wrapper_exists=False,new_constructed_model_path_exists=False,new_candidate_exists=False,latest_preparation_scope='Every original572 inequality retained under exact rational equality substitution;47775 auxiliaryoutputs/equalities eliminated,allphysicalX retained. ExactREALprojection/objective equivalence to572ZERO FACE only,notwholeoriginalLP. No universalbinary64lift/numericalequivalence/feasibility/speedclaim.',latest_new_path_scope='Actual472 INITIAL X/Y/J plus490canonicalaux/exactdrops unchanged. No primal numerical witness,motion or effort result.',main_training_started=False,ppo_started=False,fresh_holdout_started=False,holdout_started=False,policy_promoted=False,task_complete=False,repair_complete=False,automation_status='ACTIVE',notes=notes)
 (D/'status.json').write_text(json.dumps(s,indent=2));doc=R/'docs/LOW_SEAT_CONTACT_STATUS.md';doc.write_text(notes+'\n\nEarlier preserved entries follow.\n\n'+doc.read_text())
 record=dict(utc=now,scope=__doc__,preparation_only=True,new_closure=False,last_sealed_closure=str(D/'closure_record_577.json'),prior_checks=checks,protected_files=len(mf),manifest=str(manifest),manifest_sha256=sha(manifest),workers=workers,actual_exits=exits,preparation=prep,independent_audit=audit,local_verification=review,saved_scope=scope,completed_scientific_calculations_repeated=False,scientific_execution_recovery=False,new_complete_numerical_algorithm=False,new_solver_exists=False,new_complete_path=False,numerical_lp_calls=0,linear_system_solves=0,model_calls=0,root_calls=0,finite_maps=0,physics_ticks=0,policy_updates=0,primal_vectors_constructed=0,analytic_dual_vectors_constructed=0,rounded_coefficients_created=0,zero_face_feasibility_known=False,original_optimum_zero_established=False,protected_clean=64,unresolved_cases=8,repair_complete=False,task_complete=False,automation_status='ACTIVE')
 (D/'preparation_handoff_record_579.json').write_text(json.dumps(record,indent=2))
 handoff=[D/'status.json',R/'docs/LOW_SEAT_CONTACT_PROTOCOL.md',doc,D/'handoff_prepared_579.txt',manifest,D/'preparation_handoff_record_579.json',D/folders[0]/'preparation.json',D/folders[1]/'independent_audit.json',D/'affine_output_scope_579.json',D/'affine_output_local_verification_579.json']
 (D/'final_preparation_manifest_579.json').write_text(json.dumps({str(p):sha(p) for p in handoff},indent=2));print(json.dumps({k:record[k] for k in ['utc','prior_checks','protected_files','manifest','manifest_sha256','workers','new_closure','task_complete','automation_status']},indent=2),flush=True)
if __name__=='__main__':main()
