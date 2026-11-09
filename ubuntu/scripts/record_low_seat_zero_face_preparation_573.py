"""Preserve572/573 exact zero-face preparation; sealed569 and actual path unchanged."""
from pathlib import Path
import json,hashlib,datetime,shutil
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');R=Path('/home/kadir/humanoid-sitting');read=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def main():
 assert not (D/'preparation_handoff_record_573.json').exists();checks=[]
 for name in ['prior_hashes.json','coupled_support_evidence_preservation_571.json','coupled_support_supplemental_571.json']:
  m=read(D/name)
  for p,h in m.items():assert sha(p)==h,p
  checks.append(dict(manifest=name,files=len(m),passed=True))
 for p,h in read(D/'zero_face_launch_572.json')['source_input_sha256'].items():assert sha(p)==h,p
 exits={}
 for n in ['zero_face_compile_exit_572.json','zero_face_freeze_exit_572.json','zero_face_preparation_exit_572.json','zero_face_audit_exit_573.json','zero_face_index_exit_573.json','zero_face_preservation_compile_exit_573.json']:
  exits[n]=read(D/n);assert exits[n]['actual_exit']==0
 folders=['zero_face_preparation_572','zero_face_audit_573']
 for folder in folders:
  for p,h in read(D/folder/'SHA256.json').items():assert sha(D/folder/p)==h,p
 prep=read(D/folders[0]/'preparation.json');audit=read(D/folders[1]/'independent_audit.json');review=read(D/'zero_face_local_verification_573.json');scope=read(D/'zero_face_scope_573.json')
 assert prep['passed'] and audit['passed'] and review['all_hashes_verified'] and scope['numerical_lp_calls']==0 and scope['analytic_lower_bound_dual_vectors']==1
 workers=[]
 for p in Path('/proc').glob('[0-9]*/cmdline'):
  try:a=p.read_bytes().decode().split('\0')
  except (OSError,UnicodeError):continue
  if a and 'python' in a[0] and any((x.startswith('scripts/') or x.startswith(str(R/'scripts')+'/')) and Path(x).name!=Path(__file__).name for x in a[1:]):workers.append(a)
 assert not workers,workers
 notes=(D/'handoff_prepared_573.txt').read_text();s=read(D/'status.json')
 for k in list(s):
  if k.startswith('latest_round_'):
   if isinstance(s[k],bool):s[k]=False
   elif isinstance(s[k],(int,float)):s[k]=0
   else:assert k=='latest_round_new_root_fixture_scope',k
 for p,name in [(D/'status.json','status_before_zero_face_573.json'),(R/'docs/LOW_SEAT_CONTACT_STATUS.md','handoff_before_zero_face_573.md')]:
  assert not (D/name).exists();shutil.copy2(p,D/name)
 mf=read(D/'coupled_support_evidence_preservation_571.json');mf.update(read(D/'coupled_support_supplemental_571.json'))
 for folder in folders:
  for p in (D/folder).rglob('*'):
   if p.is_file():mf[str(p)]=sha(p)
 for pattern in ['*zero_face*572*.py','*zero_face*573*.py']:
  for p in (R/'scripts').glob(pattern):mf[str(p)]=sha(p)
 for pattern in ['zero_face_*.json','zero_face_*.log']:
  for p in D.glob(pattern):
   if p.is_file() and p.name!='zero_face_preservation_573.log':mf[str(p)]=sha(p)
 for name in ['handoff_prepared_573.txt','status_before_zero_face_573.json','handoff_before_zero_face_573.md']:mf[str(D/name)]=sha(D/name)
 manifest=D/'zero_face_evidence_preservation_573.json';assert not manifest.exists();manifest.write_text(json.dumps(mf,indent=2))
 for p,h in mf.items():assert sha(p)==h,p
 now=datetime.datetime.now(datetime.timezone.utc).isoformat();q=scope['primary']
 s.update(utc=now,updated_utc=now,phase='offline_zero_face572_audit573_preparation_complete',latest_handoff=str(D/'handoff_prepared_573.txt'),latest_preparation=str(D/folders[0]),latest_preparation_audit=str(D/folders[1]/'independent_audit.json'),latest_preparation_preservation=str(manifest),latest_preservation=str(manifest),completed_evidence_manifest=str(manifest),completed_evidence_count=len(mf),latest_preparation_handoff_record=str(D/'preparation_handoff_record_573.json'),latest_preparation_handoff=str(D/'handoff_prepared_573.txt'),last_sealed_closure=str(D/'closure_record_569.json'),latest_closure=str(D/'closure_record_569.json'),latest_attempt=str(D/'dual_path_566'),latest_complete_construction=str(D/'interior_path_484'),latest_audit_status='573 independent exact zero-face scalar/RHS/copy identities and one analytic dual lower-bound certificate complete. No primal witness or actual-state audit;567 sealed473 geometry/work inheritance remains.',audit_jobs_running=False,model_jobs_running=False,physical_jobs_running=False,latest_active_run=None,latest_active_snapshot=None,
 latest_round_new_zero_face_scalar_checks=q['new_scalar_coefficient_checks'],latest_round_new_zero_face_RHS_checks=q['new_opposite_RHS_checks'],latest_round_new_zero_face_equality_coefficients=q['new_native_equality_coefficients_copied'],latest_round_new_analytic_lower_bound_dual_vectors=1,latest_round_new_analytic_dual_coordinate_identities=scope['new_exact_analytic_dual_coordinate_identities'],latest_round_new_root_fixture_scope='NEW exact primary objective-zero face and one analytic dual-feasible lower-bound vector. ZERO primal vector/LP/linear solve/oldproof/fixture/model/root/AD/geometry/finite encoding/physics. The face may be empty; original optimum-zero and physical feasibility are unknown.',
 new_solver_exists=False,new_solver_scope='572 sufficient zero-face formulation only; no574 complete numerical algorithm/child/full-path wrapper/sourcefreeze/primal witness.566 remains completed read-only.',new_executable_whole_path_wrapper_exists=False,new_constructed_model_path_exists=False,new_candidate_exists=False,latest_preparation_scope='All78791 original coordinates/cost/bounds retained, but added t=0 restricts to a possibly empty proper face.68250 mechanics pairs become affine equalities. One analytic lower-bound dual proves only objective>=0. Whole original feasible-set equivalence, primal feasibility and original optimum-zero are NOT claimed.',latest_new_path_scope='Actual472 INITIAL X/Y/J plus490 canonical auxiliary/exact drops unchanged. No primal numerical witness, motion or effort result.',main_training_started=False,ppo_started=False,fresh_holdout_started=False,holdout_started=False,policy_promoted=False,task_complete=False,repair_complete=False,automation_status='ACTIVE',notes=notes)
 (D/'status.json').write_text(json.dumps(s,indent=2));doc=R/'docs/LOW_SEAT_CONTACT_STATUS.md';doc.write_text(notes+'\n\nEarlier preserved entries follow.\n\n'+doc.read_text())
 record=dict(utc=now,scope=__doc__,preparation_only=True,new_closure=False,last_sealed_closure=str(D/'closure_record_569.json'),prior_checks=checks,protected_files=len(mf),manifest=str(manifest),manifest_sha256=sha(manifest),workers=workers,actual_exits=exits,preparation=prep,independent_audit=audit,local_verification=review,saved_scope=scope,completed_scientific_calculations_repeated=False,scientific_execution_recovery=False,new_complete_numerical_algorithm=False,new_solver_exists=False,new_complete_path=False,numerical_lp_calls=0,model_calls=0,root_calls=0,finite_maps=0,physics_ticks=0,policy_updates=0,primal_vectors_constructed=0,analytic_lower_bound_dual_vectors=1,zero_face_feasibility_known=False,original_optimum_zero_established=False,protected_clean=64,unresolved_cases=8,repair_complete=False,task_complete=False,automation_status='ACTIVE')
 (D/'preparation_handoff_record_573.json').write_text(json.dumps(record,indent=2))
 handoff=[D/'status.json',R/'docs/LOW_SEAT_CONTACT_PROTOCOL.md',doc,D/'handoff_prepared_573.txt',manifest,D/'preparation_handoff_record_573.json',D/folders[0]/'preparation.json',D/folders[1]/'independent_audit.json',D/'zero_face_scope_573.json',D/'zero_face_local_verification_573.json']
 (D/'final_preparation_manifest_573.json').write_text(json.dumps({str(p):sha(p) for p in handoff},indent=2))
 print(json.dumps({k:record[k] for k in ['utc','prior_checks','protected_files','manifest','manifest_sha256','workers','new_closure','task_complete','automation_status']},indent=2),flush=True)
if __name__=='__main__':main()
