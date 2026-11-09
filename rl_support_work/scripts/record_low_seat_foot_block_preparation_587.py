"""Preserve586/587 exact rational foot-block dimension reduction; closure585 and actual path unchanged."""
from pathlib import Path
import json,hashlib,datetime,shutil
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');R=Path('/home/kadir/humanoid-sitting');read=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def main():
 assert not (D/'preparation_handoff_record_587.json').exists();checks=[]
 for name in ['prior_hashes.json','completed_probes_preservation_585.json','supplemental_preservation_585.json']:
  m=read(D/name)
  for p,h in m.items():assert sha(p)==h,p
  checks.append(dict(manifest=name,files=len(m),passed=True))
 for p,h in read(D/'foot_block_launch_586.json')['source_input_sha256'].items():assert sha(p)==h,p
 exits={}
 for n in ['foot_block_compile_exit_586.json','foot_block_freeze_exit_586.json','foot_block_preparation_exit_586.json','foot_block_audit_exit_587.json','foot_block_index_exit_587.json','foot_block_preservation_compile_exit_587.json']:
  exits[n]=read(D/n);assert exits[n]['actual_exit']==0
 folders=['foot_block_preparation_586','foot_block_audit_587']
 for folder in folders:
  for p,h in read(D/folder/'SHA256.json').items():assert sha(D/folder/p)==h,p
 prep=read(D/folders[0]/'preparation.json');audit=read(D/folders[1]/'independent_audit.json');review=read(D/'foot_block_local_verification_587.json');scope=read(D/'foot_block_scope_587.json');q=scope['primary']
 assert prep['passed'] and audit['passed'] and review['all_hashes_verified'] and scope['numerical_lp_calls']==0 and scope['primal_vectors_constructed']==0 and not scope['complete_numerical_algorithm_exists']
 assert q['reduced_columns']==14636 and q['eliminated_leg_columns']==16380 and q['exact_six_by_six_inversions']==2730 and q['statistics']['inequality']['rows']==265967 and q['statistics']['equality']['rows']==5462
 workers=[]
 for p in Path('/proc').glob('[0-9]*/cmdline'):
  try:a=p.read_bytes().decode().split('\0')
  except (OSError,UnicodeError):continue
  if a and 'python' in a[0] and any((x.startswith('scripts/') or x.startswith(str(R/'scripts')+'/')) and Path(x).name!=Path(__file__).name for x in a[1:]):workers.append(a)
 assert not workers,workers
 notes=(D/'handoff_prepared_587.txt').read_text();s=read(D/'status.json')
 for k in list(s):
  if k.startswith('latest_round_'):
   if isinstance(s[k],bool):s[k]=False
   elif isinstance(s[k],(int,float)):s[k]=0
   else:assert k=='latest_round_new_root_fixture_scope',k
 for p,name in [(D/'status.json','status_before_foot_block_587.json'),(R/'docs/LOW_SEAT_CONTACT_STATUS.md','handoff_before_foot_block_587.md')]:
  assert not (D/name).exists();shutil.copy2(p,D/name)
 mf=read(D/'completed_probes_preservation_585.json');mf.update(read(D/'supplemental_preservation_585.json'))
 for folder in folders:
  for p in (D/folder).rglob('*'):
   if p.is_file():mf[str(p)]=sha(p)
 for pattern in ['*foot_block*586*.py','*foot_block*587*.py']:
  for p in (R/'scripts').glob(pattern):mf[str(p)]=sha(p)
 for pattern in ['foot_block_*.json','foot_block_*.log']:
  for p in D.glob(pattern):
   if p.is_file() and p.name!='foot_block_preservation_587.log':mf[str(p)]=sha(p)
 for name in ['handoff_prepared_587.txt','status_before_foot_block_587.json','handoff_before_foot_block_587.md']:mf[str(D/name)]=sha(D/name)
 manifest=D/'foot_block_evidence_preservation_587.json';assert not manifest.exists();manifest.write_text(json.dumps(mf,indent=2))
 for p,h in mf.items():assert sha(p)==h,p
 now=datetime.datetime.now(datetime.timezone.utc).isoformat()
 s.update(utc=now,updated_utc=now,phase='offline_foot_block586_audit587_preparation_complete',latest_handoff=str(D/'handoff_prepared_587.txt'),latest_preparation=str(D/folders[0]),latest_preparation_audit=str(D/folders[1]/'independent_audit.json'),latest_preparation_preservation=str(manifest),latest_preservation=str(manifest),completed_evidence_manifest=str(manifest),completed_evidence_count=len(mf),latest_preparation_handoff_record=str(D/'preparation_handoff_record_587.json'),latest_preparation_handoff=str(D/'handoff_prepared_587.txt'),last_sealed_closure=str(D/'closure_record_585.json'),latest_closure=str(D/'closure_record_585.json'),latest_attempt=str(D/'graph_path_582'),latest_complete_construction=str(D/'interior_path_484'),latest_audit_status='587 independent exact B*K=I,B*G+C=0,B*k=h plus reverseALLinequality/equality reconstruction COMPLETE. ExactREAL coupledfootgraph reduction only;no primalpoint/actualstateaudit.583sealed473geometry/work inheritance remains',audit_jobs_running=False,model_jobs_running=False,physical_jobs_running=False,latest_active_run=None,latest_active_snapshot=None,
 latest_round_new_exact_leg_block_inverses=2730,latest_round_new_exact_block_coefficient_solves=2730,latest_round_new_eliminated_leg_columns=16380,latest_round_new_forward_coefficient_products=q['operations']['substitution_coefficient_products'],latest_round_new_forward_RHS_products=q['operations']['substitution_RHS_products'],latest_round_new_root_fixture_scope='NEW exact coupled6x6 footblock dimension reduction of578 rationalzeroFACE.2730exactblockinverses/coefficient solves,16380physicallegcolumns/equalities eliminated;remaining14636coordinates include10540priorcontrols+4095rootxy/f+scalar. Alloriginalconditions represented,all265967inequalities and5462remainingE retained. ExactREALprojection only,possiblyempty/properzeroFACE. No numericalLP/numericmatrix/primalgraphlift/analyticdual/model/root/AD/geometry/finiteencoding/physics. No speed/conditioning/feasibility claim.',
 new_solver_exists=False,new_solver_scope='586 exact rational footblock reduction plus587 independent rank/graph/reverseidentity audit only. No588 roundednumericinput/encodingerrorpolicy/completealgorithm/child/wholepathwrapper/budget/sourcefreeze/return-and-actualstateauditor/primalwitness.582 remains completedread-only.',new_executable_whole_path_wrapper_exists=False,new_constructed_model_path_exists=False,new_candidate_exists=False,latest_preparation_scope=scope['scope'],latest_new_path_scope='Actual472 INITIAL X/Y/J plus490canonicalaux/exactdrops unchanged. No primal numerical witness,motion or effort result.',main_training_started=False,ppo_started=False,fresh_holdout_started=False,holdout_started=False,policy_promoted=False,task_complete=False,repair_complete=False,automation_status='ACTIVE',notes=notes)
 (D/'status.json').write_text(json.dumps(s,indent=2));doc=R/'docs/LOW_SEAT_CONTACT_STATUS.md';doc.write_text(notes+'\n\nEarlier preserved entries follow.\n\n'+doc.read_text())
 record=dict(utc=now,scope=__doc__,preparation_only=True,new_closure=False,last_sealed_closure=str(D/'closure_record_585.json'),prior_checks=checks,protected_files=len(mf),manifest=str(manifest),manifest_sha256=sha(manifest),workers=workers,actual_exits=exits,preparation=prep,independent_audit=audit,local_verification=review,saved_scope=scope,completed_scientific_calculations_repeated=False,scientific_execution_recovery=False,new_complete_numerical_algorithm=False,new_solver_exists=False,new_complete_path=False,numerical_lp_calls=0,numerical_linear_system_solves=0,exact_block_inverses=2730,exact_block_coefficient_solves=2730,model_calls=0,root_calls=0,finite_maps=0,physics_ticks=0,policy_updates=0,primal_vectors_constructed=0,analytic_dual_vectors_constructed=0,graph_point_lifts=0,rounded_numeric_input_exists=False,zero_face_feasibility_known=False,original_optimum_zero_established=False,protected_clean=64,unresolved_cases=8,repair_complete=False,task_complete=False,automation_status='ACTIVE')
 (D/'preparation_handoff_record_587.json').write_text(json.dumps(record,indent=2))
 handoff=[D/'status.json',R/'docs/LOW_SEAT_CONTACT_PROTOCOL.md',doc,D/'handoff_prepared_587.txt',manifest,D/'preparation_handoff_record_587.json',D/folders[0]/'preparation.json',D/folders[1]/'independent_audit.json',D/'foot_block_scope_587.json',D/'foot_block_local_verification_587.json']
 (D/'final_preparation_manifest_587.json').write_text(json.dumps({str(p):sha(p) for p in handoff},indent=2));print(json.dumps({k:record[k] for k in ['utc','prior_checks','protected_files','manifest','manifest_sha256','workers','new_closure','task_complete','automation_status']},indent=2),flush=True)
if __name__=='__main__':main()
