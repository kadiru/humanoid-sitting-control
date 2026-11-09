"""Seal completed506/507/508 after actual report exit and local review; preserve physical repair/PPO as unfinished."""
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
    assert not (D/'closure_record_509.json').exists();checks=[]
    for name in ['prior_hashes.json','sparse_gap_evidence_preservation_503.json','sparse_gap_supplemental_503.json']:
        m=read(D/name)
        for p,h in m.items():assert sha(p)==h,p
        checks.append(dict(manifest=name,files=len(m),passed=True))
    folders=['mixed_coordinate_preparation_504','mixed_coordinate_audit_505','mixed_path_506','mixed_path_audit_507','progress_report_508']
    for name in folders:
        for p,h in read(D/name/'SHA256.json').items():assert sha(D/name/p)==h,p
    for p,h in read(D/'mixed_path_launch_506.json')['source_input_sha256'].items():assert sha(p)==h,p
    pex={n:read(D/n) for n in ['mixed_compile_exit_504.json','mixed_freeze_exit_504.json','mixed_preparation_exit_504.json','mixed_audit_exit_505.json']};assert all(x['actual_exit']==0 for x in pex.values())
    ex=read(D/'mixed_postprocessing_508.json');assert ex['main_actual_exit']==ex['independent_audit_actual_exit']==ex['report_actual_exit']==0
    main=read(D/'mixed_path_506/results.json');audit=read(D/'mixed_path_audit_507/independent_audit.json');scope=read(D/'mixed_saved_scope_509.json');review=read(D/'progress_report_508_local_review.json')
    assert audit['passed'] and review['all_hashes_verified'] and review['visual_review']
    assert scope['worker_actual_exits']==[0] and scope['worker_wall_cap_hits']==0 and scope['returned_witnesses']==0
    assert main['actual_lp_calls']==scope['actual_scipy_linprog_calls'] and main['accepted_solver_iterates']==scope['accepted_solver_iterates']
    workers=[]
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        try:a=p.read_bytes().decode().split('\0')
        except (OSError,UnicodeError):continue
        if a and 'python' in a[0] and any((x.startswith('scripts/') or x.startswith(str(R/'scripts')+'/')) and Path(x).name!=Path(__file__).name for x in a[1:]):workers.append(a)
    assert not workers,workers
    for p,name in [(D/'status.json','status_before_mixed_509.json'),(R/'docs/LOW_SEAT_CONTACT_STATUS.md','handoff_before_mixed_509.md')]:
        assert not (D/name).exists();shutil.copy2(p,D/name)
    mf=read(D/'sparse_gap_evidence_preservation_503.json');mf.update(read(D/'sparse_gap_supplemental_503.json'))
    for name in folders:
        for p in (D/name).rglob('*'):
            if p.is_file():mf[str(p)]=sha(p)
    for name in ['low_seat_mixed_coordinates_504.py','prepare_low_seat_mixed_coordinates_504.py','audit_low_seat_mixed_coordinates_505.py','low_seat_mixed_execution_506.py','solve_low_seat_mixed_lp_506.py','build_low_seat_mixed_path_506.py','freeze_low_seat_mixed_path_506.py','audit_low_seat_mixed_steps_507.py','audit_low_seat_mixed_path_507.py','audit_low_seat_mixed_result_507.py','report_low_seat_mixed_508.py','report_low_seat_mixed_508_unexecuted_draft.py','index_low_seat_mixed_509.py','record_low_seat_mixed_509.py']:
        p=R/'scripts'/name;mf[str(p)]=sha(p)
    for pattern in ['mixed*.json','mixed*.log','active_mixed_506.json','progress_report_508_local_review.json']:
        for p in D.glob(pattern):
            if p.is_file() and p.name!='mixed_preservation_509.log':mf[str(p)]=sha(p)
    for name in ['handoff_update_509.txt','status_before_mixed_509.json','handoff_before_mixed_509.md']:mf[str(D/name)]=sha(D/name)
    manifest=D/'completed_probes_preservation_509.json';manifest.write_text(json.dumps(mf,indent=2))
    for p,h in mf.items():assert sha(p)==h,p
    now=datetime.datetime.now(datetime.timezone.utc).isoformat();notes=(D/'handoff_update_509.txt').read_text();status=read(D/'status.json');inherited=bool(audit.get('all_geometry_and_work_inherited',False));newwork=int(audit['new_finite_work_columns']);changed=main['final_actual_state_changed']
    status.update(utc=now,updated_utc=now,phase='offline_mixed506_complete_actual_admission_unmet',audit_jobs_running=False,model_jobs_running=False,physical_jobs_running=False,
        latest_active_run=None,latest_active_snapshot=None,latest_handoff=str(D/'handoff_update_509.txt'),latest_preservation=str(manifest),completed_evidence_manifest=str(manifest),completed_evidence_count=len(mf),last_sealed_closure=str(D/'closure_record_509.json'),latest_closure=str(D/'closure_record_509.json'),latest_attempt=str(D/'mixed_path_506'),
        latest_complete_construction=str(D/('mixed_path_506' if changed else 'interior_path_484')),latest_complete_construction_scope='Latest newly constructed actual-model state set;506 auxiliary initial is not a changed actual path.' if not changed else '506 actual-X diagnostic; actual admission checked separately.',
        latest_independent_audit=str(D/'mixed_path_audit_507/independent_audit.json'),latest_independent_audit_passed=True,latest_report=str(D/'progress_report_508/findings.html'),latest_report_local_review=str(D/'progress_report_508_local_review.json'),
        latest_round_constructed_model_records=1467 if changed else 0,latest_round_new_complete_trial_paths=main['actual_trial_evaluations'],latest_round_new_initial_complete_lifts=0,latest_round_new_model_derivative_fixture_columns=0,latest_round_new_arithmetic_map_fixture_coordinates=0,latest_round_new_joint_arithmetic_fixtures=0,latest_round_new_certificate_arithmetic_cases=0,latest_round_new_assembly_fixtures=2,latest_round_offline_assembly_fixtures=2,latest_round_repeated_scalar_arithmetic_regression_checks=0,latest_round_new_root_fixture_states=0,
        latest_round_new_local_saved_matrix_mixed_solves=0,latest_round_new_algebraic_right_hand_sides=0,latest_round_new_affine_vector_checks=0,latest_round_new_directed_inverse_certificates=0,latest_round_new_finite_work_columns=newwork,latest_round_inherited_geometry_records=1467 if inherited else 0,latest_round_inherited_finite_work_records=sum(int(audit.get(k,0)) for k in ['inherited479_exact_records','inherited473_exact_records','inherited441_exact_records']),latest_round_duplicate_new_finite_work_records=int(audit.get('duplicate_new_records',0)),
        latest_round_new_full20s_trials=0,latest_round_initial_settling_ticks=0,latest_round_new_active_ticks=0,latest_round_new_diagnostic_prefixes=0,latest_round_new_aborted_case_attempts=0,latest_round_optimizer_updates=0,latest_round_actual_forward_filter_ticks=0,latest_round_command_ticks_per_device=0,
        latest_round_offline_lp_calls=main['actual_lp_calls'],latest_round_offline_assembled_subproblems=main['assembled_subproblems'],latest_round_offline_empty_correction_prechecks=0,latest_round_offline_map_failures=main['map_failures'],latest_round_offline_certificate_failures=main['certificate_failures'],latest_round_offline_milp_calls=0,latest_round_offline_nlp_calls=0,latest_round_offline_nonlinear_iterations=0,latest_round_offline_nonlinear_trial_evaluations=main['actual_trial_evaluations'],latest_round_offline_accepted_steps=main['accepted_solver_iterates'],
        new_solver_exists=True,new_solver_scope='506 partial-residual/nativeauxiliary HiGHS IPM run COMPLETE/read-only; first primary timeout/no witness. No510 solver or candidate.',prospective_joint_solver_assembly_exists=True,new_finite_lifted_decoder_exists=True,new_executable_whole_path_wrapper_exists=True,new_constructed_model_path_exists=changed,new_candidate_exists=False,latest_new_path_scope=review['outcome_scope'],
        latest_audit_status='507 conditional audit COMPLETE; one original/transformed assembly replay; all1467sealed473 geometry/work inherited; no new model/path.',latest_audit=str(D/'mixed_path_audit_507/independent_audit.json'),latest_audit_passed=True,latest_progress_report=str(D/'progress_report_508/findings.html'),latest_nominal_research_attempt=str(D/'mixed_path_506/results.json'),latest_preparation=str(D/'mixed_coordinate_preparation_504'),latest_preparation_audit=str(D/'mixed_coordinate_audit_505/independent_audit.json'),latest_preparation_scope='504/505 partial kinematic-residual/nativeauxiliary basis and all-row saved-matrix preparation COMPLETE; inherited497subblockproof and503incidence; primary6099134nnz. No inverse/proof/map/model fixture repeats;506 initial basis inherited.',latest_preparation_handoff=str(D/'closure_record_509.json'),latest_preparation_preservation=str(manifest),latest_preparation_handoff_record=str(D/'closure_record_509.json'),latest_candidate_status='Latest physical candidate273 remains rejected. Latest NEWactual-model construction484 hard-feasible but original rates/targets fail;506 yields no LP witness/new actual state.',main_training_started=False,ppo_started=False,fresh_holdout_started=False,holdout_started=False,policy_promoted=False,task_complete=False,repair_complete=False,automation_status='ACTIVE',notes=notes)
    (D/'status.json').write_text(json.dumps(status,indent=2));doc=R/'docs/LOW_SEAT_CONTACT_STATUS.md';doc.write_text(notes+'\n\nEarlier preserved entries follow.\n\n'+doc.read_text())
    record=dict(utc=now,scope=__doc__,prior_checks=checks,protected_files=len(mf),manifest=str(manifest),manifest_sha256=sha(manifest),workers=workers,actual_exits=ex,preparation_actual_exits=pex,preparation=read(D/'mixed_coordinate_preparation_504/preparation.json'),preparation_audit=read(D/'mixed_coordinate_audit_505/independent_audit.json'),main_result=main,independent_audit=audit,saved_scope=scope,local_review=review,new_closure=True,accepted_solver_iterates=main['accepted_solver_iterates'],actual_final_state_changed=changed,all_final_geometry_inherited=inherited,new_finite_work_columns=newwork,actual_scipy_lp_calls=main['actual_lp_calls'],finite_maps=main['finite_maps'],nonlinear_trials=main['actual_trial_evaluations'],actual_forward_filter_ticks=0,physics_ticks=0,policy_updates=0,best_clean='64/72',protected_clean=64,unresolved_cases=8,repair_complete=False,task_complete=False,automation_status='ACTIVE')
    (D/'closure_record_509.json').write_text(json.dumps(record,indent=2));print(json.dumps({k:v for k,v in record.items() if k not in ['preparation','preparation_audit','main_result','independent_audit','saved_scope','local_review']},indent=2),flush=True)
if __name__=='__main__':main()
