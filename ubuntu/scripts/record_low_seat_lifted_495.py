"""Seal completed492/493/494; preserve physical repair/PPO as unfinished."""
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
    assert not (D/'closure_record_495.json').exists();checks=[]
    for name in ['prior_hashes.json','lifted_evidence_preservation_491.json','lifted_supplemental_491.json']:
        m=read(D/name)
        for p,h in m.items():assert sha(p)==h,p
        checks.append(dict(manifest=name,files=len(m),passed=True))
    folders=['lifted_path_492','lifted_path_audit_493','progress_report_494']
    for name in folders:
        for p,h in read(D/name/'SHA256.json').items():assert sha(D/name/p)==h,p
    for p,h in read(D/'lifted_path_launch_492.json')['source_input_sha256'].items():assert sha(p)==h,p
    ex=read(D/'lifted_postprocessing_494.json');assert ex['main_actual_exit']==ex['independent_audit_actual_exit']==ex['report_actual_exit']==0
    main=read(D/'lifted_path_492/results.json');audit=read(D/'lifted_path_audit_493/independent_audit.json');scope=read(D/'lifted_saved_scope_495.json');review=read(D/'progress_report_494_local_review.json')
    assert audit['passed'] and review['all_hashes_verified'] and review['visual_review']
    assert main['actual_lp_calls']==scope['actual_scipy_linprog_calls'] and main['accepted_solver_iterates']==scope['accepted_solver_iterates']
    workers=[]
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        try:a=p.read_bytes().decode().split('\0')
        except (OSError,UnicodeError):continue
        if a and 'python' in a[0] and any((x.startswith('scripts/') or x.startswith(str(R/'scripts')+'/')) and Path(x).name!=Path(__file__).name for x in a[1:]):workers.append(a)
    assert not workers,workers
    for p,name in [(D/'status.json','status_before_lifted_495.json'),(R/'docs/LOW_SEAT_CONTACT_STATUS.md','handoff_before_lifted_495.md')]:
        assert not (D/name).exists();shutil.copy2(p,D/name)
    mf=read(D/'lifted_evidence_preservation_491.json');mf.update(read(D/'lifted_supplemental_491.json'))
    for name in folders:
        for p in (D/name).rglob('*'):
            if p.is_file():mf[str(p)]=sha(p)
    for name in ['low_seat_lifted_execution_492.py','build_low_seat_lifted_path_492.py','freeze_low_seat_lifted_path_492.py','audit_low_seat_lifted_steps_493.py','audit_low_seat_lifted_path_493.py','audit_low_seat_lifted_result_493.py','report_low_seat_lifted_494.py','index_low_seat_lifted_495.py','record_low_seat_lifted_495.py']:
        p=R/'scripts'/name;mf[str(p)]=sha(p)
    for pattern in ['lifted*.json','lifted*.log','progress_report_494_local_review.json']:
        for p in D.glob(pattern):
            if p.is_file() and p.name!='lifted_preservation_495.log':mf[str(p)]=sha(p)
    for name in ['handoff_update_495.txt','status_before_lifted_495.json','handoff_before_lifted_495.md','active_lifted_492.json']:mf[str(D/name)]=sha(D/name)
    manifest=D/'completed_probes_preservation_495.json';manifest.write_text(json.dumps(mf,indent=2))
    for p,h in mf.items():assert sha(p)==h,p
    now=datetime.datetime.now(datetime.timezone.utc).isoformat();notes=(D/'handoff_update_495.txt').read_text();status=read(D/'status.json');inherited=bool(audit.get('all_geometry_and_work_inherited',False));newwork=int(audit['new_finite_work_columns']);changed=main['final_actual_state_changed']
    status.update(utc=now,updated_utc=now,phase='offline_lifted492_complete_actual_admission_unmet',audit_jobs_running=False,model_jobs_running=False,physical_jobs_running=False,
        latest_active_run=None,latest_active_snapshot=None,latest_handoff=str(D/'handoff_update_495.txt'),latest_preservation=str(manifest),completed_evidence_manifest=str(manifest),completed_evidence_count=len(mf),last_sealed_closure=str(D/'closure_record_495.json'),latest_closure=str(D/'closure_record_495.json'),latest_attempt=str(D/'lifted_path_492'),
        latest_complete_construction=str(D/('lifted_path_492' if changed else 'interior_path_484')),latest_complete_construction_scope='Latest newly constructed actual-model state set;492 auxiliary initial is not a changed actual path.' if not changed else '492 actual-X diagnostic; actual admission checked separately.',
        latest_independent_audit=str(D/'lifted_path_audit_493/independent_audit.json'),latest_independent_audit_passed=True,latest_report=str(D/'progress_report_494/findings.html'),latest_report_local_review=str(D/'progress_report_494_local_review.json'),
        latest_round_constructed_model_records=1467 if changed else 0,latest_round_new_complete_trial_paths=main['actual_trial_evaluations'],latest_round_new_initial_complete_lifts=0,latest_round_new_model_derivative_fixture_columns=0,latest_round_new_arithmetic_map_fixture_coordinates=0,latest_round_new_joint_arithmetic_fixtures=0,latest_round_new_certificate_arithmetic_cases=0,latest_round_new_assembly_fixtures=0,latest_round_offline_assembly_fixtures=0,latest_round_repeated_scalar_arithmetic_regression_checks=0,latest_round_new_root_fixture_states=0,
        latest_round_new_finite_work_columns=newwork,latest_round_inherited_geometry_records=1467 if inherited else 0,latest_round_inherited_finite_work_records=sum(int(audit.get(k,0)) for k in ['inherited479_exact_records','inherited473_exact_records','inherited441_exact_records']),latest_round_duplicate_new_finite_work_records=int(audit.get('duplicate_new_records',0)),
        latest_round_new_full20s_trials=0,latest_round_initial_settling_ticks=0,latest_round_new_active_ticks=0,latest_round_new_diagnostic_prefixes=0,latest_round_new_aborted_case_attempts=0,latest_round_optimizer_updates=0,latest_round_actual_forward_filter_ticks=0,latest_round_command_ticks_per_device=0,
        latest_round_offline_lp_calls=main['actual_lp_calls'],latest_round_offline_assembled_subproblems=main['assembled_subproblems'],latest_round_offline_empty_correction_prechecks=0,latest_round_offline_map_failures=main['map_failures'],latest_round_offline_certificate_failures=main['certificate_failures'],latest_round_offline_milp_calls=0,latest_round_offline_nlp_calls=0,latest_round_offline_nonlinear_iterations=0,latest_round_offline_nonlinear_trial_evaluations=main['actual_trial_evaluations'],latest_round_offline_accepted_steps=main['accepted_solver_iterates'],
        new_solver_exists=True,new_solver_scope='492 bounded full-space run COMPLETE/read-only; no next solver authorized by restarting it.',prospective_joint_solver_assembly_exists=True,new_finite_lifted_decoder_exists=True,new_executable_whole_path_wrapper_exists=True,new_constructed_model_path_exists=changed,new_candidate_exists=False,latest_new_path_scope=review['outcome_scope'],
        main_training_started=False,ppo_started=False,fresh_holdout_started=False,holdout_started=False,policy_promoted=False,task_complete=False,repair_complete=False,automation_status='ACTIVE',notes=notes)
    (D/'status.json').write_text(json.dumps(status,indent=2));doc=R/'docs/LOW_SEAT_CONTACT_STATUS.md';doc.write_text(notes+'\n\nEarlier preserved entries follow.\n\n'+doc.read_text())
    record=dict(utc=now,scope=__doc__,prior_checks=checks,protected_files=len(mf),manifest=str(manifest),manifest_sha256=sha(manifest),workers=workers,actual_exits=ex,main_result=main,independent_audit=audit,saved_scope=scope,local_review=review,new_closure=True,accepted_solver_iterates=main['accepted_solver_iterates'],actual_final_state_changed=changed,all_final_geometry_inherited=inherited,new_finite_work_columns=newwork,actual_scipy_lp_calls=main['actual_lp_calls'],finite_maps=main['finite_maps'],nonlinear_trials=main['actual_trial_evaluations'],actual_forward_filter_ticks=0,physics_ticks=0,policy_updates=0,best_clean='64/72',protected_clean=64,unresolved_cases=8,repair_complete=False,task_complete=False,automation_status='ACTIVE')
    (D/'closure_record_495.json').write_text(json.dumps(record,indent=2));print(json.dumps({k:v for k,v in record.items() if k not in ['main_result','independent_audit','saved_scope','local_review']},indent=2),flush=True)
if __name__=='__main__':main()
