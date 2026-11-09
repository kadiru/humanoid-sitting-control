"""Preserve completed490/491 preparation; keep sealed487 and unfinished task."""
from pathlib import Path
import json, hashlib, datetime, shutil
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')
R=Path('/home/kadir/humanoid-sitting')
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4194304),b''):h.update(b)
    return h.hexdigest()
def main():
    assert not (D/'preparation_handoff_record_491.json').exists()
    prior=['prior_hashes.json','feature_evidence_preservation_489.json','feature_supplemental_489.json','feature_reference_scope_supplemental_489.json']
    checks=[]
    for name in prior:
        m=read(D/name)
        for p,h in m.items():assert sha(p)==h,p
        checks.append(dict(manifest=name,files=len(m),passed=True))
    exit_names=['lifted_compile_exit_490.json','lifted_preparation_freeze_exit_490.json','lifted_preparation_exit_490.json','lifted_audit_exit_491.json','lifted_record_compile_exit_491.json']
    exits={n:read(D/n) for n in exit_names}
    assert all(e['actual_exit']==0 for e in exits.values())
    launch=read(D/'lifted_preparation_launch_490.json')
    for p,h in launch['source_input_sha256'].items():assert sha(p)==h,p
    folders=['lifted_feasibility_preparation_490','lifted_feasibility_audit_491']
    for name in folders:
        for p,h in read(D/name/'SHA256.json').items():assert sha(D/name/p)==h,p
    prep=read(D/folders[0]/'preparation.json')
    audit=read(D/folders[1]/'independent_audit.json')
    local=read(D/'lifted_local_verification_491.json')
    assert prep['passed'] and audit['passed'] and local['passed']
    assert prep['new_joint_fixtures']==3 and prep['new_direct_coordinates_covered']==68250
    for record in (prep,audit):
        assert all(record[k]==0 for k in ['model_calls','root_calls','ad_calls','lp_calls'])
        assert not record['new_complete_path'] and not record['physical_admission']
    workers=[]
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        try:a=p.read_bytes().decode().split('\0')
        except (OSError,UnicodeError):continue
        if a and 'python' in a[0] and any((x.startswith('scripts/') or x.startswith(str(R/'scripts')+'/')) and Path(x).name!=Path(__file__).name for x in a[1:]):workers.append(a)
    assert not workers,workers
    for p,name in [(D/'status.json','status_before_lifted_491.json'),(R/'docs/LOW_SEAT_CONTACT_STATUS.md','handoff_before_lifted_491.md')]:
        assert not (D/name).exists();shutil.copy2(p,D/name)
    mf={}
    for name in prior[1:]:mf.update(read(D/name))
    for name in folders:
        for p in (D/name).rglob('*'):
            if p.is_file():mf[str(p)]=sha(p)
    for name in ['low_seat_lifted_feasibility_490.py','prepare_low_seat_lifted_feasibility_490.py','audit_low_seat_lifted_feasibility_491.py','record_low_seat_lifted_preparation_491.py']:
        p=R/'scripts'/name;mf[str(p)]=sha(p)
    for pattern in ['lifted*.json','lifted*.log']:
        for p in D.glob(pattern):
            if p.is_file() and p.name!='lifted_preservation_491.log':mf[str(p)]=sha(p)
    for name in ['handoff_prepared_491.txt','status_before_lifted_491.json','handoff_before_lifted_491.md']:
        p=D/name;mf[str(p)]=sha(p)
    manifest=D/'lifted_evidence_preservation_491.json';manifest.write_text(json.dumps(mf,indent=2))
    for p,h in mf.items():assert sha(p)==h,p
    now=datetime.datetime.now(datetime.timezone.utc).isoformat()
    notes=(D/'handoff_prepared_491.txt').read_text();status=read(D/'status.json')
    status.update(utc=now,updated_utc=now,
        phase='offline_lifted_feasibility490_preparation_audit491_complete_runwrapper_unimplemented',
        audit_jobs_running=False,model_jobs_running=False,physical_jobs_running=False,
        latest_active_run=None,latest_active_snapshot=None,
        latest_handoff=str(D/'handoff_prepared_491.txt'),
        latest_preparation=str(D/folders[0]),latest_preparation_audit=str(D/folders[1]/'independent_audit.json'),
        latest_preparation_audit_passed=True,latest_preservation=str(manifest),
        completed_evidence_manifest=str(manifest),completed_evidence_count=len(mf),
        last_sealed_closure=str(D/'closure_record_487.json'),latest_closure=str(D/'closure_record_487.json'),
        latest_round_constructed_model_records=0,latest_round_new_complete_trial_paths=0,
        latest_round_new_initial_complete_lifts=0,latest_round_new_model_derivative_fixture_columns=0,
        latest_round_new_root_fixture_states=0,latest_round_new_finite_work_columns=0,
        latest_round_inherited_geometry_records=0,latest_round_inherited_finite_work_records=0,
        latest_round_duplicate_new_finite_work_records=0,latest_round_new_full20s_trials=0,
        latest_round_initial_settling_ticks=0,latest_round_new_active_ticks=0,
        latest_round_new_diagnostic_prefixes=0,latest_round_new_aborted_case_attempts=0,
        latest_round_optimizer_updates=0,latest_round_actual_forward_filter_ticks=0,
        latest_round_command_ticks_per_device=0,latest_round_offline_lp_calls=0,
        latest_round_offline_assembled_subproblems=0,latest_round_offline_assembly_fixtures=2,
        latest_round_offline_empty_correction_prechecks=0,latest_round_offline_map_failures=0,
        latest_round_offline_certificate_failures=0,latest_round_offline_milp_calls=0,
        latest_round_offline_nlp_calls=0,latest_round_offline_nonlinear_iterations=0,
        latest_round_offline_nonlinear_trial_evaluations=0,latest_round_offline_accepted_steps=0,
        latest_round_new_arithmetic_map_fixture_coordinates=68250,
        latest_round_new_joint_arithmetic_fixtures=3,
        latest_round_repeated_scalar_arithmetic_regression_checks=1,
        new_solver_exists=False,
        new_solver_scope='No492 executable solver/run wrapper. New490 finite decoder and full-space LP assembly are prepared/audited only; old484 stays complete/read-only.',
        prospective_joint_solver_assembly_exists=True,new_finite_lifted_decoder_exists=True,
        new_executable_whole_path_wrapper_exists=False,new_constructed_model_path_exists=False,new_candidate_exists=False,
        latest_new_path_scope='Preparation only:78790 joint coordinates,68250 consistency rows,92267 finite intervals,3 NEW joint arithmetic fixtures. Canonical auxiliaries differ explicitly from saved actualY; actualX/Y/J unchanged. No actual path or LP solve.',
        main_training_started=False,ppo_started=False,fresh_holdout_started=False,holdout_started=False,
        policy_promoted=False,task_complete=False,repair_complete=False,automation_status='ACTIVE',notes=notes)
    (D/'status.json').write_text(json.dumps(status,indent=2))
    doc=R/'docs/LOW_SEAT_CONTACT_STATUS.md';doc.write_text(notes+'\n\nEarlier preserved entries follow.\n\n'+doc.read_text())
    record=dict(utc=now,scope=__doc__,preparation_only=True,new_closure=False,
        last_sealed_closure=str(D/'closure_record_487.json'),prior_checks=checks,
        protected_files=len(mf),manifest=str(manifest),manifest_sha256=sha(manifest),workers=workers,
        actual_exits=exits,preparation=prep,independent_audit=audit,local_verification=local,
        new_solver_exists=False,prospective_joint_solver_assembly_exists=True,new_finite_lifted_decoder_exists=True,
        new_complete_path=False,mechanics_calls=0,root_calls=0,ad_calls=0,finite_work_columns=0,lp_calls=0,
        actual_forward_filter_ticks=0,physics_ticks=0,policy_updates=0,
        new_joint_arithmetic_fixtures=3,new_direct_coordinates_covered=68250,
        repeated_scalar_arithmetic_regression_checks=1,best_clean='64/72',protected_clean=64,unresolved_cases=8,
        repair_complete=False,task_complete=False,automation_status='ACTIVE')
    (D/'preparation_handoff_record_491.json').write_text(json.dumps(record,indent=2))
    print(json.dumps({k:v for k,v in record.items() if k not in ['preparation','independent_audit','local_verification']},indent=2),flush=True)
if __name__=='__main__':main()
