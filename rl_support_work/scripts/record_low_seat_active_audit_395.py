"""Record an active395 audit without claiming its completion or closing the round."""
from pathlib import Path
import datetime, hashlib, json, shutil, subprocess

D = Path('/data/humanoid-sitting-perception/low_seat_contact_01')
R = Path('/home/kadir/humanoid-sitting')
read = lambda p: json.loads(Path(p).read_text())

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(4194304), b''):
            h.update(b)
    return h.hexdigest()

def main():
    assert not (D/'active_audit_395.json').exists()
    assert not (D/'integer_restoration_path_audit_395/independent_audit.json').exists()
    assert not (D/'closure_record_397.json').exists()
    proc = subprocess.run(['ps','-p','1288527','-o','pid,etime,pcpu,args'], check=True, text=True, capture_output=True).stdout
    assert 'scripts/audit_low_seat_integer_path_395.py' in proc
    completed_folders = ['integer_restoration_preparation_392', 'integer_restoration_preparation_audit_393', 'integer_restoration_path_394']
    entries = 0
    for folder in completed_folders:
        mf = read(D/folder/'SHA256.json')
        assert all(sha(D/folder/p) == h for p,h in mf.items())
        entries += len(mf)
    assert read(D/'integer_restoration_preparation_audit_393/independent_audit.json')['passed']
    ex = read(D/'integer_restoration_exit_394.json')
    assert ex['observed_exit_code'] == 0 and ex['source_hashes_intact']
    result = read(D/'integer_restoration_path_394/results.json')
    for p,h in result['source_input_sha256'].items():
        assert sha(p) == h, p
    prior = []
    for name in ['prior_hashes.json','completed_probes_preservation_391.json','supplemental_preservation_391.json']:
        mf = read(D/name)
        assert all(Path(p).exists() and sha(p) == h for p,h in mf.items()), name
        prior.append(dict(manifest=name, checked=len(mf), passed=True))
    # Assert still active before any mutable status/document writes.
    assert not (D/'integer_restoration_path_audit_395/independent_audit.json').exists()
    proc = subprocess.run(['ps','-p','1288527','-o','pid,etime,pcpu,args'], check=True, text=True, capture_output=True).stdout
    assert 'scripts/audit_low_seat_integer_path_395.py' in proc
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    for src,name in [(D/'status.json','status_before_active_audit_395.json'),(R/'docs/LOW_SEAT_CONTACT_STATUS.md','handoff_before_active_audit_395.md')]:
        assert not (D/name).exists()
        shutil.copy2(src,D/name)
    priorpath = D/'prior_recheck_active_audit_395.json'
    priorpath.write_text(json.dumps(dict(utc=now,checks=prior,passed=True),indent=2))
    protected = read(D/'completed_probes_preservation_391.json')
    protected.update(read(D/'supplemental_preservation_391.json'))
    for folder in completed_folders:
        for p in (D/folder).rglob('*'):
            if p.is_file(): protected[str(p)] = sha(p)
    names = ['integer_restoration_preparation_392.log','integer_restoration_preparation_audit_393.log',
             'integer_restoration_path_394.log','integer_restoration_launch_394.json','integer_restoration_exit_394.json',
             'integer_restoration_defect_397.json','handoff_running_audit_395.txt',
             'status_before_active_audit_395.json','handoff_before_active_audit_395.md',priorpath.name]
    for name in names:
        p = D/name; protected[str(p)] = sha(p)
    for name in ['prepare_low_seat_integer_restoration_392.py','audit_low_seat_integer_preparation_393.py',
                 'build_low_seat_integer_restoration_394.py','audit_low_seat_integer_path_395.py',
                 'report_low_seat_integer_restoration_396.py','record_low_seat_active_audit_395.py']:
        p = R/'scripts'/name; protected[str(p)] = sha(p)
    for p in (D/'integer_pending_postprocessing_397').rglob('*'):
        if p.is_file(): protected[str(p)] = sha(p)
    mfpath = D/'integer_restoration_evidence_preservation_395.json'
    assert not mfpath.exists()
    mfpath.write_text(json.dumps(protected,indent=2))
    assert all(sha(p)==h for p,h in protected.items())
    active = dict(utc=now,unified_exec_session=19728,worker=proc.strip(),worker_pid=1288527,
        audit_exit_status='PENDING',audit_395_started_once=True,audit_395_complete=False,
        main_394_actual_exit=ex,main_394_complete=True,main_394_nonlinear_feasible=False,
        report_396_executed=False,closure_397_executed=False,local_report_review_396_complete=False,
        pending_drafts='integer_pending_postprocessing_397',
        completed_manifest=str(mfpath),completed_files=len(protected),manifest_sha256=sha(mfpath),
        completed_folder_manifests=len(completed_folders),completed_manifest_entries=entries,
        changing_audit_log_output_excluded=True,prior_checks=prior,physics_ticks=0,
        actual_candidate_command_schedule_ticks=0,policy_optimizer_updates=0,repair_complete=False,
        prior_sealed_closure='closure_record_391.json')
    (D/'active_audit_395.json').write_text(json.dumps(active,indent=2))
    notes=(D/'handoff_running_audit_395.txt').read_text()
    s=read(D/'status.json')
    s.update(utc=now,updated_utc=now,phase='integer_restoration_394_rejected_independent_audit_395_running',
        audit_jobs_running=True,model_jobs_running=False,physical_jobs_running=False,
        latest_active_run=str(D/'active_audit_395.json'),latest_active_snapshot=None,
        latest_handoff=str(D/'handoff_running_audit_395.txt'),latest_preservation=str(mfpath),
        completed_evidence_manifest=str(mfpath),completed_evidence_count=len(protected),
        latest_offline_path=str(D/'integer_restoration_path_394'),
        latest_offline_audit=str(D/'integer_restoration_path_audit_395/independent_audit.json'),
        latest_offline_audit_passed=None,latest_audit_status='Independent395 active; no completion claim.',
        latest_offline_gate_scope='Main394 reports nonlinear rejection; independent395 still running. No actual filter or physics.',
        latest_round_new_active_ticks=0,latest_round_initial_settling_ticks=0,
        latest_round_new_full20s_trials=0,latest_round_new_diagnostic_prefixes=0,
        latest_round_new_aborted_case_attempts=0,latest_round_optimizer_updates=0,
        latest_round_command_ticks_per_device=0,latest_round_offline_nonlinear_iterations=0,
        latest_round_offline_milp_calls=1,latest_round_offline_milp_nodes=1,
        main_training_started=False,ppo_started=False,fresh_holdout_started=False,holdout_started=False,
        policy_promoted=False,task_complete=False,repair_complete=False,notes=notes)
    (D/'status.json').write_text(json.dumps(s,indent=2))
    doc=R/'docs/LOW_SEAT_CONTACT_STATUS.md'
    doc.write_text(notes+'\n\nEarlier preserved entries follow.\n\n'+doc.read_text())
    print(json.dumps(active,indent=2),flush=True)

if __name__ == '__main__':
    main()
