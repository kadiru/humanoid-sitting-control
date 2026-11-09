"""Preserve completed diagnostic189, including its FAILED global replay gate."""
from pathlib import Path
import datetime, hashlib, json, re, shutil, subprocess

D = Path('/data/humanoid-sitting-perception/low_seat_contact_01')
R = Path('/home/kadir/humanoid-sitting')

def read(p):
    return json.loads(Path(p).read_text())

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(4194304), b''):
            h.update(b)
    return h.hexdigest()

def main():
    assert not (D/'closure_record_195.json').exists()
    audit = read(D/'substep_audit_v2_190/independent_audit.json')
    diagnosis = read(D/'substep_diagnosis_v3_191/results.json')
    video = read(D/'all_attempts_video_audit_193/independent_audit.json')
    assert audit['diagnostic_checks_passed'] is False
    assert audit['exact_prefix_replay'] is False and audit['physical_admission'] is False
    assert audit['all_substep_raw_exact'] and audit['substep_raw_checks'] == 4016
    assert audit['fixed_fingers'] == 14 and audit['source_hashes_unchanged']
    assert diagnosis['exact_prefix_replay'] is False
    assert video['passed'] and video['full_trials'] == 792 and video['recorded_prefixes'] == 96
    assert video['event_frames_checked'] == 544
    accounting = audit['accounting']
    assert accounting['active_ticks'] == 15600 and accounting['physics_substeps'] == 2600
    assert accounting['initial_settling_excluded'] == 600 and accounting['complete20s_episodes'] == 0
    assert accounting['optimizer_updates'] == 0 and not accounting['fresh_holdout']
    sources = read(D/'substep_revision_189.json')['sources']
    assert len(sources) == 43
    for p,h in sources.items():
        assert sha(p) == h, p
    folders = ['progress_report_194','substep_audit_v2_190','substep_diagnosis_v3_191',
               'all_attempts_video_192','all_attempts_video_audit_193',
               'substep_shutdown_recovery_189','substep_audit_recovery_190',
               'substep_diagnosis_recovery_191']
    for folder in folders:
        for p,h in read(D/folder/'SHA256.json').items():
            assert sha(D/folder/p) == h, (folder,p)
    review = read(D/'progress_report_194_local_review.json')
    assert review['local_hashes_verified'] and review['total_hash_entries'] == 43
    assert review['browser_visually_inspected'] and review['video_frames_visually_inspected']
    assert review['embedded_video_loaded'] and not review['physical_admission']
    processes = subprocess.run(['ps','-eo','pid,args'],capture_output=True,text=True,check=True).stdout.splitlines()
    workers = [s for s in processes if re.search(
        r'^\s*\d+\s+\S*python\S* .*scripts/(run|refine|optimize|audit|diagnose|schedule|check|bound|screen|render|verify|finalize|report|launch)_low_seat_\S+\.py',s)]
    assert not workers, workers
    checks = []
    for name in ['prior_hashes.json','completed_probes_preservation_188.json','supplemental_preservation_188.json']:
        m = read(D/name)
        bad = [p for p,h in m.items() if not Path(p).exists() or sha(p) != h]
        checks.append(dict(manifest=name,checked=len(m),mismatches=bad,passed=not bad))
        assert not bad, bad[:5]
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    prior = dict(utc=now,passed=True,checks=checks)
    priorpath = D/'prior_recheck_after_substep_195.json'
    priorpath.write_text(json.dumps(prior,indent=2))
    status = D/'status.json'
    doc = R/'docs/LOW_SEAT_CONTACT_STATUS.md'
    protocol = R/'docs/LOW_SEAT_CONTACT_PROTOCOL.md'
    for p,name in [(status,'status_before_round_195.json'),(doc,'handoff_before_round_195.md'),
                   (protocol,'protocol_before_round_195.md')]:
        assert not (D/name).exists()
        shutil.copy2(p,D/name)
    # Old per-candidate statuses are preserved evidence; do not rewrite them.
    protected = dict(read(D/'completed_probes_preservation_188.json'))
    protected.update(read(D/'supplemental_preservation_188.json'))
    excluded = {'closure_195.log','completed_probes_preservation_195.json','closure_record_195.json'}
    for p in D.iterdir():
        match = re.search(r'_(\d+)(?:\.py)?(?:\.[^.]*)?$',p.name)
        if not match or not 189 <= int(match[1]) <= 195 or p.name in excluded:
            continue
        if p.is_dir():
            for f in p.rglob('*'):
                if f.is_file():
                    protected[str(f)] = sha(f)
        elif p.is_file():
            protected[str(p)] = sha(p)
    for p in (R/'scripts').glob('*.py'):
        match = re.search(r'_(\d+)\.py$',p.name)
        if match and 189 <= int(match[1]) <= 195:
            protected[str(p)] = sha(p)
    protected[str(D/'progress_report_194_local_review.json')] = sha(D/'progress_report_194_local_review.json')
    mf = D/'completed_probes_preservation_195.json'
    mf.write_text(json.dumps(protected,indent=2))
    assert all(sha(p) == h for p,h in protected.items())
    notes = (D/'handoff_update_195.txt').read_text()
    doc.write_text(notes+'\n\nEarlier preserved entries follow.\n\n'+doc.read_text())
    with protocol.open('a') as f:
        f.write('\n\nRound195 closure: diagnostic189 completed13s,15600 active control environment ticks+600 settling,0 complete20s trials/optimizer. Global exact replay FAILED and stays failed; selected poses match but torque telemetry does not. Report194/video192/audit193 completed; no repair/PPO/holdout admission. Implicit-PD torque telemetry is approximate and cached-state read order must be considered; no library/control/reward/criteria/material change. See closure_record_195.json and handoff_update_195.txt.\n')
    st = read(status)
    st.update(utc=now,updated_utc=now,phase='substep_diagnostic_complete_global_exact_failed',
        task_complete=False,repair_complete=False,physical_repair_complete=False,
        main_training_started=False,ppo_started=False,fresh_holdout_started=False,holdout_started=False,
        policy_promoted=False,physical_jobs_running=False,model_jobs_running=False,audit_jobs_running=False,
        latest_running_candidate=None,model_job=None,latest_candidate='touchdown_probe_178',
        latest_physical_candidate='touchdown_probe_178',latest_physical_admission_passed=False,
        latest_report=str(D/'progress_report_194'),latest_diagnostic=str(D/'substep_diagnostic_189'),
        latest_diagnosis=str(D/'substep_diagnosis_v3_191'),latest_diagnostic_audit=str(D/'substep_audit_v2_190/independent_audit.json'),
        latest_diagnostic_completed=True,latest_diagnostic_checks_passed=False,latest_diagnostic_exact_prefix_replay=False,
        latest_diagnostic_accounting=accounting,latest_diagnostic_source_hashes_unchanged=True,
        latest_diagnostic_internally_exact_raw_checks=4016,
        effort_telemetry_limitation=audit['effort_semantics'],
        prior_recheck=str(priorpath),initial_prior_hash_check=priorpath.name,latest_preservation=str(mf),
        completed_evidence_manifest=str(mf),completed_evidence_count=len(protected),
        latest_handoff=str(D/'handoff_update_195.txt'),latest_video=str(D/'all_attempts_video_192/all_attempts.mp4'),
        latest_video_audit=str(D/'all_attempts_video_audit_193/independent_audit.json'),
        video_ready_locally=True,video_duration_s=58.4,video_complete_trials=792,video_successes=461,
        video_failures=331,video_recorded_prefixes=96,video_aborted_case_attempts=72,
        local_report_original_hashes_verified=17,local_supplemental_hashes_verified=0,
        local_report_visually_inspected=True,latest_round_new_full20s_trials=0,
        latest_round_new_diagnostic_prefixes=24,latest_round_new_active_ticks=15600,
        latest_round_initial_settling_ticks=600,latest_round_optimizer_updates=0,
        best_development_clean=64,best_development_total=72,remaining_slipping=8,remaining_housing=2,
        notes=notes)
    status.write_text(json.dumps(st,indent=2))
    closure = dict(utc=now,task_complete=False,repair_complete=False,new_active_physics_ticks=15600,
        new_initial_settling_ticks=600,new_indexed_warmup_ticks=0,new_optimizer_updates=0,new_full20s_trials=0,
        new_diagnostic_prefixes=24,new_shared5ms_steps=2600,holdout_started=False,
        diagnostic='substep_diagnostic_189',diagnostic_completed=True,original_runtime_exit_failed_after_saved_outputs=True,
        diagnostic_checks_passed=False,exact_prefix_replay=False,physical_admission_passed=False,
        internally_exact_raw_checks=4016,frozen_sources_checked=43,prior_recheck=prior,
        completed_evidence_manifest=str(mf),protected_completed_files=len(protected),manifest_sha256=sha(mf),
        physical_workers_running=False,model_workers_running=False,audit_workers_running=False,
        report='progress_report_194',video='all_attempts_video_192/all_attempts.mp4',
        video_summary={k:video[k] for k in ['full_trials','successes','failures','recorded_prefixes','aborted_prefixes_disclosed','event_frames_checked','source_runs']},
        local_review=review,handoff_sha256=sha(D/'handoff_update_195.txt'),next=notes)
    (D/'closure_record_195.json').write_text(json.dumps(closure,indent=2))
    print(json.dumps(dict(prior_files=checks[0]['checked'],previous_evidence=checks[1]['checked'],
        completed_evidence=len(protected),workers_running=False,diagnostic_global_exact=False,task_complete=False)),flush=True)

if __name__ == '__main__':
    main()
