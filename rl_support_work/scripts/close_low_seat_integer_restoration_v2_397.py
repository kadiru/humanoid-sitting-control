"""Seal completed392–396 evidence after395 exits; preserve active-audit7285-file chain as well as prior evidence. Unexecuted v1 draft retained."""
from pathlib import Path
import datetime, hashlib, json, re, shutil, subprocess

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
    assert not (D/'closure_record_397.json').exists()
    folders = ['integer_restoration_preparation_392','integer_restoration_preparation_audit_393','integer_restoration_path_394','integer_restoration_path_audit_395','progress_report_396']
    original_entries = 0
    for folder in folders:
        manifest = read(D/folder/'SHA256.json')
        for p, h in manifest.items():
            assert sha(D/folder/p) == h, (folder, p)
        original_entries += len(manifest)
    for folder in ['integer_restoration_preparation_audit_393','integer_restoration_path_audit_395']:
        assert read(D/folder/'independent_audit.json')['passed']
    x = read(D/'integer_restoration_exit_394.json')
    assert x['observed_exit_code'] == 0 and x['model_results_complete'] and x['source_hashes_intact']
    execution = read(D/'integer_restoration_postprocessing_396.json')
    assert execution['audit_exit_code'] == 0 and execution['report_exit_code'] == 0
    r = read(D/'integer_restoration_path_394/results.json'); op = r['optimizer']; z = r['dense']
    assert not r['physical_admission'] and not z['actual_filter_or_dynamic_check']
    review = read(D/'progress_report_396_local_review.json')
    assert all(review[k] for k in ['local_hashes_verified', 'visual_layout_inspected',
                                  'browser_visually_inspected', 'embedded_video_loaded'])
    for name in review.get('supplemental_manifests', []):
        for p, h in read(D/'progress_report_396'/name).items():
            assert sha(D/'progress_report_396'/p) == h
    process_text = subprocess.run(['ps','-eo','pid,args'], capture_output=True, text=True, check=True).stdout
    workers = [s for s in process_text.splitlines() if re.search(
        r'^\s*\d+\s+\S*python\S* .*scripts/(run|build|bound|audit|refine|optimize|diagnose|report|review|schedule|render|verify|finalize|launch|prepare|check|screen|assess|evaluate)_low_seat_\S+\.py', s)]
    assert not workers, workers
    checks = []
    for name in ['prior_hashes.json', 'integer_restoration_evidence_preservation_395.json', 'supplemental_preservation_391.json']:
        data = read(D/name)
        bad = [p for p,h in data.items() if not Path(p).exists() or sha(p) != h]
        assert not bad, bad[:5]
        checks.append(dict(manifest=name, checked=len(data), passed=True, mismatches=[]))
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    prior = dict(utc=now, passed=True, checks=checks)
    priorpath = D/'prior_recheck_after_integer_restoration_397.json'
    priorpath.write_text(json.dumps(prior, indent=2))
    status = D/'status.json'; doc = R/'docs/LOW_SEAT_CONTACT_STATUS.md'; protocol = R/'docs/LOW_SEAT_CONTACT_PROTOCOL.md'
    for p, name in [(status,'status_before_round_397.json'), (doc,'handoff_before_round_397.md'), (protocol,'protocol_before_round_397.md')]:
        assert not (D/name).exists()
        shutil.copy2(p, D/name)
    notes = (D/'handoff_update_397.txt').read_text()
    protected = read(D/'integer_restoration_evidence_preservation_395.json')
    protected.update(read(D/'supplemental_preservation_391.json'))
    for p in D.iterdir():
        match = re.search(r'_(\d+)(?:\.[^.]*)?$', p.name)
        if not match or not 392 <= int(match[1]) <= 397 or p.name in ['closure_397.log','closure_record_397.json','completed_probes_preservation_397.json']:
            continue
        for f in (p.rglob('*') if p.is_dir() else [p]):
            if f.is_file() and not f.is_symlink():
                protected[str(f)] = sha(f)
    for p in (R/'scripts').glob('*.py'):
        match = re.search(r'_(\d+)\.py$', p.name)
        if match and 392 <= int(match[1]) <= 397:
            protected[str(p)] = sha(p)
    protected[str(D/'progress_report_396_local_review.json')] = sha(D/'progress_report_396_local_review.json')
    mf = D/'completed_probes_preservation_397.json'
    mf.write_text(json.dumps(protected, indent=2))
    assert all(sha(p) == h for p,h in protected.items())
    doc.write_text(notes+'\n\nEarlier preserved entries follow.\n\n'+doc.read_text())
    with protocol.open('a') as f:
        f.write('\n\nRound397: preparation392 and independent393 preceded ONE bounded mixed-integer tangent restoration394 of the complete115-knot geometry/COM/force/timing path, with five integer event budgets. Independent395 checks the linear primal witness and all nonlinear geometry/virtual-work/constraints/clock rows. The linear subproblem succeeds but nonlinear gates fail; no polishing, restart or physical admission. Actual float32 schedule/intermediate geometry/runtime/both-placement/parent/physics/PPO remain unstarted. All64 protected/all8 unresolved; no gains/quiet/material/physical criteria changed. Report396 reviewed; video307 unchanged.\n')
    s = read(status)
    s.update(utc=now, updated_utc=now, phase='offline_integer_restoration_path_394_complete_repair_unresolved',
        task_complete=False, repair_complete=False, main_training_started=False, ppo_started=False,
        fresh_holdout_started=False, holdout_started=False, policy_promoted=False,
        physical_jobs_running=False, model_jobs_running=False, audit_jobs_running=False,
        latest_diagnostic_running=False, latest_running_candidate=None, latest_running_diagnostic=None, latest_active_snapshot=None, latest_active_run=None,
        latest_report=str(D/'progress_report_396'), latest_diagnosis=str(D/'integer_restoration_preparation_392'),
        latest_offline_path=str(D/'integer_restoration_path_394'), latest_offline_paths=[str(D/'integer_restoration_path_394')],
        latest_offline_bound=None, latest_round_force_lp_started=False,
        latest_offline_audit=str(D/'integer_restoration_path_audit_395/independent_audit.json'),
        latest_offline_audit_passed=True, latest_offline_command_arithmetic_passed=True,
        latest_offline_time_gate_passed=z['guarded_timing_constraints_passed'],
        latest_offline_geometry_passed=z['model_geometry_passed'],
        latest_offline_static_commands_passed=z['model_command_passed'],
        latest_offline_nlp_passed=op['numerical_constraints_passed'],
        latest_offline_gate_scope='Completed integer restoration394; independent395 audit is arithmetic only. No physical admission. See exact signed final constraints.',
        latest_offline_screen_scope='One complete115-knot nominal support/COM/force/timing path and five integer event budgets, chosen by a bounded tangent restoration. Linear witness success does not imply nonlinear or ideal-clock feasibility. Saved final gates are not actual float32 execution, support or measured quiet settling.',
        latest_handoff=str(D/'handoff_update_397.txt'), latest_preservation=str(mf),
        completed_evidence_manifest=str(mf), completed_evidence_count=len(protected), prior_recheck=str(priorpath),
        latest_round_new_full20s_trials=0, latest_round_new_diagnostic_prefixes=0,
        latest_round_new_aborted_case_attempts=0, latest_round_new_active_ticks=0,
        latest_round_initial_settling_ticks=0, latest_round_optimizer_updates=0,
        latest_round_offline_nonlinear_iterations=0, latest_round_offline_milp_calls=1, latest_round_offline_milp_nodes=op['iterations'], latest_round_command_ticks_per_device=0,
        latest_round_hover_guard_exercised=False, notes=notes)
    status.write_text(json.dumps(s, indent=2))
    c = dict(utc=now, task_complete=False, repair_complete=False, new_active_physics_ticks=0,
        new_initial_settling_ticks=0, new_optimizer_updates=0, new_full20s_trials=0,
        new_diagnostic_prefixes=0, new_aborted_case_attempts=0, new_command_schedule_ticks=0,
        holdout_started=False, physical_admission_passed=False, best_verified_controller='current_retention_09',
        best_development_clean=64, latest_physical_candidate='roll_feedback_probe_273', offline_optimizer=op,
        offline_geometry_passed=z['model_geometry_passed'], offline_static_commands_passed=z['model_command_passed'],
        actual_schedule_executed=False, hover_guard_exercised=False, observed_solver_process_exit=x,
        postprocessing_exit_evidence=execution, independent_audit_passed=True, prior_recheck=prior,
        completed_evidence_manifest=str(mf), protected_completed_files=len(protected), manifest_sha256=sha(mf),
        physical_workers_running=False, model_workers_running=False, audit_workers_running=False,
        report='progress_report_396/findings.html', video='all_attempts_video_307/all_attempts.mp4',
        video_unchanged=True, original_folder_manifests=len(folders), original_manifest_entries=original_entries,
        local_review=review, next=notes)
    (D/'closure_record_397.json').write_text(json.dumps(c, indent=2))
    print(json.dumps(dict(utc=now, prior_checks=checks, protected_completed_files=len(protected),
                         manifest_sha256=sha(mf), workers_running=False, task_complete=False)), flush=True)

if __name__ == '__main__':
    main()
