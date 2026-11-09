"""Preserve completed150 physical evidence and full video; no experiment rerun."""
from pathlib import Path
import datetime,hashlib,json,re,shutil,subprocess
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');R=Path('/home/kadir/humanoid-sitting')
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4194304),b''):h.update(b)
    return h.hexdigest()
def main():
    assert not (D/'closure_record_158.json').exists()
    final=read(D/'balanced_finalization_status_155.json');assert final['phase']=='complete' and final['sources_unchanged']
    a=read(D/'balanced_force_probe_150/independent_balanced_audit_151.json');v=read(D/'balanced_force_probe_150/independent_dual_view_audit_152.json');video=read(D/'all_attempts_video_audit_v2_154/independent_audit.json');report=read(D/'progress_report_157/summary.json')
    assert a['software_audit_passed'] and v['software_audit_passed'] and video['passed']
    assert report['remaining_target_cases'] and not report['physical_admission_passed']
    assert video['full_trials']==744
    for name in ['balanced_force_probe_150/revision.json','balanced_round_sources_150.json','balanced_finalization_sources_155.json']:
        data=read(D/name);sources=data.get('sources',data)
        for p,h in sources.items():assert sha(p)==h,p
    folders=['progress_report_157','all_attempts_video_v2_153','all_attempts_video_audit_v2_154','balanced_diagnosis_156','balanced_force_paths_144','balanced_force_audit_146','balanced_filter_audit_147','balanced_runtime_preparation_v2_149']
    for folder in folders:
        for p,h in read(D/folder/'SHA256.json').items():assert sha(D/folder/p)==h,p
    for p,h in read(D/'progress_report_157/REVIEW_SHA256.json').items():assert sha(D/'progress_report_157'/p)==h,p
    review=read(D/'progress_report_157_local_review.json');assert review['browser_visually_inspected'] and review['video_frames_visually_inspected'] and review['local_hashes_verified']
    processes=subprocess.run(['ps','-eo','pid,args'],capture_output=True,text=True,check=True).stdout.splitlines()
    workers=[s for s in processes if re.search(r'python\S* .*scripts/(run|refine|optimize|audit|diagnose|schedule|check|bound|screen|render|verify|finalize)_low_seat_\S+\.py',s)]
    assert not workers,workers
    checks=[]
    for name in ['prior_hashes.json','completed_probes_preservation_142.json','supplemental_preservation_142.json']:
        m=read(D/name);bad=[p for p,h in m.items() if not Path(p).exists() or sha(p)!=h]
        checks.append(dict(manifest=name,checked=len(m),mismatches=bad,passed=not bad));assert not bad,bad[:5]
    now=datetime.datetime.now(datetime.timezone.utc).isoformat();prior=dict(utc=now,passed=True,checks=checks)
    priorpath=D/'prior_recheck_after_balanced_158.json';priorpath.write_text(json.dumps(prior,indent=2))
    status=D/'status.json';doc=R/'docs/LOW_SEAT_CONTACT_STATUS.md';protocol=R/'docs/LOW_SEAT_CONTACT_PROTOCOL.md'
    for p,name in [(status,'status_before_round_158.json'),(doc,'handoff_before_round_158.md'),(protocol,'protocol_before_round_158.md'),(D/'status_balanced_force_probe_150.json','status_balanced_before_158.json')]:shutil.copy2(p,D/name)
    ss=read(D/'status_balanced_force_probe_150.json');ss.update(phase='complete_rejected',audit_completed=True,software_audit_passed=True,exact_preservation_gate_passed=a['exact_preservation_gate_passed'],physical_admission_passed=False,repair_complete=False,updated_utc=now);(D/'status_balanced_force_probe_150.json').write_text(json.dumps(ss,indent=2))
    protected=dict(read(D/'completed_probes_preservation_142.json'));protected.update(read(D/'supplemental_preservation_142.json'))
    for p in D.iterdir():
        match=re.search(r'_(\d+)(?:\.py)?(?:\.[^.]*)?$',p.name)
        if not match or not 143<=int(match[1])<=158 or p.name in ['closure_158.log','completed_probes_preservation_158.json','closure_record_158.json']:continue
        if p.is_dir():
            for f in p.rglob('*'):
                if f.is_file():protected[str(f)]=sha(f)
        elif p.is_file():protected[str(p)]=sha(p)
    for p in (R/'scripts').glob('*.py'):
        match=re.search(r'_(\d+)\.py$',p.name)
        if match and 143<=int(match[1])<=158:protected[str(p)]=sha(p)
    protected[str(D/'progress_report_157_local_review.json')]=sha(D/'progress_report_157_local_review.json')
    mf=D/'completed_probes_preservation_158.json';mf.write_text(json.dumps(protected,indent=2));assert all(sha(p)==h for p,h in protected.items())
    notes=(D/'handoff_update_158.txt').read_text();doc.write_text(notes+'\n\nEarlier preserved entries follow.\n\n'+doc.read_text())
    with protocol.open('a') as f:f.write('\n\nRound158 closure: force-allocation probe150 is complete; physical admission remains false. Audit, diagnosis and updated744-trial complete video finished. No retention/PPO/holdout followed. All physical/material criteria unchanged. See closure_record_158.json and handoff_update_158.txt.\n')
    st=read(status);st.update(latest_candidate='balanced_force_probe_150',latest_candidate_outcome=a['candidate'],latest_independent_audit_passed=a['passed'],latest_exact_preservation_gate_passed=a['exact_preservation_gate_passed'],latest_audit_status='balanced_finalization_status_155.json',video_complete_trials=video['full_trials'],video_successes=video['successes'],video_failures=video['failures'],video_duration_s=58.4,video_ready_locally=True,local_report_original_hashes_verified=20,local_supplemental_hashes_verified=3,new_physical_ticks_since43=48000,new_initial_settling_ticks_since43=1200,new_optimizer_updates_since43=0,utc=now,updated_utc=now,phase='balanced_force_probe_rejected_audits_video_complete',task_complete=False,repair_complete=False,physical_repair_complete=False,main_training_started=False,ppo_started=False,fresh_holdout_started=False,holdout_started=False,policy_promoted=False,physical_jobs_running=False,model_jobs_running=False,audit_jobs_running=False,latest_running_candidate=None,model_job=None,
        latest_report=str(D/'progress_report_157'),latest_diagnostic=str(D/'balanced_diagnosis_156/recorded_diagnosis.json'),latest_diagnosis=str(D/'balanced_diagnosis_156'),latest_diagnostic_audit=str(D/'balanced_force_probe_150/independent_dual_view_audit_152.json'),latest_independent_audit='balanced_force_probe_150/independent_balanced_audit_151.json',latest_software_audit_passed=True,latest_physical_candidate='balanced_force_probe_150',latest_physical_exact_preservation_passed=a['exact_preservation_gate_passed'],latest_physical_admission_passed=False,latest_physical_outcomes=a['candidate'],best_development_clean=64,best_development_total=72,remaining_slipping=8,remaining_housing=2,prior_recheck=str(priorpath),initial_prior_hash_check=priorpath.name,latest_preservation=str(mf),completed_evidence_manifest=str(mf),completed_evidence_count=len(protected),latest_handoff=str(D/'handoff_update_158.txt'),latest_video=str(D/'all_attempts_video_v2_153/all_attempts.mp4'),latest_video_audit=str(D/'all_attempts_video_audit_v2_154/independent_audit.json'),local_report_visually_inspected=True,notes=notes)
    status.write_text(json.dumps(st,indent=2))
    closure=dict(utc=now,task_complete=False,repair_complete=False,new_active_physics_ticks=24000,new_initial_settling_ticks=600,new_indexed_warmup_ticks=0,new_optimizer_updates=0,new_full20s_trials=24,holdout_started=False,physical_candidate='balanced_force_probe_150',candidate_rejected=True,software_audit_passed=True,exact_preservation_gate_passed=a['exact_preservation_gate_passed'],physical_admission_passed=False,candidate_outcomes=a['candidate'],prior_recheck=prior,completed_evidence_manifest=str(mf),protected_completed_files=len(protected),manifest_sha256=sha(mf),new_frozen_sources_rechecked=True,physical_workers_running=False,model_workers_running=False,audit_workers_running=False,report='progress_report_157',video='all_attempts_video_v2_153/all_attempts.mp4',video_summary={k:video[k] for k in ['full_trials','successes','failures','event_frames_checked','source_runs']},local_review=review,handoff_sha256=sha(D/'handoff_update_158.txt'),next=notes)
    (D/'closure_record_158.json').write_text(json.dumps(closure,indent=2));print(json.dumps(dict(prior_files=checks[0]['checked'],previous_evidence=checks[1]['checked'],completed_evidence=len(protected),workers_running=False,task_complete=False)),flush=True)
if __name__=='__main__':main()
