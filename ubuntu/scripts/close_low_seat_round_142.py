"""Preserve rejected physical134, audits and complete video; no experiment rerun."""
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
    for name in ['closure_record_142.json','completed_probes_preservation_142.json','status_before_round_142.json','handoff_before_round_142.md']:
        assert not (D/name).exists(),name
    final=read(D/'fixed_finalization_status_139.json');assert final['phase']=='complete' and final['sources_unchanged']
    a=read(D/'fixed_foot_probe_134/independent_fixed_audit_135.json')
    assert a['software_audit_passed'] and not a['passed'] and not a['exact_preservation_gate_passed'] and not a['physical_admission_passed']
    v=read(D/'fixed_foot_probe_134/independent_dual_view_audit_136.json');assert v['software_audit_passed']
    video=read(D/'all_attempts_video_audit_138/independent_audit.json');assert video['passed'] and video['full_trials']==720 and video['event_frames_checked']==488
    for p,h in read(D/'fixed_foot_probe_134/revision.json')['sources'].items():assert sha(p)==h,p
    for p,h in read(D/'fixed_finalization_sources_139.json').items():assert sha(p)==h,p
    for folder in ['progress_report_141','all_attempts_video_137','all_attempts_video_audit_138','fixed_diagnosis_140']:
        for p,h in read(D/folder/'SHA256.json').items():assert sha(D/folder/p)==h,p
    review=read(D/'progress_report_141_local_review.json');assert review['browser_visually_inspected'] and review['video_frames_visually_inspected']
    processes=subprocess.run(['ps','-eo','pid,args'],capture_output=True,text=True,check=True).stdout.splitlines()
    workers=[v for v in processes if re.search(r'python\S* .*scripts/(run|refine|optimize|audit|diagnose|schedule|check|bound|screen|render|verify|finalize)_low_seat_\S+\.py',v)]
    assert not workers,workers
    checks=[]
    for name in ['prior_hashes.json','completed_probes_preservation_125.json']:
        m=read(D/name);bad=[p for p,h in m.items() if not Path(p).exists() or sha(p)!=h]
        checks.append(dict(manifest=name,checked=len(m),mismatches=bad,passed=not bad));assert not bad,bad[:5]
    now=datetime.datetime.now(datetime.timezone.utc).isoformat();prior=dict(utc=now,passed=True,checks=checks)
    priorpath=D/'prior_recheck_after_fixed_probe_142.json';priorpath.write_text(json.dumps(prior,indent=2))
    status=D/'status.json';doc=R/'docs/LOW_SEAT_CONTACT_STATUS.md';protocol=R/'docs/LOW_SEAT_CONTACT_PROTOCOL.md'
    shutil.copy2(status,D/'status_before_round_142.json');shutil.copy2(doc,D/'handoff_before_round_142.md');shutil.copy2(protocol,D/'protocol_before_round_142.md')
    s134=D/'status_fixed_foot_probe_134.json';shutil.copy2(s134,D/'status_fixed_foot_probe_before_142.json')
    s=read(s134);s.update(phase='complete_rejected',audit_completed=True,software_audit_passed=True,exact_preservation_gate_passed=False,physical_admission_passed=False,repair_complete=False,updated_utc=now)
    s134.write_text(json.dumps(s,indent=2))
    protected=dict(read(D/'completed_probes_preservation_125.json'))
    for p in D.iterdir():
        match=re.search(r'_(\d+)(?:\.[^.]*)?$',p.name)
        if not match or not 126<=int(match[1])<=142 or p.name in ['closure_142.log','completed_probes_preservation_142.json','closure_record_142.json']:continue
        if p.is_dir():
            for f in p.rglob('*'):
                if f.is_file():protected[str(f)]=sha(f)
        elif p.is_file():protected[str(p)]=sha(p)
    for p in (R/'scripts').glob('*.py'):
        match=re.search(r'_(\d+)\.py$',p.name)
        if match and 126<=int(match[1])<=142:protected[str(p)]=sha(p)
    reviewpath=D/'progress_report_141_local_review.json';protected[str(reviewpath)]=sha(reviewpath)
    mf=D/'completed_probes_preservation_142.json';mf.write_text(json.dumps(protected,indent=2))
    assert all(sha(p)==h for p,h in protected.items())
    notes=(D/'handoff_update_142.txt').read_text();doc.write_text(notes+'\n\nEarlier preserved entries follow.\n\n'+doc.read_text())
    with protocol.open('a') as f:f.write('\n\nRound142 closure: physical134 is rejected (four added falls/contacts, no clean gains). All physical and exact-preservation gates remain false; no retention/PPO/holdout follows. Independent135/136 and complete-video138 audits finished. No definitions or physical/material criteria changed. See closure_record_142.json and handoff_update_142.txt.\n')
    st=read(status);st.update(utc=now,updated_utc=now,phase='fixed_foot_probe_rejected_audits_video_complete',task_complete=False,
        repair_complete=False,physical_repair_complete=False,main_training_started=False,ppo_started=False,fresh_holdout_started=False,holdout_started=False,
        policy_promoted=False,physical_jobs_running=False,model_jobs_running=False,audit_jobs_running=False,latest_running_candidate=None,model_job=None,
        latest_report=str(D/'progress_report_141'),latest_diagnostic=str(D/'fixed_diagnosis_140/recorded_diagnosis.json'),latest_diagnosis=str(D/'fixed_diagnosis_140'),
        latest_diagnostic_audit=str(D/'fixed_foot_probe_134/independent_dual_view_audit_136.json'),latest_independent_audit='fixed_foot_probe_134/independent_fixed_audit_135.json',
        latest_software_audit_passed=True,latest_physical_candidate='fixed_foot_probe_134',latest_physical_exact_preservation_passed=False,
        latest_physical_admission_passed=False,latest_physical_outcomes=a['candidate'],best_development_clean=64,best_development_total=72,
        remaining_slipping=8,remaining_housing=2,prior_recheck=str(priorpath),initial_prior_hash_check=priorpath.name,
        latest_preservation=str(mf),completed_evidence_manifest=str(mf),completed_evidence_count=len(protected),latest_handoff=str(D/'handoff_update_142.txt'),
        latest_video=str(D/'all_attempts_video_137/all_attempts.mp4'),latest_video_audit=str(D/'all_attempts_video_audit_138/independent_audit.json'),
        local_report_original_hashes_verified=review['original_report_hashes_verified'],local_report_visually_inspected=True,
        notes='Probe134 complete and rejected:18stable14clean10slip6unwanted/physical; four added falls, no clean gains/losses. First pause3.08–3.26s with pelvis~26cm above seat; holding did not stabilize. No requested unloading or parent action before early failures. Audits and full720-trial58.4s video complete. All8physicalfailures remain; preserve64. NoPPO/holdout/promotion.')
    status.write_text(json.dumps(st,indent=2))
    closure=dict(utc=now,task_complete=False,repair_complete=False,new_active_physics_ticks=24000,new_initial_settling_ticks=600,new_indexed_warmup_ticks=0,
        new_optimizer_updates=0,new_full20s_trials=24,holdout_started=False,physical_candidate='fixed_foot_probe_134',candidate_rejected=True,
        software_audit_passed=True,exact_preservation_gate_passed=False,physical_admission_passed=False,candidate_outcomes=a['candidate'],
        prior_recheck=prior,completed_evidence_manifest=str(mf),protected_completed_files=len(protected),manifest_sha256=sha(mf),
        new_frozen_sources_rechecked=True,physical_workers_running=False,model_workers_running=False,audit_workers_running=False,
        report='progress_report_141',video='all_attempts_video_137/all_attempts.mp4',video_audit=video,local_review=review,
        handoff_sha256=sha(D/'handoff_update_142.txt'),next='Do not rerun134. Diagnose/control dynamic balance, actual foot-load realization and stable pause before a new complete contact transition; no blind sweeps. Preserve64/remove8 across72 matched full20s before PPO. Keep fresh holdout closed and all-attempt video complete.')
    (D/'closure_record_142.json').write_text(json.dumps(closure,indent=2))
    print(json.dumps(dict(prior_files=checks[0]['checked'],previous_evidence=checks[1]['checked'],completed_evidence=len(protected),workers_running=False,task_complete=False)),flush=True)
if __name__=='__main__':main()
