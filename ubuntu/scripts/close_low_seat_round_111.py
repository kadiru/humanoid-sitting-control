"""Preserve completed offline98–110 and update mutable handoffs honestly."""
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
    for name in ['closure_record_111.json','completed_probes_preservation_111.json','status_before_round_111.json','handoff_before_round_111.md']:
        assert not (D/name).exists(),name
    for name in ['stance_com_path_audit_v2_99','contact_force_authority_audit_105','contact_load_filter_audit_107','contact_filtered_schedule_audit_109']:
        assert read(D/name/'independent_audit.json')['passed']
    for p,h in read(D/'pending_sources_98.json').items():assert sha(p)==h,p
    processes=subprocess.run(['ps','-eo','pid,args'],capture_output=True,text=True,check=True).stdout.splitlines()
    workers=[v for v in processes if re.search(r'python\S* .*scripts/(run|refine|optimize|audit|diagnose|schedule)_low_seat_\S+\.py',v)]
    assert not workers,workers
    checks=[]
    for name in ['prior_hashes.json','completed_probes_preservation_101.json']:
        m=read(D/name);bad=[p for p,h in m.items() if not Path(p).exists() or sha(p)!=h]
        checks.append(dict(manifest=name,checked=len(m),mismatches=bad,passed=not bad));assert not bad,bad[:5]
    prior=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,checks=checks)
    priorpath=D/'prior_recheck_after_force_filter_111.json';priorpath.write_text(json.dumps(prior,indent=2))
    status=D/'status.json';doc=R/'docs/LOW_SEAT_CONTACT_STATUS.md'
    shutil.copy2(status,D/'status_before_round_111.json');shutil.copy2(doc,D/'handoff_before_round_111.md')
    protected=dict(read(D/'completed_probes_preservation_101.json'))
    for p in D.iterdir():
        match=re.search(r'_(\d+)(?:\.[^.]*)?$',p.name)
        if not match or not 98<=int(match[1])<=111 or p.name in ['closure_111.log','completed_probes_preservation_111.json','closure_record_111.json']:continue
        if p.is_dir():
            for f in p.rglob('*'):
                if f.is_file():protected[str(f)]=sha(f)
        elif p.is_file():protected[str(p)]=sha(p)
    for p in (R/'scripts').glob('*.py'):
        match=re.search(r'_(\d+)\.py$',p.name)
        if match and 98<=int(match[1])<=111:protected[str(p)]=sha(p)
    review=D/'progress_report_110_local_review.json';assert review.exists();protected[str(review)]=sha(review)
    mf=D/'completed_probes_preservation_111.json';mf.write_text(json.dumps(protected,indent=2))
    assert all(sha(p)==h for p,h in protected.items())
    notes=(D/'handoff_update_111.txt').read_text();doc.write_text(notes+'\n\nEarlier preserved entries follow.\n\n'+doc.read_text())
    now=datetime.datetime.now(datetime.timezone.utc).isoformat();st=read(status)
    st.update(utc=now,updated_utc=now,phase='offline_force_filter_checks_complete_repair_unresolved',
              task_complete=False,repair_complete=False,physical_repair_complete=False,main_training_started=False,ppo_started=False,
              fresh_holdout_started=False,holdout_started=False,policy_promoted=False,physical_jobs_running=False,
              model_jobs_running=False,audit_jobs_running=False,latest_running_candidate=None,model_job=None,
              latest_report=str(D/'progress_report_110'),latest_completed_model=str(D/'contact_load_sharing_106/results.json'),
              latest_model_preparation=str(D/'contact_filtered_schedule_108/results.json'),
              latest_diagnostic=str(D/'contact_force_authority_104/results.json'),latest_diagnosis=str(D/'contact_force_authority_104'),
              latest_diagnostic_audit=str(D/'contact_filtered_schedule_audit_109/independent_audit.json'),
              latest_independent_audit='contact_filtered_schedule_audit_109',latest_software_audit_passed=True,
              prior_recheck=str(priorpath),initial_prior_hash_check=priorpath.name,latest_preservation=str(priorpath),
              completed_evidence_manifest=str(mf),completed_evidence_count=len(protected),
              latest_handoff=str(D/'handoff_update_111.txt'),local_report_original_hashes_verified=12,
              local_supplemental_hashes_verified=0,local_report_visually_inspected=True,
              latest_physical_candidate='dual_view_approach_40',latest_physical_exact_preservation_passed=False,
              latest_offline_command_arithmetic_passed=True,latest_offline_time_gate_passed=False,
              notes='All offline jobs through109 complete. Command/load-sharing106 passes static command limits but fails timing; constructed unchanged-filter108 schedule25.10s is exactly CPU-replayed109 but fails13.5s. No new physical/PPO/holdout; latest physical40 gates remain failed. Preserve64, remove8 beforePPO.')
    status.write_text(json.dumps(st,indent=2))
    closure=dict(utc=now,task_complete=False,repair_complete=False,new_physical_ticks=0,new_optimizer_updates=0,
                 holdout_started=False,prior_recheck=prior,completed_evidence_manifest=str(mf),protected_completed_files=len(protected),
                 manifest_sha256=sha(mf),pending98_sources_rechecked=True,physical_workers_running=False,
                 model_workers_running=False,audit_workers_running=False,report='progress_report_110',
                 local_review=read(review),handoff_sha256=sha(D/'handoff_update_111.txt'),
                 next='Do not repeat98–109. Jointly plan command headroom/load transfer/transition timing under unchanged physical criteria; static force sharing and slowing this fixed path do not admit physics. All72 matched physical gates remain beforePPO. Video59 current; no new motion.')
    (D/'closure_record_111.json').write_text(json.dumps(closure,indent=2))
    print(json.dumps(dict(prior_files=checks[0]['checked'],previous_evidence=checks[1]['checked'],completed_evidence=len(protected),workers_running=False,task_complete=False)),flush=True)
if __name__=='__main__':main()
