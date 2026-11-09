"""Preserve completed offline112–124 and update mutable handoffs honestly."""
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
    for name in ['closure_record_125.json','completed_probes_preservation_125.json','status_before_round_125.json','handoff_before_round_125.md']:
        assert not (D/name).exists(),name
    for name in ['contact_command_bound_audit_113','contact_headroom_audit_115','contact_headroom_trust_audit_v2_117']+[f'fixed_foot_contact_audit_{c}_119' for c in [0,2,6,10]]+[f'staged_fixed_foot_audit_{c}_121' for c in [0,6]]+[f'staged_fixed_foot_filter_audit_v2_{c}_123' for c in [0,6]]:
        assert read(D/name/'independent_audit.json')['passed']
    for revision in ['command_bound_revision_112.json','headroom_geometry_revision_114.json','headroom_trust_revision_116.json','headroom_trust_revision_v2_116.json','fixed_foot_contact_revision_118.json','staged_fixed_foot_revision_120.json','staged_fixed_filter_revision_122.json']:
        for p,h in read(D/revision)['source_sha256'].items():assert sha(p)==h,p
    processes=subprocess.run(['ps','-eo','pid,args'],capture_output=True,text=True,check=True).stdout.splitlines()
    workers=[v for v in processes if re.search(r'python\S* .*scripts/(run|refine|optimize|audit|diagnose|schedule|check|bound|screen)_low_seat_\S+\.py',v)]
    assert not workers,workers
    checks=[]
    for name in ['prior_hashes.json','completed_probes_preservation_111.json']:
        m=read(D/name);bad=[p for p,h in m.items() if not Path(p).exists() or sha(p)!=h]
        checks.append(dict(manifest=name,checked=len(m),mismatches=bad,passed=not bad));assert not bad,bad[:5]
    prior=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,checks=checks)
    priorpath=D/'prior_recheck_after_fixed_foot_125.json';priorpath.write_text(json.dumps(prior,indent=2))
    status=D/'status.json';doc=R/'docs/LOW_SEAT_CONTACT_STATUS.md'
    shutil.copy2(status,D/'status_before_round_125.json');shutil.copy2(doc,D/'handoff_before_round_125.md')
    protected=dict(read(D/'completed_probes_preservation_111.json'))
    for p in D.iterdir():
        match=re.search(r'_(\d+)(?:\.[^.]*)?$',p.name)
        if not match or not 112<=int(match[1])<=125 or p.name in ['closure_125.log','completed_probes_preservation_125.json','closure_record_125.json']:continue
        if p.is_dir():
            for f in p.rglob('*'):
                if f.is_file():protected[str(f)]=sha(f)
        elif p.is_file():protected[str(p)]=sha(p)
    for p in (R/'scripts').glob('*.py'):
        match=re.search(r'_(\d+)\.py$',p.name)
        if match and 112<=int(match[1])<=125:protected[str(p)]=sha(p)
    review=D/'progress_report_124_local_review.json';assert review.exists();protected[str(review)]=sha(review)
    mf=D/'completed_probes_preservation_125.json';mf.write_text(json.dumps(protected,indent=2))
    assert all(sha(p)==h for p,h in protected.items())
    notes=(D/'handoff_update_125.txt').read_text();doc.write_text(notes+'\n\nEarlier preserved entries follow.\n\n'+doc.read_text())
    now=datetime.datetime.now(datetime.timezone.utc).isoformat();st=read(status)
    st.update(utc=now,updated_utc=now,phase='two_offline_fixed_foot_paths_pass_physical_repair_unresolved',
              task_complete=False,repair_complete=False,physical_repair_complete=False,main_training_started=False,ppo_started=False,
              fresh_holdout_started=False,holdout_started=False,policy_promoted=False,physical_jobs_running=False,
              model_jobs_running=False,audit_jobs_running=False,latest_running_candidate=None,model_job=None,
              latest_report=str(D/'progress_report_124'),latest_completed_model=str(D/'staged_fixed_foot_contact_120'),
              latest_model_preparation=str(D/'staged_fixed_foot_filtered_6_122/results.json'),
              latest_diagnostic=str(D/'contact_command_bound_112/results.json'),latest_diagnosis=str(D/'contact_command_bound_112'),
              latest_diagnostic_audit=str(D/'staged_fixed_foot_filter_audit_v2_6_123/independent_audit.json'),
              latest_independent_audit='staged_fixed_foot_filter_audit_v2_6_123',latest_software_audit_passed=True,
              prior_recheck=str(priorpath),initial_prior_hash_check=priorpath.name,latest_preservation=str(priorpath),
              completed_evidence_manifest=str(mf),completed_evidence_count=len(protected),
              latest_handoff=str(D/'handoff_update_125.txt'),local_report_original_hashes_verified=14,
              local_supplemental_hashes_verified=3,local_report_visually_inspected=True,
              latest_physical_candidate='dual_view_approach_40',latest_physical_exact_preservation_passed=False,
              latest_offline_command_arithmetic_passed=True,latest_offline_time_gate_passed=True,latest_offline_gate_scope='Only cases0/6 staged120 and command-only122; no physicalgatechange',all_required_model_geometry_paths_passed=False,
              notes='Offline112–123 complete. Two91-knot fixedfoot paths120 forcases0/6 passgeometry/staticcommands and actualfilterCPU122/123 in11.40/11.92s. No sensedload/dynamic/runtime/parentcomposition validation orphysicaladmission. Cases2/10 remainmodel-unresolved; ALL8physicalfailures unchanged. NoPPO/holdout. Preserve64,remove8 beforePPO.')
    status.write_text(json.dumps(st,indent=2))
    closure=dict(utc=now,task_complete=False,repair_complete=False,new_physical_ticks=0,new_optimizer_updates=0,
                 holdout_started=False,prior_recheck=prior,completed_evidence_manifest=str(mf),protected_completed_files=len(protected),
                 manifest_sha256=sha(mf),new_planning_sources_rechecked=True,physical_workers_running=False,
                 model_workers_running=False,audit_workers_running=False,report='progress_report_124',
                 local_review=read(review),handoff_sha256=sha(D/'handoff_update_125.txt'),
                 next='Do not rerun112–123. Prepare measuredsensor runtime and boundedvalidation for two offline paths onlyafter completecontrol/load/parentchecks; designremaining36/+20and36.5/+17.5. No automaticphysicallaunch fromauditpass. All72matchedphysicalgates beforePPO. Video59current; no newmotion.')
    (D/'closure_record_125.json').write_text(json.dumps(closure,indent=2))
    print(json.dumps(dict(prior_files=checks[0]['checked'],previous_evidence=checks[1]['checked'],completed_evidence=len(protected),workers_running=False,task_complete=False)),flush=True)
if __name__=='__main__':main()
