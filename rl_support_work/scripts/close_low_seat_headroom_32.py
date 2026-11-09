"""Preserve completed evidence and update mutable handoffs; task stays unfinished."""
from pathlib import Path
import json,hashlib,shutil,datetime,subprocess
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');R=Path('/home/kadir/humanoid-sitting');read=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 out=D/'closure_record_32.json';assert not out.exists();a=read(D/'tracking_headroom_31/independent_tracking_audit_v2_31.json');assert a['software_audit_passed'] and not a['exact_preservation_gate_passed']
 assert not (D/'ppo_01').exists()
 reservation=read(D/'reservation.json');assert all(sha(p)==h for p,h in reservation['hashes'].items())
 checks=read(D/'prior_recheck_during_headroom_31.json');assert checks['passed']
 revision=read(D/'tracking_headroom_31/revision.json');assert all(sha(p)==h for p,h in revision['sources'].items())
 previous=read(D/'completed_probes_preservation_27.json');assert all(sha(p)==h for p,h in previous.items())
 archive=D/'completed_source_archive_32';archive.mkdir(exist_ok=False)
 names=['screen_low_seat_staged_approach_28.py','screen_low_seat_early_anchor_clearance_29.py','diagnose_low_seat_command_tracking_30.py','low_seat_tracking_headroom_31.py','run_low_seat_tracking_headroom_31.py','launch_low_seat_tracking_headroom_31.py','audit_low_seat_tracking_headroom_31.py','audit_low_seat_tracking_headroom_v2_31.py','recover_low_seat_headroom_audit_31.py','report_low_seat_headroom_32.py','report_low_seat_headroom_v2_32.py','format_low_seat_review_32.py',Path(__file__).name]
 for n in names:shutil.copy2(R/'scripts'/n,archive/n)
 added={}
 for name in ['staged_approach_screen_28','early_anchor_clearance_screen_29','command_tracking_diagnosis_30','tracking_headroom_31','tracking_headroom_audit_recovery_31','report_recovery_32','progress_report_32','progress_report_v2_32','completed_source_archive_32']:
  for p in (D/name).rglob('*'):
   if p.is_file() and not p.is_symlink():added[str(p)]=sha(p)
 for pattern in ['*headroom*31*.log','*screen*28.log','*screen*29.log','report_headroom*32.log','prior_recheck_during_headroom_31.json']:
  for p in D.glob(pattern):
   if p.is_file():added[str(p)]=sha(p)
 evidence=dict(previous,**added);(D/'completed_probes_preservation_32.json').write_text(json.dumps(evidence,indent=2))
 utc=datetime.datetime.now(datetime.timezone.utc).isoformat()
 notes=f'''Latest completed physical probe — {utc}

The sixteenth candidate tracking_headroom_31 is COMPLETE and REJECTED:16/24 stable,14clean,5slipping,8forbidden housing contacts/physical failures,17conservative warnings,zero illegal-patch cases. No clean case gained or lost versus bestnormal3s on this24-case isolating bank. Six housing failures remain added relative to bestnormal3s; all eight targeted failure rows remain unresolved. No full72case retention, PPO, fresh holdout or promotion followed. No workers remain.

Revision31 explicitly changed only applied command-rate cap on six measured-selected joint17 approach rows during8<=t<15:0.20rad/s instead of0.15, then original0.15 from15s. Nominal path,filter0.1,feedback,parent,materials and physical criteria unchanged. It was a PRE-TRAINING control revision, not an execution fix. Runtime low_seat_tracking_headroom_31.py imports the unchanged17 planner. Every20ms command recorded:1000steps,2100selected env-ticks,maxstep0.004000008rad,float32 bound passed. Frozen417input/action/reward/parent/planner/startfirst8/fingers/zerooptimizer audits passed. Full20s unselected exactness remainsFALSE; maxbodycomponentdifference0.620mm,q0.000962rad. Old17 failed exact/physical gates stayfalse. Do not call the new overall gate passed.

Tracking improved but did not repair clearance: sampled hip reference-command gaps near targeted contacts fell from0.052–0.063 toabout0.017–0.020rad,estimated inward error to2–4mm. Contacts movedlater to14.92–15.04s but allsix stillfailed. Sensor-based unloading waszero beforeoldjoint17contacts,contradicting prematureunloading as their explanation. Two new36/+15episodes hadvalid=false,geometryage0.8s beforecontact; otherfailedrowsfresh. Do not claimallfailureshadcurrentgeometry. No solecauseproof; more rate alone is not supported as a repair. Next planning must address model clearance, dynamic support/tracking and loss of measured geometry, not repeat blind gain/lean/arm sweeps.

Auditorv1 incorrectly searched thumb/index/middle names rather than the model's numbered14finger names. Original source/log/status preserved in tracking_headroom_audit_recovery_31. v2 retains the same bit-exact14finger and command/input limits and passes software checks; tracking_headroom_audit_recovery_status_31.json supersedes original historical status failure. No physics rerun. Reportv1 incorrectly assumedfreshgeometrybeforeeveryfailure; report_recovery_32 preserves source/log/partialreport. v2 records missing values/lastfresh time withouttruth/replayed-sensor substitution. No criteria changed.

Offline28: two earlierstarts1/4s,20mmhover,0.5shold,normaldescent,0.5ssettle:all8testedpaths metgeometryscreens butfailedfinishbefore15s under0.15rad/s. Offline29early1sanchorendpointsearch bestclearances7.08/3.70/6.95/5.66mm;onlyfirstreached7mmresearchreserve. Neitherboundedmodelsearchprovesimpossibility or physicalsafety. command_tracking_diagnosis_30 coversall48priorrecordedepisodes. No simulation or PPO inthese diagnostics.

Best verified full72 controller remains current_retention_09:64clean,70stable,8slip,2housing;all60originalsuccesses+4gainsretained. Preserveall64.50margin-warningcases,22strictclean. Remaining36/+15,+20;37/+16.25;36.5/+17.5,twiceeach. Thedeepest36/+20remainsunrepaired. New31common14cleanpairs vsnormal3s whole20torque²+0.00194%,final5+0.01018%;tinyincreases,noeffortsaving. Earlier09vsselected62111@8 common20olddev:whole20+4.38%,final5+25.45%. No failed-fragment savings or newPPOgains.

progress_report_v2_32 copied locally;16 original SHA entries and supplemental original/review hashes verified;comparison.png,tracking.png,findings.html/review.html visually inspected includingefforttable. URLhttp://127.0.0.1:8807/progress_report_v2_32/review.html. All24,242priorstudyfiles and2048prior-evidencefiles unchanged; completed_probes_preservation_32 now{len(evidence)}files. Newprobe24000active+600initialsettling,zerooptimizer. Sources/reservedbanksverified. Mutablehandoffsupdated;automationACTIVE,taskunfinished.

PPO63111/63117 must wait untiltargetedfailuresremovedandall64successfulconditionsretainedinmatchedfull20s72developmenttests. Fresh24reservedholdoutremainsCLOSED throughrepair,normalization,trainingandselection. No promotion,thresholdrelaxation or mainbudgetextension. Reference-assistedsimulation/rigidslabsandtwoseedsdonotestablishsittingorleaninganywhere;leaninghasnotbroadened.
'''
 history=D/'documentation_history_01';history.mkdir(exist_ok=True)
 for p in [R/'docs/LOW_SEAT_CONTACT_STATUS.md',R/'docs/UBUNTU_CODEX_HANDOFF.md']:
  shutil.copy2(p,history/(p.stem+'_before32.md'));p.write_text(notes+'\nEarlier entries follow.\n\n'+p.read_text())
 old=read(D/'status.json');shutil.copy2(D/'status.json',history/'status_before32.json')
 old.update(utc=utc,phase='physical_repair_unresolved_headroom_probe_rejected',latest_candidate='tracking_headroom_31',latest_candidate_outcome=a['candidate'],latest_independent_audit_passed=False,latest_software_audit_passed=True,latest_exact_preservation_gate_passed=False,physical_jobs_running=False,model_jobs_running=False,latest_report='progress_report_v2_32/review.html',latest_audit_status='tracking_headroom_audit_recovery_status_31.json',completed_evidence_manifest='completed_probes_preservation_32.json',completed_evidence_count=len(evidence),initial_prior_hash_check='prior_recheck_during_headroom_31.json',local_report_original_hashes_verified=16,local_supplemental_hashes_verified=2,local_report_visually_inspected=True,scope='One24case bounded rate-headroom probe rejected; unchanged64/72best. Physical prerequisite unmet; no PPO/holdout.')
 (D/'status.json').write_text(json.dumps(old,indent=2));closure=dict(utc=utc,task_complete=False,notes=notes,status=old,evidence_files=len(evidence),prior_verified=24242,prior_evidence_verified=2048,report_original_hashes_verified=16,review_visually_inspected=True,all_new_sources_verified=True,reserved_banks_verified=True)
 out.write_text(json.dumps(closure,indent=2));print(json.dumps({k:v for k,v in closure.items() if k not in ['notes','status']},indent=2))
if __name__=='__main__':main()
