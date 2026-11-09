from pathlib import Path
import json,hashlib,datetime,shutil,subprocess
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');R=Path('/home/kadir/humanoid-sitting')
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(p,v):
 with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n')
now=datetime.datetime.now(datetime.timezone.utc).isoformat();stamp=now.replace(':','').replace('+','_')
assert read(D/'prior_recheck_after_diagnostics_27.json')['passed']
a=read(D/'sensor_prefix_v4_24/independent_audit.json');s=read(D/'progress_report_27/summary.json');assert not s['task_complete'] and a['source_hashes_unchanged']
assert all(v['exact'] for group in a['replay_to_original'].values() for v in group.values())
assert not subprocess.run(['tmux','list-sessions','-F','#{session_name}'],capture_output=True,text=True).stdout.strip()
recovery=dict(utc=now,static_render_attempts_22='Inconclusive; static pose writes did not yield valid matching attached-camera transforms/support geometry.0 physics integration. Original v1 unavailable fits, v2 read-only NumPy error, v3/v4 pose mismatch preserved; no runtime diagnosis inferred from them.',sensor_prefix_attempts_24='Original baseline ended without persisted exception; instrumented v2 reproduced the accounting assertion:1680 actual5ms simulator substeps, not420, for8.4s. v3 indentation error ended after120 active ticks. v4 fixed persistence scope and elapsed-time accounting only. Both v4 prefixes exactly reproduce the corresponding original saved traces. Earlier premature audit log retained; final audit waited for completed inputs. No physics/controller/tolerance or physical-criterion change.',accounting=s['diagnostic_accounting'],no_new_full20s_repair_trial=True,main_training=False,fresh_holdout=False)
save(D/'diagnostic_recovery_27.json',recovery)
archive=D/'completed_source_archive_27';archive.mkdir()
for p in sorted((R/'scripts').glob('*low_seat*.py')):
 if p.stem.endswith(tuple('_'+str(i) for i in range(22,28))):shutil.copy2(p,archive/p.name)
old=read(D/'completed_probes_preservation_21.json');m=dict(old)
for p in D.iterdir():
 if any('_'+str(i) in p.name for i in range(22,28)) and p.name!='completed_probes_preservation_27.json':
  for f in ([p] if p.is_file() else p.rglob('*')):
   if f.is_file() and '__pycache__' not in str(f):m[str(f)]=sha(f)
save(D/'completed_probes_preservation_27.json',m)
notes=f'''Latest diagnostic update — {now}

Physical repair remains unresolved. Best verified current_retention_09 still retains all60 original successes plus4 added successes:64/72 clean,70stable,8slipping,2housing-contact cases. All64 remain the prerequisite retention set. The fifteenth candidate joint_path_17 remains REJECTED,16/24stable/14clean/6slip/8housing; its exact-preservation and physical gates remain false. No new full20s repair trial, PPO, fresh holdout or policy promotion occurred in this diagnostic round. No physical/model worker remains.

A meaningful measurement diagnosis is now complete: sensor_prefix_v4_24 captured two same-GPU8.4s prefixes. BOTH exactly replayed their original full trace prefix arrays (time,body,q,command,geometry,reference,torque). Until8.0s unselected depths were identical. At8.2s unselected q/gravity/body/camera pose/intrinsics stayed EXACT, but rendered depth differed by up to3.814697µm and plane features by1.490116e-6. At8.4s depth validity edges also differed; maximum finite depth difference remained3.814697µm. Independent18-case plane refits EXACTLY reproduced both recorded geometries. This locates the first divergence in rendered measurements, before unselected control or motion changes; the internal renderer mechanism is not established. Do not relabel the original failed full20s preservation gate as passed or substitute truth/replayed sensors. Audit: sensor_prefix_v4_24/independent_audit.json. No tolerances relaxed.

Static pose-render attempts22 are INCONCLUSIVE because attached-camera transforms/support fits did not match written poses. Preserve all four sources/logs/results. Short physical prefix recovery24 preserved: wrong assertion expecting420 simulator steps rather than1680×5ms; v3 capture-write indentation failed atstep1; an audit was attempted prematurely before changed inputs finished. Completedv4 pair20160active+1200settling, zero complete20s episodes/optimizer/holdout. Failed diagnostic attempts add at most20280active+1800settling; overall upper bound40440active+3000settling. These are separate diagnostic budgets, not main training or repaired trials. diagnostic_recovery_27.json records everything.

Offline sensed_clearance_23 shows measured plane+encoders warn below5mm before all2 best-normal and8 rejected-path housing contacts, but only0.08–0.20s before, with6/14 clean cases also warning. Near-event model-to-label errors roughly2mm.5mm is a diagnostic line, not a physical threshold or certified safety filter. All48 recorded episodes are included;0.1s poses only bracket0.02s events.

Offline bounded screen25 uses ACTUAL saved7s IMU/encoder/plane inputs and exact frozen bridge inertial model. With fixed feet, previous fixed−0.35rad arm offset, pitch<=0.9rad,5mm model CoP/15mm patch limits, three local searches pergeometry found best tested housing clearances6.85/3.46/6.70/5.41mm for36/+15,36/+20,37/+16.25,36.5/+17.5. None reached the7mm research reserve motivated by recorded tracking error. This is not a proof of impossibility or a changed physical pass criterion.

Offline screen26 minimizes proposed symmetric sagittal foot displacement within±60mm under the same model support constraints and7mm reserve. It found hypothetical rearward18.0/22.5mm endpoints for36/+15 and37/+16.25 (7.24mm clearance), but none in the bounded searches for36/+20 or36.5/+17.5. These endpoints have NO stepping/whole-path/rate/physical certificate and MUST NOT be executed as dragged feet or teleported endpoints. No new physical candidate was launched. Next work should address the limited support/clearance envelope and dynamically feasible approach/contact planning, with explicit versioned scope and measured tracking/clearance feedback. Do not repeat blind lean/gain/arm sweeps, or presume the largest nominal endpoint clearance removes slipping.

progress_report_27 is copied locally to validated_low_seat_contact_01/progress_report_27;12 original SHA entries verified. clearance_diagnosis.png,depth_comparison.png and findings.html visually inspected including model table. URL http://127.0.0.1:8807/progress_report_27/findings.html. It is interim DEVELOPMENT diagnosis, not PPO/unseen/final results. Previous effort remains mixed; against selected62111@8 on20 common clean old-development trials, verified09 whole20 torque²+4.38%,last5+25.45%; no new complete-episode effort claim.

All24,242 initial prior-study hashes and1,810 completed prior-evidence hashes rechecked intact in prior_recheck_after_diagnostics_27.json. completed_probes_preservation_27.json protects{len(m)} files. Fresh24 holdout remains CLOSED. Preserve all64 successes and remove all8 target slip/two housing cases in matched full20s72development tests BEFORE small PPO63111/63117. Keep the documented8update/seed uninterrupted budget and full independent audits only after physical admission. Automation stays ACTIVE; do not pause or claim completion. Sitting/leaning anywhere and real-robot capability remain unestablished.

Earlier entries follow.

'''
hist=D/'documentation_history_01';hist.mkdir(exist_ok=True)
for p in [D/'status.json',R/'docs/LOW_SEAT_CONTACT_STATUS.md',R/'docs/UBUNTU_CODEX_HANDOFF.md']:shutil.copy2(p,hist/(p.name+'.before_diagnostics_27.'+stamp))
st=read(D/'status.json');st.update(utc=now,phase='physical_repair_unresolved_sensor_diagnosis_completed',physical_jobs_running=False,model_jobs_running=False,latest_report='progress_report_27/findings.html',latest_diagnostic='sensor_prefix_v4_24/independent_audit.json',completed_evidence_manifest='completed_probes_preservation_27.json',completed_evidence_count=len(m),initial_prior_hash_check='prior_recheck_after_diagnostics_27.json',local_report_original_hashes_verified=12,local_supplemental_hashes_verified=0,local_report_visually_inspected=True,scope='New sensor and endpoint diagnostics only. Best verified64/72clean unchanged; no new full20s repair/PPO/freshholdout. Physical prerequisite unmet.')
(D/'status.json').write_text(json.dumps(st,indent=2)+'\n')
for p in [R/'docs/LOW_SEAT_CONTACT_STATUS.md',R/'docs/UBUNTU_CODEX_HANDOFF.md']:p.write_text(notes+p.read_text())
save(D/'closure_record_27.json',dict(utc=now,task_complete=False,notes=notes,new_evidence_count=len(m),prior_files_verified=24242,prior_evidence_verified=1810,report_files_verified=12,report_visually_inspected=True,status=st))
print(json.dumps(dict(evidence=len(m),phase=st['phase'],utc=now)))
