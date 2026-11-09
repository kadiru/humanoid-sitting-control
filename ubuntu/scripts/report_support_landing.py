"""Assemble honest validation gates, camera diagnosis and review artifacts."""
import hashlib,json,shutil,subprocess
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];STORE=Path('/data/humanoid-sitting-perception');OUT=STORE/'support_landing_validation_results_02'
OUT.mkdir(exist_ok=False)
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
banks={'new_starts':'support_landing_validation_capture_01','regression':'support_landing_regression_capture_01','fresh':'support_landing_fresh_capture_01'}
reports={};checks={};evidence={};startup_refusals={}
frozen=read(STORE/'support_landing_frozen_01/manifest.json')
for path,digest in frozen['hashes'].items():assert sha(ROOT/path)==digest,('frozen source changed',path)
shutil.copytree(STORE/'support_landing_frozen_01',OUT/'frozen')
for key,name in banks.items():
 cap=STORE/name;report=read(cap/'shadow_score/score.json');manifest=read(cap/'manifest.json');audit=read(cap/'render_audit.json')
 assert audit['refresh']=='due'
 max_age=max(abs(a-b) for r in audit['checks'] for a,b in zip(r['camera_timestamp'],r['camera_last_update']))
 assert max_age<1e-6
 assert len(audit['checks'])==201
 for path,digest in manifest['hashes'].items():assert sha(ROOT/path)==digest,(name,path)
 checks[key]=dict(accuracy=report['all']['errors_above_2cm']==0 and report['all']['p95_error_cm']<1.,
     coverage=report['all']['availability']>=.95 and report['near_contact']['availability']>=.95,
     capture_age_s=max_age,online_samples=read(cap/'shadow_runtime.json')['samples'])
 reports[key]=report
 startup_refusals[key]=[dict(case=r['env'],reason=r['startup_refusal_reason']) for r in read(cap/'results.json')['episodes'] if not r['startup_ready']]
 target=OUT/key;target.mkdir()
 for filename in ['manifest.json','render_audit.json','shadow_runtime.json','results.json','calibration.json','geometry_bank.json']:
  shutil.copy2(cap/filename,target/filename)
 shutil.copytree(cap/'shadow_score',target/'tracking')
 evidence[key]={x.name:sha(x) for x in cap.iterdir() if x.is_file()}
# Exact online/offline replay is checked across every saved output array.
online=STORE/banks['fresh']/'shadow_score/tracking.npz';offline=STORE/'support_landing_fresh_replay_none_01/tracking.npz'
with np.load(online) as a,np.load(offline) as b:
 assert set(a.files)==set(b.files)
 for key in a.files:np.testing.assert_array_equal(a[key],b[key],err_msg=key)
checks['online_offline_bitwise']=True
shutil.copy2(STORE/'support_landing_fresh_replay_none_01/manifest.json',OUT/'offline_replay_manifest.json')
failure_audit=read(STORE/'support_landing_failure_audit_01/audit.json')
shutil.copytree(STORE/'support_landing_failure_audit_01',OUT/'failure_audit')
shutil.copytree(STORE/'support_landing_fresh_visual_02',OUT/'visualization')
shutil.copytree(STORE/'support_landing_development_visual_02',OUT/'development_visual')
historical=read(STORE/'support_rl_retention_easy_parent_01/results.json')
current=read(STORE/banks['regression']/'results.json')
control_regression=dict(historical_successes=historical['successes'],synchronized_successes=current['successes'],
 lost_cases=[dict(case=a['env'],historical_peak_force_N=b['peak_force_N'],synchronized_peak_force_N=a['peak_force_N'],failures=a['failures'])
 for a,b in zip(current['episodes'],historical['episodes']) if b['success'] and not a['success']],
 note='Frozen policy, same known geometry/start seeds; synchronized startup and live camera samples differ. The new tracker is passive; this is not a tracker-driven controller test.')
stress={}
for fault in ['noise','blackout']:
 path=STORE/f'support_landing_fresh_replay_{fault}_01';stress[fault]=read(path/'score.json');shutil.copytree(path,OUT/fault,ignore=shutil.ignore_patterns('records.jsonl'))
with np.load(STORE/'support_landing_fresh_replay_blackout_01/tracking.npz') as z:
 middle=(z['t']>11.80001)&(z['t']<=12.40001)
 assert not z['valid'][middle].any(),'Stale distance during blackout'
 checks['blackout_withholds_stale']=True
 recovery=[]
 for case in range(z['valid'].shape[1]):
  ids=np.flatnonzero((z['t']>12.40001)&z['valid'][:,case]);recovery.append(float(z['t'][ids[0]]-12.4) if len(ids) else None)
 checks['blackout_recovery_after_end_s']=recovery
checks['stress_no_errors_above_2cm']=all(s['all']['errors_above_2cm']==0 for s in stress.values())
checks['parent_unchanged']=sha(STORE/'support_rl_independent_train_04/final.pt')=='2c0f6907196be68bce9a1ee2e6b6ef6e4c82bf0ecdff622dc55b81fe8ef58d13'
assert checks['parent_unchanged']
assert subprocess.check_output(['git','-C',str(ROOT/'IsaacLab'),'status','--porcelain'],text=True)==''
checks['isaaclab_clean']=True
checks['data_free_gib']=shutil.disk_usage(STORE).free/2**30
assert checks['data_free_gib']>30
for name in ['support_landing_tests_07.log','support_camera_tests_01.log','support_landing_preservation_01.log']:
 shutil.copy2(ROOT/'logs'/name,OUT/name)
shutil.copytree(ROOT/'docs',OUT/'protocol',ignore=lambda root,names:[x for x in names if x!='SUPPORT_LANDING_VALIDATION_PROTOCOL.md'])
source=OUT/'source';source.mkdir()
for p in [ROOT/'sitting_task/src/humanoid_sitting/support_landing_geometry.py',ROOT/'sitting_task/src/humanoid_sitting/support_camera.py',*list((ROOT/'scripts').glob('*support_landing*.py')),ROOT/'scripts/score_landing_shadow.py',ROOT/'scripts/test_support_camera.py',ROOT/'scripts/audit_landing_losses.py',ROOT/'scripts/run_support_landing_validation_01.sh']:
 shutil.copy2(p,source/p.name)
for name in ['support_landing_diagnosis_01','support_landing_timing_old_01','support_landing_timing_sync_03','support_landing_development_track_02','support_landing_sync_track_05']:
 path=STORE/name;target=OUT/name;target.mkdir()
 for f in ['score.json','summary.json','manifest.json']:
  if (path/f).exists():shutil.copy2(path/f,target/f)
passed=all(checks[b]['accuracy'] and checks[b]['coverage'] for b in banks) and checks['stress_no_errors_above_2cm']
result=dict(passed=passed,banks=banks,checks=checks,reports=reports,stress=stress,evidence=evidence,startup_refusals=startup_refusals,failure_audit=failure_audit,control_regression=control_regression,rl_training=False,
 capture_metadata_note='The base capture calibration flag original_controller_inputs_unchanged refers to its observer wrapper, not historical image parity. The synchronized subclass changes camera samples; actor architecture and weights are unchanged.',
 scope='Initially aligned rigid horizontal seats; new tracker passive; every failed motion included. No hardware or moving/deforming seat validation.')
(OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
fig,axs=plt.subplots(1,3,figsize=(14,4.2));labels=['New starts','Regression','Fresh shapes']
x=np.arange(3);old=[reports[k]['at_depth_frames']['legacy_near_contact_accepted']/reports[k]['at_depth_frames']['near_contact_frames']*100 for k in banks];new=[reports[k]['at_depth_frames']['new_near_contact_accepted']/reports[k]['at_depth_frames']['near_contact_frames']*100 for k in banks]
axs[0].bar(x-.18,old,width=.36,color='#c2410c',label='Original on fresh images');axs[0].bar(x+.18,new,width=.36,color='#047857',label='New tracker');axs[0].set(xticks=x,xticklabels=labels,ylim=(0,105),ylabel='Available camera frames (%)',title='Landing camera updates (10 Hz)');axs[0].legend(fontsize=8)
axs[1].bar(x,[reports[k]['all']['max_error_cm'] for k in banks],color='#047857');axs[1].axhline(2,color='#dc2626',ls='--',label='2 cm gate');axs[1].set(xticks=x,xticklabels=labels,ylabel='Worst accepted error (cm)',title='All 48 full attempts included');axs[1].legend(fontsize=8)
oldtime=read(STORE/'support_landing_timing_old_01/summary.json');newtime=read(STORE/'support_landing_timing_sync_03/summary.json')
for d,c,label in [(oldtime,'#c2410c','Old acquisition'),(newtime,'#047857','Fixed acquisition')]:axs[2].plot(np.arange(9)*20,d['median_angle_by_lag_deg'],color=c,label=label)
axs[2].set(xlabel='IMU sample age (ms)',ylabel='Median surface / IMU angle (degrees)',title='Camera timing checked from sensors');axs[2].legend(fontsize=8)
for ax in axs:ax.grid(axis='y',alpha=.15)
fig.tight_layout();fig.savefig(OUT/'validation_summary.png',dpi=160);plt.close(fig)
lines=['# Landing distance tracking validation','',f"Validation gate: {'PASS' if passed else 'NOT PASSED'}; no RL training was run.",'','| Bank | All-tick availability | Landing availability at 50 Hz | Worst accepted error | At first supported contact |','|---|---:|---:|---:|---:|']
for key,r in reports.items():lines.append(f"| {key} | {r['all']['availability']*100:.2f}% | {r['near_contact']['availability']*100:.2f}% | {r['all']['max_error_cm']:.3f} cm | {sum(v['valid'] for v in r['event_cases'].values())}/{len(r['event_cases'])} |")
lines+=['', 'Fresh-bank loss diagnosis: the four long gaps start after physical failures, with torso tilt about 64–106 degrees. Before the first physical failure, fresh-bank availability is 99.74%; on its eight successful full trials it is 99.95%. These are post-hoc conditional diagnostics, not substitutes for the failed all-frame gate.', '',f'Startup refusals (included in the all-tick denominator): {json.dumps(startup_refusals)}', '', 'Overall release requires at least 95% availability both overall and around landing on every bank, and no accepted error above 2 cm. A failed gate blocks promotion and more RL with this new observation pipeline.', '', 'Camera reads previously received images about 95 ms old. Explicit acquisition now captures at the controller timestamp. Sensor-only floor-normal/IMU agreement improved from a 100-ms best lag to zero lag; development maximum current angle is 0.063 degrees.','',
'The tracker follows fresh seat depth, uses joints and IMU to carry a measured floor reference, and bridges camera intervals with a short kinematic prediction. If the seat is briefly hidden, a currently observed floor can carry its previously measured rigid height for at most 0.6 seconds. Predictions expire and uncertainty can make the output unavailable.','',
'Online and sensor-only offline replay match bit for bit. Simulator labels are separate and used only for scoring. The tracker remains passive: it does not yet replace the frozen actor observation contract. Synchronized camera samples do change that original actor’s inputs, so historical motion parity is not claimed.','',
f'Control limitation: the unchanged policy completed {current["successes"]}/16 known trials with synchronized camera samples, versus {historical["successes"]}/16 historically. Lost cases: {json.dumps(control_regression["lost_cases"])}. Improved estimation does not establish improved control, and no new tracker output was sent to the actor.', '',
'All code, datasets and checkpoints were retained. Installed IsaacLab remains clean. The old preservation checker verifies preservation but still fails its pre-existing 30-GiB root-disk reserve rule; the /data output volume is checked separately and has more than 30 GiB available.','',
'These results cover aligned rigid horizontal seats and the recorded motion distribution. Simultaneous body translation and foot motion cannot be inferred from joint encoders alone; moving or deforming hidden seats are not validated.']
(OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(dict(passed=passed,checks=checks),indent=2))
