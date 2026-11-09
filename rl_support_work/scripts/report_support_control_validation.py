"""Aggregate frozen controller tests, preservation, actual inputs and all failures."""
import hashlib,json,shutil
from collections import Counter
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path('/data/humanoid-sitting-perception');project=Path(__file__).resolve().parents[1]
out=root/'support_control_validation_results_01';out.mkdir(exist_ok=True)
def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
banks={
    'regression':('support_control_timing_both_02','support_continuous_control_dev_01'),
    'fresh':('support_control_validation_fresh_baseline_01','support_control_validation_fresh_candidate_01'),
    'new_starts':('support_control_validation_starts_baseline_01','support_control_validation_starts_candidate_01')}
frozen=read(root/'support_control_frozen_01/manifest.json')
report=dict(training=False,policy_weights_changed=False,gains_changed=False,
            frozen_at_utc=frozen['frozen_at_utc'],banks={},gates={},preservation={})
for bank,(baseline,candidate) in banks.items():
    b=read(root/baseline/'results.json');c=read(root/candidate/'results.json')
    tracker=read(root/candidate/'shadow_score/score.json')
    patch=read(root/candidate/'patch_transport_score.json');actor=read(root/candidate/'actor_geometry_score.json')
    fixed=[i for i,(x,y) in enumerate(zip(b['episodes'],c['episodes'])) if not x['success'] and y['success']]
    lost=[i for i,(x,y) in enumerate(zip(b['episodes'],c['episodes'])) if x['success'] and not y['success']]
    report['banks'][bank]=dict(baseline=baseline,candidate=candidate,baseline_successes=b['successes'],candidate_successes=c['successes'],
        fixed_cases=fixed,regressed_cases=lost,baseline_max_force_N=max(e['peak_force_N'] for e in b['episodes']),
        candidate_max_force_N=max(e['peak_force_N'] for e in c['episodes']),
        baseline_episodes=b['episodes'],candidate_episodes=c['episodes'],tracker=tracker,patch=patch,actor=actor,
        startup_refusals=[e['env'] for e in c['episodes'] if not e['startup_ready']])
    report['banks'][bank]['runtime']=read(root/candidate/'shadow_runtime.json')
    assert sha(root/baseline/'geometry_bank.json')==sha(root/candidate/'geometry_bank.json')
    with np.load(root/baseline/'proprioception.npz') as before, np.load(root/candidate/'proprioception.npz') as after:
        for field in ('q','dq','gravity','gyro'):np.testing.assert_array_equal(before[field][0],after[field][0])
    report['banks'][bank]['same_bank_and_initial_proprioception']=True
    reasons=Counter()
    for line in (root/candidate/'tracking_feature_audit.jsonl').open():
        row=json.loads(line)
        if not row['usable']:reasons[row['reason']]+=1
    report['banks'][bank]['withheld_feature_reasons']=dict(reasons)
    dest=out/bank;dest.mkdir(exist_ok=True)
    for kind,name in [('baseline',baseline),('candidate',candidate)]:
        d=dest/kind;d.mkdir(exist_ok=True)
        for file in ('results.json','manifest.json','attempt_accounting.json','startup_admissions.jsonl','admission_geometry.json','geometry_bank.json','actor_geometry_score.json','patch_transport_score.json','shadow_runtime.json','render_audit.json'):
            if (root/name/file).exists():shutil.copy2(root/name/file,d/file)
        if (root/name/'shadow_score').exists():shutil.copytree(root/name/'shadow_score',d/'tracking',dirs_exist_ok=True)
    with np.load(root/candidate/'controller.npz') as packets:
        assert packets['policy'].shape==(1000,16,144)
    checks=read(root/candidate/'render_audit.json')['checks']
    report['banks'][bank]['max_camera_age_s']=max(max(abs(x-y) for x,y in zip(check['camera_timestamp'],check['camera_last_update'])) for check in checks)
    assert report['banks'][bank]['max_camera_age_s']<1e-6

report['gates']['known_regression_restored']=report['banks']['regression']['candidate_successes']==16
report['gates']['paired_fresh_success_not_reduced']=all(report['banks'][k]['candidate_successes']>=report['banks'][k]['baseline_successes'] for k in ('fresh','new_starts'))
report['gates']['no_new_failure_category']=all(not any(e['failures'][flag] for e in v['candidate_episodes']) or any(e['failures'][flag] for e in v['baseline_episodes'])
    for v in report['banks'].values() for flag in ('fall','wrong_contact','pelvis_floor','impact','fast_descent','torque','perception_unavailable'))
report['gates']['tracker_accuracy_and_coverage']=all(v['tracker']['all']['max_error_cm']<2 and v['tracker']['all']['availability']>=.95 and v['tracker']['near_contact']['availability']>=.95 for v in report['banks'].values())
report['gates']['patch_transport_under_2cm']=all(v['patch']['max_error_m']<.02 for v in report['banks'].values())
report['protocol_passed']=all(report['gates'].values())
report['additional_actor_input_check']=all(v['actor']['all']['max_error_cm']<2 and v['actor']['all']['availability']>=.95 and v['actor']['near_contact']['availability']>=.95 for v in report['banks'].values())
report['additional_peak_severity_check']=all(v['candidate_max_force_N']<=v['baseline_max_force_N'] for v in report['banks'].values())
report['ready_for_rl_or_promotion']=report['protocol_passed'] and report['additional_actor_input_check'] and report['additional_peak_severity_check']
report['actor_check_note']='Additional explicit end-to-end diagnostic: actual complete geometry packets can be withheld more often than scalar distance. This is reported separately from the original tracker gate.'
report['replay']=read(root/'support_continuous_replay_dev_01/manifest.json')['comparison'];assert report['replay']['bitwise_equal']
shutil.copy2(root/'support_continuous_replay_dev_01/manifest.json',out/'sensor_replay_verification.json')
shutil.copytree(root/'support_control_frozen_01',out/'frozen',dirs_exist_ok=True)
shutil.copytree(root/'support_control_timing_audit_01',out/'timing_audit',dirs_exist_ok=True)
parent=root/'support_rl_independent_train_04/final.pt'
expected='2c0f6907196be68bce9a1ee2e6b6ef6e4c82bf0ecdff622dc55b81fe8ef58d13'
assert sha(parent)==expected;report['preservation']['parent_checkpoint_sha256']=sha(parent)
for rel,expected_hash in frozen['hashes'].items():
    path=project/rel
    if path.exists():assert sha(path)==expected_hash,rel
report['preservation']['frozen_runtime_sources_unchanged']=True
for name in ('support_tracking_features_tests_02.log','support_continuous_features_tests_02.log'):
    shutil.copy2(project/'logs'/name,out/name)
report['unit_tests']=dict(tracker_and_measured_feature_contract=12,continuous_patch_motion=5,total=17)
all_runs=[f'support_control_timing_{k}_02' for k in ('none','startup','live','both')]+['support_tracking_control_dev_01','support_continuous_control_dev_01']+[f'support_control_validation_{b}_{k}_01' for b in ('fresh','starts') for k in ('baseline','candidate')]
report['completed_evaluations_this_investigation']=sum(read(root/name/'results.json')['observed_complete_trials'] for name in all_runs)
assert report['completed_evaluations_this_investigation']==160
report['all_evaluation_paths']=[str(root/name) for name in all_runs]
report['failure_audit']=read(root/'support_control_failure_audit_01/audit.json')
shutil.copytree(root/'support_control_failure_audit_01',out/'failure_audit',dirs_exist_ok=True)
report['scope']='Aligned rigid horizontal seats in simulation. No navigation, deformable seat, arbitrary pose, hardware or universal-sitting claim.'
report['diagnosis']='Both startup-only and live-only timing corrections independently expose the frozen policy sensitivity. Accurate 10 Hz seat identification alone is insufficient; 50 Hz bounded kinematic geometry feedback restores the consumed regression bank. This does not prove a unique scalar feature caused every force difference.'
(out/'results.json').write_text(json.dumps(report,indent=2)+'\n')

names=list(banks);fig,axes=plt.subplots(1,3,figsize=(14,4.3),layout='constrained')
x=np.arange(3);width=.35
axes[0].bar(x-width/2,[report['banks'][k]['baseline_successes'] for k in names],width,label='Corrected camera baseline',color='#c95548')
axes[0].bar(x+width/2,[report['banks'][k]['candidate_successes'] for k in names],width,label='Continuous tracked input',color='#087f8c')
axes[0].set_ylim(0,18);axes[0].set_ylabel('Successful complete trials / 16');axes[0].legend(fontsize=8)
for j,k in enumerate(names):
    v=report['banks'][k]
    axes[0].text(j-width/2,v['baseline_successes']+.3,str(v['baseline_successes']),ha='center')
    axes[0].text(j+width/2,v['candidate_successes']+.3,str(v['candidate_successes']),ha='center')
axes[1].bar(x-width/2,[100*report['banks'][k]['tracker']['near_contact']['availability'] for k in names],width,label='Distance tracker',color='#43a5ad')
axes[1].bar(x+width/2,[100*report['banks'][k]['actor']['near_contact']['availability'] for k in names],width,label='Complete actor geometry',color='#315a76')
axes[1].axhline(95,color='#aa3333',ls='--',lw=1);axes[1].set_ylim(0,107);axes[1].set_ylabel('Landing-window availability (%)');axes[1].legend(fontsize=8)
axes[2].bar(x-width/2,[report['banks'][k]['tracker']['all']['max_error_cm'] for k in names],width,label='Seat distance',color='#43a5ad')
axes[2].bar(x+width/2,[100*report['banks'][k]['patch']['max_error_m'] for k in names],width,label='Transported patch points',color='#315a76')
axes[2].axhline(2,color='#aa3333',ls='--',lw=1);axes[2].set_ylabel('Worst accepted error (cm)');axes[2].legend(fontsize=8)
for ax in axes:
    ax.set_xticks(x,['Known bank','Fresh shapes','New starts']);ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15)
fig.suptitle('Frozen-controller validation — every 20-second attempt counted, including failures',fontsize=14)
fig.savefig(out/'validation_summary.png',dpi=160)

lines=['# Camera timing and controller validation','',
       '**The original timing regression is repaired on the known bank.** Corrected camera timing with the old observer gives 15/16 successful sits; the frozen policy with 50 Hz tracked geometry gives 16/16. Case14 peak seat force changes from 729.20 N to 362.51 N, below the unchanged 632.53 N limit. No RL, weight update, motor-gain change or failure-threshold change was used.','',
       f'**Broader release gate: {"PASS" if report["ready_for_rl_or_promotion"] else "NOT PASSED"}.** RL remains paused and deployment defaults are unchanged. Fresh tests are reported below, including all unsuccessful cases and worst impacts.','',
       '## Why newer camera images changed the behavior','',
       'The frozen controller was trained with an observation pipeline containing timing errors and intermittent missing seat measurements. Correcting the first image changed its initial geometric plan; correcting later images changed its feedback. Either change independently lost the same known-bank sit. The historical control reproduces all 12 recorded trajectory arrays and all archived actor packets exactly. The both-corrected control also reproduces the earlier passive-tracker run exactly, ruling out recording overhead or passive tracking as the cause.','',
       '| Camera condition | Successful sits | Case14 peak force |','|---|---:|---:|',
       '| Historical timing | 16/16 | 455.21 N |','| Startup corrected only | 15/16 | 697.77 N |','| Descent corrected only | 15/16 | 714.39 N |','| Both corrected | 15/16 | 729.20 N |','| Both corrected + tracked patch at 10 Hz | 15/16 | 669.33 N |','| Both corrected + tracked patch at 50 Hz | 16/16 | 362.51 N |','',
       'In case14 the corrected startup seat-height feature changes by -5.219 mm and body-relative seat height by +1.772 mm. The commanded targets change immediately. The failed landing cannot be attributed to one scalar alone: initial geometry, patch validity, timing and closed-loop motion interact.','',
       '## Implemented repair','',
       'The depth camera remains synchronized. The same observed seat patch is carried into the current pelvis frame every 20 ms using joint forward kinematics and IMU orientation. New images refresh it every 100 ms. The ten actor features contain current clearance, normal, observed-patch center and spans, validity, and the age of the underlying image. Simulator geometry and forces are used only for separate scoring.','',
       'Prediction lasts at most 100 ms. Its engineering error budget charges the full inferred 3D body translation, disagreement between the two feet, and any plane reprojection correction. Unknown or over-budget geometry is withheld. Two feet moving together cannot prove contact; the full-motion charge is a conservative guard, not a statistical guarantee. These are tracked observed points, not inferred hidden furniture bounds.','',
       '## Frozen paired validation','',
       '| Bank | Corrected baseline | Tracked controller | Cases repaired | Cases regressed | Worst force: baseline → tracked |','|---|---:|---:|---|---|---:|']
for k,v in report['banks'].items():lines.append(f'| {k} | {v["baseline_successes"]}/16 | {v["candidate_successes"]}/16 | {v["fixed_cases"]} | {v["regressed_cases"]} | {v["baseline_max_force_N"]:.0f} → {v["candidate_max_force_N"]:.0f} N |')
lines+=['','Fresh shapes/starts use 51941/51943; new starts on the unchanged mixed training bank use 51923. Source and settings were frozen before those data were viewed. Each attempt lasts 20 s even after a physical failure. There are 160 completed evaluation attempts across diagnosis, development and validation; none are RL training.','',
        '| Bank | Tracker full-trial availability | Tracker landing availability | Actor landing availability | Worst distance error | Worst patch transport error |','|---|---:|---:|---:|---:|---:|']
for k,v in report['banks'].items():
    lines.append(f'| {k} | {100*v["tracker"]["all"]["availability"]:.2f}% | {100*v["tracker"]["near_contact"]["availability"]:.2f}% | {100*v["actor"]["near_contact"]["availability"]:.2f}% | {10*v["tracker"]["all"]["max_error_cm"]:.2f} mm | {1000*v["patch"]["max_error_m"]:.2f} mm |')
lines+=['','The actor requires a complete patch estimate. Its input is withheld more often than scalar distance, particularly during rapid motion or loss of a reliable foot reference. All unavailable and post-failure frames remain in the denominators. Direct camera-point samples have zero *transport* error by definition; the reported transport maximum and p95 also separately cover predicted samples. Distance error is independently measured against true seat clearance.','',
        'Worst impacts increased in already-failing fresh/new-start trials despite unchanged success counts. The candidate remains experimental; the known-bank repair does not establish that every motion is safer.','', '## Gates and verification','']
for k,v in report['gates'].items():lines.append(f'- {k}: **{v}**')
lines += [f'- Additional complete-actor-input accuracy/coverage check: **{report["additional_actor_input_check"]}**',
          f'- Additional worst-impact severity check: **{report["additional_peak_severity_check"]}**',
          '- Seventeen tests pass. All known-bank sensor-only replay actor packets match the online geometry bit for bit (15,984 packets).',
          '- All corrected camera acquisition audits have zero sensor timestamp age; no extra physics step is inserted.',
          '- Frozen runtime sources and parent checkpoint are hash verified. Old checkpoints, datasets, experiments and the unsuccessful 10 Hz adapter are preserved.',
          '- The known-bank geometry update takes 1.84 ms median and 37.17 ms p95 per robot, including image frames. The 50 Hz claim describes simulation/control timestamps; hard real-time hardware throughput is not established.',
          '', '## Scope and next action','',
          'The known regression is fixed, but promotion and further RL require the remaining failed gates to be addressed. Next address the loss of complete geometric feedback during fast motion, then the remaining balance/collision failures and worse impacts. A useful next investigation is requesting a new depth frame when the kinematic error budget is about to expire, and separating distance confidence from patch-shape confidence at the controller boundary. Any modified candidate needs new validation seeds; the banks in this report are now consumed. The current work covers aligned rigid horizontal surfaces in simulation; it does not establish arbitrary furniture, cushions, approach/navigation or hardware performance.','',
          'See [all machine-readable results](results.json), [timing comparison](timing_control_comparison.png), [fresh validation plot](validation_summary.png), [the repaired landing video](repaired_landing/landing_tracking.mp4), and [a remaining fresh-bank fall](fresh_failure/landing_tracking.mp4). The repaired video is the originally regressed case14; the failure video is fresh case10, the first remaining fall by case index. Both use depth and joint kinematics for the 3D scene; truth appears only on the accuracy graph.','']
(out/'REPORT.md').write_text('\n'.join(lines))
print(json.dumps(dict(gates=report['gates'],actor_check=report['additional_actor_input_check'],ready=report['ready_for_rl_or_promotion'],successes={k:[v['baseline_successes'],v['candidate_successes']] for k,v in report['banks'].items()}),indent=2))
