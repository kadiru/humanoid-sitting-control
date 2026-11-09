"""Write reproducible measured progress, source snapshots and integrity manifest."""
from pathlib import Path
import json,hashlib,shutil,re
root=Path(__file__).resolve().parents[1];out=root/'runs/soft_landing_01';rows=json.loads((out/'final_metrics/metrics.json').read_text());by={Path(r['run']).name:r for r in rows}
def table(names):
 text='| Run | Success | Falls / unintended | Force failures | Peak N: median / P95 / max | Max descent m/s | Hold min / median s |\n|---|---:|---:|---:|---:|---:|---:|\n'
 for n in names:
  r=by[n];f=r['force_peak_quantiles_N'];h=r['hold_quantiles_s'];text+=f"| {n} | {r['successes']}/{r['n']} | {r['falls']} / {r['unintended_contacts']} | {r['force_failures']} | {f['p50']:.0f} / {f['p95']:.0f} / {f['max']:.0f} | {r['max_descent_m_s']:.3f} | {h[0]:.2f} / {h[1]:.2f} |\n"
 return text
primary=['deployment_status_fresh_05','soft_landing_actions_clean_01','deployment_status_fresh_noisy_05','soft_landing_actions_noisy_01','soft_landing_baseline_fresh_clean_01','soft_landing_actions_fresh_clean_01','soft_landing_baseline_fresh_noisy_01','soft_landing_actions_fresh_noisy_01','deployment_geometry_novel_04','soft_landing_contact_novel_01']
experiments=['soft_landing_no_contact_clean_01','soft_landing_limiter_confirmation_noise_01','soft_landing_limiter_fresh_clean_01','soft_landing_limiter_fresh_noise_01','soft_landing_limiter_novel_01','soft_landing_clean_01','soft_landing_rl_clean_01','soft_contact_bc_clean_01','soft_contact_bc_clean_02','soft_contact_ankle_bc_clean_03','soft_landing_dagger_contact_clean_04','soft_landing_dagger_ankle_clean_04','soft_landing_staged_ankle_clean_05']
free=shutil.disk_usage(root).free/2**30
text='''# Sensor-based soft landing: measured progress

Actual seat impacts improved on matched simulation trials. Most improvement comes from a generic ankle-command limiter around the existing learned student. Learned contact feedback adds a modest average-force reduction but has not established a reliability gain. None of these candidates passes all unchanged safety criteria; no hardware deployment or checkpoint promotion occurred.

The actor remains `runs/deployment_status_mask_05/student.pt`, unchanged. The principal contact-aware experiment pairs it with `runs/contact_estimator_02/contact.pt` and the `actions` mode of `humanoid_sitting.soft_landing`. The stronger clean ablation disables contact feedback and retains the fixed ankle limiter. Original teacher, PPO, DAgger and strict student checkpoints remain preserved.

''' +table(primary)+'''
All rows include failed attempts. Hold durations describe physical stable-hold intervals; a later stable pose never erases an earlier safety violation. Primary contact-aware trials have zero falls, unintended contacts, torque-limit violations and excessive-descent violations; remaining failures are excessive seat forces. The historical novel-shape baseline uses checkpoint 04; the other baseline rows use checkpoint 05. They have identical learned weights and clean status preprocessing, but the recorded checkpoint identities are retained.

Confirmation bank 25001 crosses 12 rigid geometries with three standing configurations and seed 3401. The candidate was frozen before fresh bank 26001: eight new geometries, seeds 3501/3502, and three configurations (48 trials per sensor condition). Geometry families are chairs, benches, boxes and rigid couch-shaped surfaces. Height is 0.275–0.325 m, depth 0.4–0.6 m, width 0.75–1.05 m, with existing small position/orientation perturbations. Novel bank 24001 uses held-out elliptical/chamfered seat footprints, four geometries and three configurations. These remain aligned sitting transitions, not approach/alignment, compliant-cushion or physical-robot tests.

Noisy perception means 5 mm depth noise plus 10% dropout, followed by the same depth-only geometry estimator; clean sensors still include the existing encoder/IMU/torque emulation noise and 20 ms delay. New body-mounted camera captures were valid in 48/48 clean and noisy cases. Severe occlusion is still refused by the existing perception gate; it was not counted as a sitting success.

[Impact distributions](../runs/soft_landing_01/impact_distributions.png) · [Paired physics](../runs/soft_landing_01/paired_physics.png) · [Paired transition video](../runs/soft_landing_01/paired_transition.mp4) · [Complete metrics](../runs/soft_landing_01/final_metrics/metrics.json) · [Matched-condition and ablation audit](../runs/soft_landing_01/paired_analysis.json).

The paired video uses identical initial root/joints, seat estimate, seed and geometry (bank 25001, geometry 4, configuration 0). The original student reaches a visually seated pose but fails the force criterion at 973 N; the contact-aware limiter passes at 512 N with an 8.36 s hold. Both videos were rendered and their representative frame sheets inspected. Their one-environment sensor-noise realization differs from the 36-environment benchmark; the videos are a separately measured matched pair.

**Controller and information boundary.** The packet has 109 channels: 23 joint angles, 23 joint velocities, 23 estimated motor torques, 3 gyro channels, 3 gravity-direction channels, 23 previous joint targets, 10 depth-estimated seat fields, and one sit command. Confidence/validity are masked for motion (107 effective channels); no verified foot-feedback channels are included. Geometry is a latched RGB-D estimate, not simulator geometry. Gyro/gravity and EMA applied-torque are physical-sensor surrogates; real IMU/torque calibration, proposed camera mount and physical motor mapping remain unverified.

The frozen student supplies learned absolute body targets. Its body GRU still carries implicit timing; only the ankle head uses instantaneous proprioceptive feedback. It is not a fully closed-loop learned whole-body transition. Fourteen simulated hand targets remain parked. No exact root position/velocity, COM, simulator contact, ground-truth furniture geometry or teacher action enters the actor or contact-aware controller. The software boundary audit rejects prohibited packet fields. Simulator state/forces appear only in demonstrations, teacher labels, training rewards and diagnostics/evaluation.

For knee encoder mean >1.0 rad, the generic limiter caps ankle-pitch target changes at 0.004 rad/control step (0.2 rad/s). Contact-aware operation decreases this to 0.002 rad/step as predicted contact rises, and caps hip/knee changes at 0.0025 rad/step after predicted contact probability >0.65 to make transfer gradual. The fixed-limiter ablation disables both contact effects. This is explicitly an authored governor, not learned action improvement. It uses physical-equivalent observations only, with no object-specific trajectory tuning.

**Safety unchanged.** The force ceiling remains 632.53 N (2 mg); unintended contact must stay <=10 N at every 5 ms contact-history sample; downward root speed <=0.8 m/s; original effort limits, falls, pelvis-floor checks, upright posture, initial-standing verification and >=3 s continuous seated hold remain enforced. No automatic resets or root constraints are used. Original actuator gains and collision/seat physics are unchanged. The simulated effort limits are not verified physical G1 limits. Body velocity is sampled at 20 ms; force peaks use 5 ms history. Post-contact velocity can already contain rebound, so the metrics separately identify the last pre-contact and first post-contact samples.

The prior [contact-binding audit](../runs/contact_binding_validation_02/audit.json) confirms all 44 robot bodies plus the seat and includes a detected/rejected knee-contact positive control. Missing-rigid warnings refer to non-rigid prims included by the upstream global view. No warning suppression or collision change was introduced here.

**Demonstrations and learning.** Recorded 24 teacher trajectories in `datasets/soft_contact_teacher_01` and 96 in `soft_contact_teacher_02` (96 unique geometry/initialization cases; the first 24 repeat a subset with a different sensor-batch realization). All teacher executions passed; maximum teacher force was 475.71 N. Recorded another 24 trajectories at guarded student states in `datasets/soft_landing_dagger_01`, all passing with peak 494.86 N. These training banks are 22001, not holdout banks 24001/25001/26001. All sources retain 1,000 synchronized 20 ms samples, physical-equivalent packets, executed full joint targets, q/dq, IMU, torque, privileged teacher targets, seat metadata, contact diagnostics and outcomes. Teacher actions are used only in the explicitly teacher-driven demonstration sets; DAgger expert labels never drive learner actions.

Contact labels are previous post-step pelvis/thigh upward seat load at current packet time, with >0.05 mg indicating contact and clipped [0,1] load fraction regression. Labels are never appended to deployment packets. The 0.05 mg training label does not replace the original >0.25 mg load-bearing success requirement. Label timing is 20 ms; this is not a 5 ms onset sensor or an estimator of forces above full body weight.

Estimator 02 trains on 72 teacher trajectories and withholds 24 trajectories from four entire geometries. Teacher-only false-positive rate is 0.11%, false-negative rate 0.048%, load MAE 0.90% mg. On actual clean student transitions: false-positive rate 1.13%, false-negative 0.253%, load MAE 3.48% mg; median first detection is 120 ms early, including premature detections >200 ms in 6/36 episodes. A new 32-sample (640 ms) window estimator avoids episode-length recurrent timing, but teacher-only training did not resolve student onset errors.

Estimator 04 adds guarded student-state data with privileged labels. On the clean 36-case benchmark it improves load MAE to 1.27% mg, false-positive rate to 0.782% and false-negative rate to 0.039%; premature detections >200 ms remain in 3/36 episodes. One startup prediction is spurious, so the minimum onset-delay statistic (-11.78 s) must not be interpreted as physical contact. Startup torque consistency, negative-example coverage and confidence calibration remain unresolved. It still achieves only 33/36 and is not a promoted controller.

''' +table(experiments)+'''
Slowing the body GRU's final descent was tested and rejected after new falls/contacts and a 2.01 kN maximum. New teacher-action imitation runs (1,600 warm-start steps; 6,000 contact-weighted steps from scratch; 1,200 ankle-only steps) did not improve safety. A 400-step ankle-only DAgger update also failed severely even with the same fixed limiter; its checkpoint and failure evidence remain preserved and must not be used as a working controller.

A final posture-gated variant retains the preserved ankle controller through standing and blends the learned head as measured knee mean rises from 1.0 to 1.2 rad. Initial standing verification passed 36/36, but the transition still failed 0/36, with 18 falls, 32 unintended contacts and a 7.43 kN maximum. The learned-action problem therefore persists during descent, even with initial balance protected. It is rejected and its exact startup-controller configuration is saved.

The RL comparison is intentionally narrow: 24 training episodes / 24,000 control transitions, one episode-level REINFORCE update of two stochastic phase-governor rates. It is not full actor PPO refinement. The mean changed from [0.96,1.02] to approximately [0.9626,1.0176]; unchanged actor weights were used for held-out evaluation. It reached 18/36 but did not improve high-force tails (1.49 kN maximum). Reward gain or three extra successes is insufficient evidence of safe soft landing, so extensive full-policy RL was not launched.

Across the five matched contact-feedback/fixed-limiter sets (180 trials, 24 geometry clusters), contact feedback wins only one additional success (166 vs165), but reduces mean per-attempt peak force by 21.12 N. Exploratory geometry-cluster bootstrap gives [-34.62,-8.15] N for the mean force difference and [-2.19,+3.57] percentage points for success difference. This supports a modest average-force effect, not a reliable success or worst-case safety improvement. The larger reduction relative to the original student comes from command smoothing. No consistent improvement from newly learned action weights or RL refinement was demonstrated.

**Resume.** Existing outputs refuse overwrites. Use a fresh output directory; all executed argument sets are retained in each run's `command.json`, and scripts `run_soft_landing_*checks.sh` retain the exact test matrix. Do not launch the failed imitation checkpoints as defaults.

```bash
cd /home/kadir/humanoid-sitting
bash scripts/run.sh python ../scripts/evaluate_geometry_reference.py \\
  --deployment-checkpoint ../runs/deployment_status_mask_05/student.pt \\
  --contact-estimator ../runs/contact_estimator_02/contact.pt \\
  --soft-landing-mode actions \\
  --deployment-perception ../datasets/deploy_rgbd_confirmation_05 \\
  --deployment-action-rate .05 --compact-recording --summary-only \\
  --geometry-seed 25001 --geometry-count 12 --seeds 3401 \\
  --output ../runs/soft_landing_resume_01 --device cuda:0
```

Add `--disable-contact-feedback` for the fixed-limiter ablation. Use `runs/soft_landing_perception_fresh_01/depth_noise_5mm_dropout_10pct` with bank 26001, count8 and seeds3501/3502 for the corresponding noisy holdout. Full records require removing `--summary-only`; add `--record-deployment-inputs` only when physical packets are needed. No broad RL run should start until startup contact errors and force-limit failures are resolved on the same benchmarks.

'''
text+=f'Approximately {free:.2f} GiB remains free. New data/artifacts stayed within the 500 MiB budget and above the 30 GiB free-space floor; exact usage and hashes are in `runs/soft_landing_01/final_integrity.json`. RTX4090 GPU0 was used; video startup confirms renderer GPU0, while the RTX4070 remains the desktop GPU. No installations, NVIDIA driver changes or hardware commands were made.\n'
(root/'docs/soft-landing-progress.md').write_text(text)
readme=root/'README.md'
if not (out/'source_before/README.md').exists():shutil.copy2(readme,out/'source_before/README.md')
old=re.sub(r'\n\n\*\*Latest soft-landing milestone:\*\*.*?\n\n','\n\n',readme.read_text(),count=1,flags=re.S).replace('**Latest transfer milestone:**','**Preserved transfer baseline:**',1).replace('Approximately **35.1 GiB** remains free;','At that earlier milestone, approximately **35.1 GiB** remained free;',1)
intro=f'''**Latest soft-landing milestone:** the unchanged sensor-based student plus a generic ankle-command limiter and learned contact estimator passes **33/36 clean, 32/36 noisy**, and **44/48 clean / 46/48 noisy** on a fresh furniture bank. Zero falls/unintended contacts in these tests; force-limit failures remain. Command smoothing provides most improvement; contact feedback adds a modest mean-force reduction, but new action imitation and the bounded RL pilot did not establish safe learned-policy improvement. All original checkpoints are preserved; no hardware deployment. [Measured report and resume commands](docs/soft-landing-progress.md) · [Paired video](runs/soft_landing_01/paired_transition.mp4) · [Impact distributions](runs/soft_landing_01/impact_distributions.png). About **{free:.2f} GiB** remains free.\n\n'''
readme.write_text(old.replace('# Humanoid sitting\n\n','# Humanoid sitting\n\n'+intro,1))
# Source snapshot, original teacher checksums, unchanged checkpoint identities.
snap=out/'source_final';snap.mkdir(exist_ok=True)
files=[root/'sitting_task/src/humanoid_sitting/soft_landing.py',root/'sitting_task/src/humanoid_sitting/deployment.py',root/'sitting_task/src/humanoid_sitting/sim_deployment_sensor.py']+list((root/'scripts').glob('*soft_landing*'))+[root/'scripts/evaluate_geometry_reference.py',root/'scripts/train_deployment_student.py',root/'scripts/train_contact_estimator.py',root/'scripts/train_window_contact_estimator.py',root/'docs/soft-landing-progress.md']
manifest={}
for p in files:
 target=snap/p.relative_to(root);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target);manifest[str(p.relative_to(root))]=hashlib.sha256(p.read_bytes()).hexdigest()
(out/'source_final_sha256.json').write_text(json.dumps(manifest,indent=2))
teacher=root/'baselines/geometry-conditioned-rigid-v1';tm=json.loads((teacher/'sha256.json').read_text());bad=[p for p,h in tm.items() if hashlib.sha256((teacher/p).read_bytes()).hexdigest()!=h]
assert not bad,bad
baseline_hash={}
for line in (out/'baseline_sha256.txt').read_text().splitlines():
 h,p=line.split(maxsplit=1);assert hashlib.sha256((root/p).read_bytes()).hexdigest()==h;baseline_hash[p]=h
roots=list((root/'runs').glob('soft*'))+list((root/'runs').glob('contact_estimator*'))+list((root/'datasets').glob('soft*'))+[root/'baselines/soft-landing-experimental-v1'];artifact=sum(p.stat().st_size for r in roots for p in r.rglob('*') if p.is_file());logs=sum(p.stat().st_size for p in (root/'logs').glob('*') if p.is_file() and (p.name.startswith('soft_') or p.name.startswith('contact_estimator')))
assert artifact+logs<500*2**20,(artifact+logs)/2**20
integrity={'teacher_files_checked':len(tm),'teacher_mismatches':bad,'original_checkpoint_sha256':baseline_hash,'artifact_MiB':artifact/2**20,'new_logs_MiB':logs/2**20,'free_GiB':free,'data_budget_MiB':500,'minimum_free_GiB':30,'hardware_deployed':False,'new_policy_promoted':False}
(out/'final_integrity.json').write_text(json.dumps(integrity,indent=2));print(json.dumps(integrity,indent=2))
# Immutable experimental baseline references all retained data/results without duplicating them.
b=root/'baselines/soft-landing-experimental-v1';(b/'manifest.json').write_text(json.dumps({'status':'experimental; force failures remain','actor':'runs/deployment_status_mask_05/student.pt','contact_estimator':'runs/contact_estimator_02/contact.pt','mode':'actions','fixed_limiter_ablation':'--disable-contact-feedback','evidence':[r['run'] for r in rows],'not_promoted':True,'teacher_and_actor_checkpoints_unchanged':True},indent=2));(b/'sha256.json').write_text(json.dumps({str(p.relative_to(b)):hashlib.sha256(p.read_bytes()).hexdigest() for p in b.rglob('*') if p.is_file() and p.name!='sha256.json'},indent=2))
