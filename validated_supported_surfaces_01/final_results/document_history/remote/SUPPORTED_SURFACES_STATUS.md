# Standing-to-supported sitting and leaning — active PPO pilot

October12,2026. The user's current goal is supported postures across heights/slopes, followed by learning to reach them from standing, with effort secondary and unseen-surface tests. This supersedes the completed seated-effort study. No previous code/data/checkpoint was removed; all46 prior bounded-study frozen hashes still match.

## Validated before launch

- New arbitrary-angle plane geometry, finite-face contact checks and calibrated rear depth view. The actor uses no simulator surface pose, body translation or contact force. Downward view excluded after identifying front-face selection errors.
-13 training goals all held for20seconds:25/30/35/40cm level seating; tilted sitting;45/60deg inclined pelvis-supported leaning. The full26-goal initial validation passed25; one unseen changed-outline goal has98.7% qualifying ticks but misses final2seconds continuous support and remains in the heldout test.
- Separate steep-pose tests found5/18 valid75deg back-supported poses. Three are reserved for unseen evaluation. Vertical90deg candidates remain unvalidated. Rear `logo_link` is the fixed torso shell; only its rear face is admitted in this NEW task. Original study labels remain unchanged.
- Stable standing motor preload calibrated with45 preserved candidates. Initial nominal command had tipped forward. Selected corrected state held20seconds across all29 final train/test geometries with no physical failure and constant commands.
- Standing depth validation2929/2929 samples available;95th-percentile plane-normal error0.178deg and offset error2.571mm. This is ideal simulation evidence, not real-camera accuracy. These metrics describe standing, not yet learned motion.
- Reset isolation exact: no other-row state/command/observation change and no global physics-clock advance. Five geometry/tracker unit tests passed. A2-update development smoke test (1664transitions) changed13 finite checkpoint tensors.
- Interpolated intermediate poses were physically/sensor tested. Only admissible stages retained before training. Low seats currently have only standing/supported reset endpoints, an exploration limitation. No in-episode teleportation.

## Running now

Remote root `/data/humanoid-sitting-perception/supported_surfaces_01`.
Study `transition_ppo_01`, persistent tmux `supported_transition_ppo_01`.
Coordinator `scripts/run_supported_transition_study.py`.
Frozen contract: `docs/SUPPORTED_TRANSITION_PROTOCOL.md`;36 source/bank hashes in study protocol.
Two fresh seeds54211/54217 on4090/4070,224updates each,26environments,64rollout steps;372736transitions/seed including indexed-reset warmup. Initial0.5s settle excluded. Last64updates use standing only. Earlier stages select the nearest prevalidated start and include standing resets.

At the latest observed snapshot, updates60/59 had completed with finite weights and no full episode successes yet. This is early training, not the final matched evaluation. Do not claim learned sitting/leaning or generalization from endpoint validation. Early repeated failures consume substantial warmup ticks; final report must separate active/warmup budgets and curriculum-reset successes from standing transitions.

333 sensor-only channels; absolute joint targets with fixed physical range/rate limits. Reaching and keeping support has3/s reward; effort cost is capped at0.05/s. No goal pose or true geometry in actor/critic. Ground truth used only in training/reset/evaluation labels.

After both runs, coordinator automatically tests fixed stance and identical final checkpoints from standing for20seconds on13training and16unseen configurations. Unseen:27.5/32.5/37.5cm heights,10/30deg slopes, narrower elliptical/chamfered outlines, and75deg back support. Final5seconds continuous supported rest plus no physical failures defines successful transition.

## Completion workflow

Existing same-chat heartbeat `finish-resting-ppo-comparison` is reactivated and renamed “Finish supported-transition PPO comparison”; it stays quiet during normal progress. Do not change the frozen experiment or launch extra training. Previous capped/resting studies stay complete.

Independent verifier prepared: `scripts/verify_supported_transition_study.py --study /data/humanoid-sitting-perception/supported_surfaces_01/transition_ppo_01 --output <fresh-directory>`. It intentionally refuses incomplete studies and any exact startup mismatch; diagnose before weakening a check. Inspect physical failures and sensor errors on valid motion, not just reward. Copy verified final results/plots to Mac `validated_supported_surfaces_01/final_results`, inspect plots, update handoffs, deliver comparison, then pause the same monitor. No automatic policy promotion.
