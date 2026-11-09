# Traction-safe weight transfer and small PPO comparison

Work directly from the Mac through SSH to `kfuUbuntu`; never start another coding agent. Study root: `/data/humanoid-sitting-perception/traction_transfer_01`. Previous studies remain immutable. No policy promotion is authorized by this protocol.

## Physical repair and admission

The reference preserves the complete first10s approach. A quasi-static foot-wrench observer uses joint positions, applied motor torques, IMU gravity and the robot's static kinematic/inertial model. Bounded horizontal foot-goal feedback relieves internal sideways force. The reference planner uses the measured seat plane and body geometry to select an upright or forward-leaning resting target. It can fall back to its bounded0.66rad lean when its requested model-clearance margin is infeasible; this is not a claim of model-certified clearance. Full simulation validation is mandatory. Live simulator pose, geometry and contact truth remain reset/reward/evaluation labels, not controller or network inputs.

The selected physical probe is `planned_v2_clear6_01`: all18 trials across38/40/42cm and-5/0/+5deg remain stable for20s with zero slipping, unwanted contact, torque-limit, joint-limit, impact or fall failures. Strict clean success remains16/18. At42cm/-5deg, each repeated placement has one20ms touchdown sample at roughly0.914 friction utilization, above the conservative0.90 reporting margin but below physical capacity1.0. This is retained as a margin violation, not relabeled as strict success.

Before main training, the new environment must exactly reproduce the selected reference's saved motion, pass indexed-reset isolation after feedback/plan memory is active, and pass two optimizer smoke evaluations. Admission requires18/18 stable full20s trials, zero all-tick slipping/unwanted contacts/physical failures, no utilization above1.0, and no additional conservative margin ticks relative to the selected reference. The strict0.90 metric, physical failure definitions and reward are unchanged. This new-study admission deliberately separates physical slipping from a conservative transient margin flag; its rationale is saved before any new PPO starts. The frozen previous studies and their gates remain unchanged.

## Learning contract and budget

Fresh seeds58311/58317 each run one uninterrupted64-rollout process:32 policy decisions per rollout,5 physics ticks per decision,18 environments. Each receives184320 physics transitions,368640 total, including indexed-reset warmup and reference-controlled ticks. Initial global settling450 ticks per seed is excluded. First8 scheduled rollouts are critic-only;56 are actor-enabled. Empty learning batches have no Adam steps. Report actual actor/critic optimizer steps and control/reference/warmup/unfinished accounting, not just scheduled rollouts. Separate8-rollout optimizer smokes use23040 transitions per seed,46080 total; smoke weights are not used for main training.

Every episode starts standing. The first11s remain reference-controlled; PPO adjusts the ongoing weight transfer and resting posture thereafter. This is reference-assisted residual PPO, not unassisted standing-to-sitting discovery. All nine development geometries are eligible; none are excluded by a hidden training selector. Actor and critic receive414 channels: the inherited406 sensor/reference/clock channels plus six filtered estimated foot-wrench values and two feedback offsets. Actions remain23 bounded absolute body-joint residuals;14 finger commands stay fixed. Existing slew, joint bounds and prior bridge residual spans are unchanged.

Reward is the frozen `run_progressive_seat_ppo.py` reward, including valid support, seat load, all37 physical joint torques squared and friction utilization. No new reward or criterion is tuned after main launch. Both actor output layers start exactly at zero, giving reference motion before learning, though random hidden weights differ by seed. Actor/critic/Adam state continues in the same process.

## Comparisons and withheld evaluation

Development comparisons include the current corrected reference, PPO56211/56217, the previous bridge reference, PPO58111/58117, the new traction reference and both final new policies. Use matched physical/sensor starts and the exact expected checkpoint. Both full-budget new checkpoints must be frozen before any reserved evaluation. No checkpoint selection or automatic policy promotion.

The fresh reserved bank contains eight geometries at38.5/39.5/40.5/41.5cm with slopes+/-3,+/-4,+/-1,+/-3deg respectively, each repeated at two placements. It was reserved before this repair and is excluded from physical tuning, learning, calibration and selection. Holdout means withheld from this study; do not imply broad arbitrary-surface generalization.

All final evaluations run20s. Report stability and physical failures first, then strict clean results and remaining margin/slip/contact cases by seed, height and slope. Compare whole20s, final10s and final5s summed squared torque, absolute torque, mean instantaneous all-joint RMS Nm, and leg/upper-body contributions only on common full-duration successes. Also report strict-policy subsets. Separate controller-repair gains from incremental PPO gains. No savings claims from failed fragments. Torque is an effort proxy, not electrical energy or evidence of a global optimum.

## Verification and completion

Freeze source, configuration, bank and prior-result hashes only after all preflight checks pass. Verify source preservation, finite changed actor/critic weights, critic-only actor identity, actual Adam steps, budgets, checkpoint loads, initial-state matching, the locked reference prefix, fixed fingers, bounded commands and holdout timing. Inspect geometry availability/errors before any failure. Preserve failed outputs and sources, document any narrow execution/report repair, and never change a running experiment to improve its scores.

Run independent verification twice, export the report/replay/plots/results and copy them to `/Users/kadir/humanoid-sitting-control/validated_traction_transfer_01/final_results`. Verify the export manifest and inspect the plot and replay. Update local/remote handoffs, deliver an honest comparison, then pause the existing completion heartbeat. Two seeds and repeated placements on nearby rigid slabs are limited evidence.
