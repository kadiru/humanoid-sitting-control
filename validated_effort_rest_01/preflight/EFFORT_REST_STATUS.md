# Direct effort-minimization PPO — running

October 12, 2026 (PDT). The user clarified that the objective is minimizing total
joint effort while supported, analogous to using skeletal alignment and external
support to reduce muscle effort. The new reward directly favors lower sum of
squared physical Nm across all 37 actuated joints, subject to unchanged valid
support and physical failure criteria. Effort is now the primary optimization
objective. This phase begins already supported; it does not teach reaching the
surface from standing.

## Implemented and validated

- Separate new code: `scripts/run_effort_rest.py` and `humanoid_sitting/effort_rest.py`.
- The remembered reference motor command is outside trainable network layers.
  Policy actions are bounded absolute offsets; zero action exactly retains the
  reference and persistent bias cannot integrate away. Motor gains and torque
  limits are unchanged.
- Four controller/reward tests and indexed reset isolation pass.
- Fixed posture passes 13/13 training and 16/16 held-out surfaces for 20 seconds.
- The two-update optimizer smoke changed finite actor/critic weights. Its final
  policy passes 13/13 full 20-second evaluations (at least 99.9% qualifying ticks),
  loaded checkpoint confirmed, all initial state/command/observation arrays
  bitwise matched to fixed baseline. Maximum requested start adjustment 0.212°.
- Random exploration with std .30 and .15 stays upright on all 13, but one lowest
  seat encounters forbidden contact; noisy full-rest scores are 2/13 and 3/13.
  These remain failures. The pilot uses std .15. Requiring every random probe to
  succeed was removed as a launch gate, with original orchestration/source/probes
  and the decision record preserved. Physical success criteria were not relaxed.

## Training and evaluation

Remote root: `/data/humanoid-sitting-perception/effort_rest_01`.
Study: `ppo`; persistent tmux `effort_rest_ppo_01`.
Coordinator: `scripts/run_effort_rest_study.py`.
Protocol: `docs/EFFORT_REST_PROTOCOL.md` and `ppo/protocol.json`.

Two fresh seeds 54511/54517 run on physical GPUs 0/1, each 256 updates × 64 steps ×
26 environments = 425,984 transitions. Total 851,968, with indexed-reset warmup
included and separately counted; initial global settle and the 3,328-transition
smoke are excluded. All starts are supported. Actor and critic use the same 402
sensor-derived channels, including the remembered command; no true geometry,
root translation, contact labels or goal pose enters the policy.

Snapshot: updates 34/33 completed; successful training episodes 6/9. All weights
remain finite. Requested mean start offsets are about 1.9°/1.8°, bounded around
the stable reference. Training episodes include random exploration; these counts
are not the final deterministic comparison or evidence of effort saving.

The coordinator automatically evaluates each identical final checkpoint from
matched supported starts for the full 20 seconds on 13 training and 16 held-out
surfaces. Compare physical failures first, then whole-trial and final-ten-second
summed squared torque, summed absolute torque, joint RMS and leg/upper-body
contributions on common successful trials. A 10% saving with retained stability
is a useful target; no global minimum or arbitrary-surface capability is claimed.
The hold-out bank includes intermediate heights, 10/30° slopes, changed outlines
and 75° back support; it was used by prior studies, but is withheld from this
policy's training. The historically borderline outline endpoint is retained.

## Completion workflow

Run `scripts/verify_effort_rest_study.py --study /data/humanoid-sitting-perception/effort_rest_01/ppo --output <fresh-directory>`
after both runs and four evaluations finish. Verify frozen sources/banks, prior
study preservation, full budgets and active/warmup accounting, finite changed
weights, correct final checkpoint loading, exact matched startups, full-duration
rollouts, bounded learned targets and constant fixed commands. Inspect tracking
accuracy during valid motion. Torque is an effort proxy, not electrical energy.
Two seeds on finite rigid surfaces do not prove broad generalization.

Deliver verified results and plots here under `final_results`, update local and
remote handoffs, report whether PPO reduces effort without losing support, and
pause the existing completion monitor. Do not silently change this frozen study,
add training, select a checkpoint using held-out scores or promote a policy.
Previous standing-transition, bounded-resting and capped-impact studies remain
complete and unchanged.
