# How I would structure a fresh sitting-RL project

I would keep PPO, Isaac Lab, the G1 model, the verified kinematics and the depth
representation. The main change would be what the policy controls and how it
practices. I would train a policy to own the sitting motion and its timing. The
current experiment trains small corrections to a frozen, already learned motion.
This is a proposed next research direction, not a change to the active capped
impact experiment.

| Aspect | Current experiment | Fresh design I would test |
|---|---|---|
| Motion ownership | Frozen sensor05/contact02 motion plus23 learned joint corrections, bounded to0.08rad | Trainable whole-body joint-target policy under physical joint, speed and torque limits |
| Timing | Motion generator advances at its fixed rate; corrections can affect physical timing indirectly | Policy can slow, settle, pause and shift weight in response to current geometry and contact |
| Practice | Every attempt starts standing; broad rigid-seat prototypes with randomized starts | Start seated, then progressively move reset states earlier in the sit; retain some full standing trials throughout |
| Actor inputs | Current151-channel sensor-derived packet and frozen-controller outputs | Depth-derived support geometry, joint/IMU/torque/contact estimates, previous actions and short history; no runtime truth |
| Critic inputs | Same sensor inputs as the actor | Consider a training-only critic with simulator state/contact labels; never used for runtime decisions |
| Initial behavior | Existing motion generator stays in the control path | Use successful motions to initialize or regularize a trainable policy, gradually reducing imitation dependence |
| Rewards | Several terms changed around an inherited objective | Establish reliable supported sitting first, then isolate landing and economy terms; audit their scales |
| Validation | Paired full20-second tests, now with integer hold scoring | Keep those tests; also test extended sitting, perturbations, noisy sensing, and systematically held-out geometry |

## Learn the skill from the end backwards

First learn to remain seated, upright and stable, with weight genuinely supported
by the seat. Use physically valid seated initial states and small perturbations.
Then initialize a few centimetres above the surface and learn touchdown and
weight transfer. Next start halfway down, then from standing. These are simulator
reset curricula; the eventual policy must still pass complete standing-to-sitting
trials without assistance. Keep the same motor physics throughout.

That puts much more training time near contact and seated balance. In the latest
128-update study, qualifying supported sitting occupied only420–471 robot-seconds
out of2621 simulated robot-seconds. A curriculum would be an experiment to make
those useful experiences more frequent, not a guarantee that it improves learning.

I would let a recurrent policy or a policy with a short sensor history choose
joint targets. The first comparison would keep low-level position control and
the existing actuator limits. Learning motor torques from scratch is not needed
to give the policy control over the full sitting movement.

## Reuse demonstrations without freezing the strategy

The project already has successful teacher and sensor-policy motions. They can
provide initial examples of plausible sitting. I would use those examples to
initialize the policy or add an imitation term early in training, then let task
reward increasingly determine the motion. A demonstration should guide learning,
not require the robot to follow the same clock when the seat is higher or lower.
Combining example-motion objectives with task objectives is an established idea
in [DeepMimic](https://arxiv.org/abs/1804.02717). That paper is support for the
methodological idea, not proof of a deployable G1 sitting solution.

If the goal were strictly learning from random weights without demonstrations,
I would lean more heavily on the seated-to-standing-reset curriculum and expect
a larger training budget. Discarding useful existing demonstrations is not my
preferred way to restart this project.

## Keep the robot's observations honest

The actor should continue to use the common body/seat/floor frame reconstructed
from depth and kinematics. Include clearance, relative seat orientation and
usable area, validity/age information, encoders, joint velocities, IMU, previous
commands and available torque/contact estimates. A short history helps it
interpret contact and temporary occlusion. Foot-force readings should only be
actor inputs if the intended robot actually has suitable sensors.

For training, I would consider an asymmetric critic: the evaluator of an action
can see the simulator's answer key, while the actor making decisions cannot.
The critic is discarded at deployment. This separation is used in
[Asymmetric Actor Critic for Image-Based Robot Learning](https://arxiv.org/abs/1710.06542).
It would be a deliberate training-design change: both actor and critic in the
current capped-impact study remain sensor-only. Ground truth would still never
enter the perception estimator or deployed policy.

## Make completion the priority, economy the refinement

A good sitting episode should reach the intended support region, transfer weight
gently, remain balanced, and maintain a supported posture for an extended hold.
Reward low leg effort only after genuine stable support. Keep feet available for
balance rather than demanding exactly zero foot load. Include torso effort and
actuator-normalized torque or mechanical-work diagnostics; minimum leg torque
alone does not guarantee a good whole-body posture.

The actor should also get a useful learning signal before impact, when it can
still slow down. I would compare a bounded near-contact descent-speed term with
contact-based impact terms as separate experiments. Whole-episode failures must
remain visible, including forbidden contacts late in a sit. Discount and rollout
length should be chosen in seconds for this task, rather than inherited without
checking whether they connect earlier choices to later outcomes.

Start with a generous rigid platform, then vary height, depth, width, offset,
yaw, friction and sensor delay/noise. Add thin edges, unusual shapes, compliant
cushions and approach/navigation later, once each can be represented and tested
faithfully. A surface outside the robot's reachable or stable sitting region
should be rejected rather than counted as an obligation to sit anywhere.

## Scale the learning loop, not just the number of updates

The present camera/kinematics pipeline makes16-robot rollouts expensive. A fresh
full-motion policy will need substantially more experience than the small
continuation studies. I would profile and batch the geometry and physics path,
increase parallelism as memory allows, and keep real sensor-derived actor inputs.
I would not speed it up by silently substituting perfect simulator geometry for
the actor's measured geometry.

Use multiple training seeds and validation checkpoints, reserve genuinely unseen
test geometries, and report failed attempts as carefully as successes. A language
or vision-language model could later select a seat and issue a goal. For learning
touchdown, balance and weight transfer, I would still start with a dedicated
feedback motor policy trained with PPO.

The capped-impact study remains useful: it tests a specific reward-scaling
diagnosis with the current controller held fixed. Even a positive result would
not establish that small residual corrections are the best long-term architecture
for sitting on varied surfaces.
