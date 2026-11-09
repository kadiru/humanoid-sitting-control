# Independent distance interface and bounded PPO pilot

The observation interface now preserves reliable clearance when seat edges are uncertain. The first144 inputs and the parent action function were preserved at initialization; seven independent distance channels were added.

## Paired full-horizon results

| Bank | Frozen controller | Trained policy | Worst force: frozen → trained | Newly lost successes |
|---|---:|---:|---:|---|
| Known regression | 16/16 | 16/16 | 520.5 → 616.5 N | [] |
| Fresh shapes / starts | 12/16 | 13/16 | 1078.1 → 1176.1 N | [] |
| New starts | 7/16 | 8/16 | 6447.9 → 6447.9 N | [] |

![PPO comparison](ppo_comparison.png)

![Distance interface](distance_interface.png)

## Interpretation

Exactly 32 updates / 32768 transitions. Final checkpoint evaluated without selection. Promotion checks: {'heldout_success_improved': True, 'no_lost_successes': True, 'no_worse_bank_peak_force': False}. No defaults were changed.

The held-out banks contain16 attempts each; this is a small paired experiment, not evidence of universal sitting or hardware readiness. Rigid, aligned simulation seats only. Whole-trial failures, impact extremes, all paired case outcomes and distance coverage are retained in results.json.

Runtime estimates use depth, joint kinematics and IMU. Simulator state supplies reward/evaluation labels only. Patch and distance validity are independent; unavailable distance is explicit, not held indefinitely. The engineering error bound is conditional, not calibrated uncertainty.

New-input learning evidence: {'new_weight_max': 0.0023200437426567078, 'initial_new_weight_max': 0.0, 'normalization_bitwise_unchanged': True, 'masked_distance_action_difference_max': 0.000990728847682476, 'valid_distance_missing_patch_samples': 662, 'valid_distance_missing_patch_action_difference_max': 0.0008721519261598587, 'note': 'Offline input masking measures action sensitivity, not a closed-loop success ablation.'}

Preservation: {'frozen_sources_and_original_checkpoints_unchanged': True, 'original_parent_sha256': '2c0f6907196be68bce9a1ee2e6b6ef6e4c82bf0ecdff622dc55b81fe8ef58d13', 'defaults_changed': False}

Checkpoint: `/data/humanoid-sitting-perception/support_split_ppo_train_01/final.pt` SHA256 `ece102b139323af18bd1c7aad6d8fb06e478f38946f07e52406a31bd4acc7dda`.

Sources/protocol, interface checks, training verification and paired per-case results accompany this report.
