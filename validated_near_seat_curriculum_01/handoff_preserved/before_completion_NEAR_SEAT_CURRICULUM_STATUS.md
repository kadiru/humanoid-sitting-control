# Near-seat curriculum status

Main training is active. Root: `/data/humanoid-sitting-perception/near_seat_curriculum_01`. Frozen protocol: `docs/NEAR_SEAT_CURRICULUM_PROTOCOL.md` and `ppo_01/protocol.json`.

Validated near-seat reference:18/18 strict20s transfers on9development geometries from two distinct starts;~75%BW onseat. Fixed posture18/18. Matched final5s squared-torque reductions2.67% (.1rad) and35.13% (.2rad). Near harnessstage5/stage4 and learned smoke each18/18strict. Reset isolation stages1–5passed; captured approach exactly reproduces currentreference. New standing reference3/18,2physical failures; no standing claim.

Main seeds57111/57117 each128updates,368640physics transitions; first8critic-only. Independent curriculum advancement only after all18strict no-slip/friction/contact successes and>=60% seat load. Firststage5/stage4 gates passed both; nowstage3. See live `ppo_01/status_{seed}.json`, `seed_{seed}/curriculum.json`, and `finalization_status.json` for current state.

Workers:tmux near_seat_ppo_57111/near_seat_ppo_57117; finalizer:near_seat_finalize_01. No additional training or policy promotion. Same-chat completion automation finish-resting-ppo-comparison is ACTIVE every30minutes, quiet during normal progress. Pause after final comparison. After completion verify twice, copy/inspect final_export_01, report standing separately from near-seat retention and paired whole-body torque only on common full-duration successes. Preserve all prior studies and failed probes.

## Interim check at72 evaluated updates — October15,2026 PDT

Neither seed has reached standing starts:0 standing-training transitions. Both passed stages5 and4 with18/18 strict success, then remained atstage3 (current approach10s crouch). Latest full20s stage3 evaluation:57111=1/18 stable;57117=9/18 stable; both0/18 strict and2/18 forbidden-contact failures, on42cm/−5degree surface repeatedtwice. First forbidden contact at0.18s after trial start; friction-margin violations occur in everycase and slipping in16/11 cases respectively. Measured plane remained available before failures. This is not reliable performance and not a matched standing effort comparison. Final comparisons against currentreference/currentPPOs remain pending. Training and finalizer continue within frozen budget; no extra training or changes. Evidence:near_seat_curriculum_01/interim_72_updates.json.
