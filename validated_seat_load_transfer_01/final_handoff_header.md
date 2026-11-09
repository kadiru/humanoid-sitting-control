# Latest completed study: seat-load target and standing PPO — October 14, 2026

Read `docs/SEAT_LOAD_TRANSFER_STATUS.md` and `docs/SEAT_LOAD_TRANSFER_PROTOCOL.md` on Ubuntu. Work directly through Mac SSH to `kfuUbuntu`; do not start another coding agent. Root: `/data/humanoid-sitting-perception/seat_load_transfer_01`.

The seated target passed 18/18 full20s tests with no unwanted contacts, 73–74% vertical bodyweight on the seat and 91–96 Nm² total squared torque. The standing-transition pilot did not preserve reliability: reserved current reference/current policies56211/56217 each16/16; new reference0/16; new PPO56811=0/16 and56817=1/16. Each new arm has4 forbidden-contact failures; other misses mostly concern friction margin. Development new PPO0/18 and2/18 versus current18/18. Geometry remained available; foot motion challenges the stationary-foot kinematic estimate.

One successful reserved trial has 30.5% lower whole-motion squared torque and83.5% lower final5s torque versus current reference, but this is not generalizable evidence and does not offset stability loss. Static seated retention is separate from reaching it. No policy was promoted. No training or new automation is pending; existing completion automation remains paused.

Two64-update fresh seeds completed368640 total physics transitions; first8critic-only,56actor-enabled. Both independent verifications pass with identical JSON, exact startups, final checkpoint identity, budgets and52frozen/3341prior hashes. Full result package: remote `final_export_02`; Mac `/Users/kadir/humanoid-sitting-control/validated_seat_load_transfer_01/final_results`. Report/replay: `http://127.0.0.1:8798/final_results/findings.html`. Export01's CSV error and all probes/failed smokes remain preserved.

Proposed next research: establish a valid near-rest-to-rest transfer with low tangential foot forces and housing clearance, then progressively move starts earlier; do not extend the unchanged all-standing pilot or tune on its holdout. The broader goal remains unresolved.

---

Historical completed studies follow; the latest result above takes precedence.

