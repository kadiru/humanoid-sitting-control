FURNITURE TEST ENVIRONMENT — 2026-10-28

Five CC0 assets, 14 isolated layouts: couch/armchair seat, oblique seat, arm and rear-back approaches; stool and both tables at two orientations. Robot faces +X, furniture is behind it at -X.

scene_preview.jpg: CPU rendering of actual furniture meshes and robot collision hulls at a saved standing pose.
scenes/*/furniture.usda: exact triangle colliders, anchored kinematic rigid bodies, friction .8/.6.
scenes/*/layout.usda: static inspection scene with robot collision-hull preview, not an articulated simulation.
scene_adapter.py: creates the actual articulated G1 environment using the existing task configuration and saved original reset.
scene_smoke.py: one 125-control scene/reset check; refuses to start if the 4090 already has a compute process. It never selects the 4070.

STATIC VALIDATION PASSED: all 14 USD meshes round-trip; finite vertices; floor aligned; behind robot; no initial triangle-AABB overlap with robot collision hulls. Python source syntax checked. No simulator was launched. The simulator adapter is not yet runtime-tested.

SCOPE: couch width was intentionally scaled to 2.2m; armchair uniformly scaled by 1.2. Tables/stool retain publisher scale. Solid rigid approximations, no cushion softness or tipping. Original 72 development cases and 24-case holdout are unchanged.

NEXT: after the existing PPO queue ends and the 4090 is idle, run ONE scene smoke with a new output directory:
CUDA_VISIBLE_DEVICES=GPU-ab1950b0-01dd-0fab-407f-2d17936b27f4 timeout 5400s /home/kadir/humanoid-sitting/IsaacLab/.venv/bin/python /home/kadir/humanoid-sitting/experiments/furniture_test_20261028_v1/scene_smoke.py --case 6MoOyPtetL_seat --output /home/kadir/humanoid-sitting/outputs/furniture_scene_smoke_01 --headless

No smoke or policy evaluation is queued automatically. Validate sensor filters and multiple contact surfaces before scoring policy success. Reusing the old single-slab plane scorer would mislabel couch arm/back contacts; target_region.json is a proposal, never actor input or a validated pass label. Success thresholds must remain unchanged.
