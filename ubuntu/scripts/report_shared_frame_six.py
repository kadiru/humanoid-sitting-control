"""Write the completed six-surface experiment and preserved failure analysis."""
import json
from pathlib import Path

root=Path(__file__).resolve().parents[1]
def read(name):return json.loads((root/name).read_text())
after=read('runs/shared_frame_six_validation_02/validation.json')
before=read('runs/shared_frame_six_validation_01/validation.json')
regression=read('runs/shared_frame_component_regression_01/audit.json')
live=read('runs/shared_frame_six_live_contract_02.json')
integrity=read('runs/shared_frame_six_integrity_03.json')
diagnostic=read('runs/shared_frame_component_diagnostic_01/diagnostic.json')
evaluation=read('runs/shared_frame_rear_down_six_clean_02/evaluation.json')
qs=('clearance','forward','lateral','yaw')
lines=['# Six-surface depth perception: validation and contour association fix', '',
    'The rear vertical observer camera substantially improves measured pelvis-to-seat alignment on benches and boxes. Testing all six surfaces exposed a shape-association bug on the rounded box; it is fixed and the failing recordings are retained. Chair and couch boundaries remain incomplete, so this is not a perception-driven controller release.', '',
    'The original controller camera, controller weights, physics, and trajectories were held fixed. No PPO or soft-landing tuning was performed. Runtime uses only depth, static calibration, encoder kinematics, gyro increments, and gravity. Simulator truth supplies offline error labels.', '',
    '## Results after the fix', '',
    '| Condition | All alignment gates | Clearance available | Clearance P95 (mm) | Forward P95 (mm) | Lateral P95 (mm) | Yaw P95 (deg) | Threshold violations |',
    '|---|---:|---:|---:|---:|---:|---:|---:|']
for mode,r in after['runs'].items():
    q=r['quantities'];err=[q[k]['accepted_error']['p95'] for k in qs]
    lines.append(f"| {mode} | {r['alignment_valid']}/{r['samples']} | {q['clearance']['available']}/{r['samples']} | {10*err[0]:.2f} | {10*err[1]:.2f} | {10*err[2]:.2f} | {err[3]:.3f} | {len(r['violations'])} |")
lines += ['', 'P95 errors apply only where each quantity is accepted. Every rejected estimate and its error remains in the run files. Limits are 2 cm clearance, 3 cm forward/lateral, and 5 degrees axis yaw; these are provisional simulation criteria, not calibrated probabilities. Noisy means 5 mm Gaussian axial noise, 10% independent missing pixels, and 60 ms receipt latency. Occlusion removes all depth from 6–6.5 s and 8–10 s. Higher availability in the noisy run reflects sensitivity of the heuristic edge/observability gates; it is not evidence that noise improves sensing.', '',
    '| Surface | Original camera, clean | New camera, clean | New camera, noisy | New camera, occlusion |', '|---|---:|---:|---:|---:|']
for i,c in enumerate(after['runs']['clean']['cases']):
    g=c['geometry'];v=[after['runs'][m]['cases'][i]['alignment_valid'] for m in ('original','clean','noisy','occlusion')]
    lines.append(f"| {g['family']} / {g['shape']} | "+' | '.join(f'{n}/161' for n in v)+' |')
lines += ['', 'Clearance is distance from the pelvis collision geometry to the currently observed seat plane. The finite observed support cells are retained separately; alignment availability is not a guarantee that a complete sitting trajectory fits.',
    f"Clean all-frame P95 body-position error: {evaluation['aggregate']['body_position_error_cm']['p95']:.3f} cm; pelvis-position error: {evaluation['aggregate']['pelvis_position_error_cm']['p95']:.3f} cm. These use only the initial XY/yaw gauge alignment, then compare the reconstructed body to truth offline.", '',
    '## Failure found and repaired', '',
    'Contours were associated by plane height before the shape fit. Six unrelated edges at the same height as the seat contaminated the rounded support contour. The robust ellipse optimizer found a plausible model, but its all-edge residual rejected it. The fallback minimum-area rectangle then reported a polygon-facet axis as reliable.',
    f"At sensor t=2.0 s, the reported axis changed from {diagnostic['before_axis_deg']:.2f}° to {diagnostic['after_axis_deg']:.2f}° after removing the unrelated edges; the accepted ellipse residual is {diagnostic['after_ellipse_rms_mm']:.2f} mm. The fix associates edges with the selected connected support component using a 3 cm proximity gate before shape fitting. It uses no furniture labels or truth.", '',
    '| Condition | Before: violating quantity/frame pairs | After |', '|---|---:|---:|']
for mode in ('clean','noisy','occlusion'):
    lines.append(f"| {mode} | {len(before['runs'][mode]['violations'])} | {len(after['runs'][mode]['violations'])} |")
lines += ['', 'A frame can contribute both yaw and lateral violations in this table. The pre-fix runs remain in `shared_frame_rear_down_six_*_01`. The corrected runs use suffix `_02`. The revised observer SHA-256 is `'+live['observer_sha256']+'`.', '',
    '## Validation and preservation', '',
    '- 22 CPU tests passed, including a new disconnected-contour regression. Existing tests still reject image-crop boundaries, ambiguous straight-edge reacquisition, and blind common-foot slips.',
    f"- All {after['depth_blackout']['samples']} fully missing-depth frames rejected both clearance and alignment. Reacquisition after depth returns at 10 s occurred at: "+', '.join(f"case {r['case']}: {r['first_relocalized_stamp']:.1f} s" for r in after['reacquisition'])+'.',
    '- Original and added-camera trajectories match exactly in all six physics scenes (root position and orientation). The controller inputs and frozen weights were preserved.',
    f"- A fresh six-scene physics capture with the revised observer matched all {live['samples']} replayed frames. Every consumed sensor/calibration array was exactly equal before reusing the replay; maximum pose/clearance difference was {max(live['maximum_absolute_difference'].values()):.3g}, with identical nested quality values.",
    f"- Preservation check: {integrity['original_source_files']} original source files, archived baselines, checkpoints, and Isaac Lab remained unchanged. Free Ubuntu disk at verification: {integrity['free_gib']:.2f} GiB.", '',
    '| Prior regression | Clearance / forward / lateral / yaw accepted | Above-limit accepted samples |', '|---|---:|---:|']
for name,r in regression.items():
    count=' / '.join(str(r['quantities'][q]['valid']) for q in qs)
    violations=sum(r['quantities'][q]['errors_above_threshold_when_valid'] for q in qs)
    lines.append(f"| {name} | {count} | {violations} |")
lines += ['', 'The two analytical motion banks remain regressions, including the second bank previously used to diagnose map cropping. They are not untouched holdouts or physical walking/slip controllers.', '',
    '## Remaining work', '',
    'The chair has no confirmed rear seat extent; the couch also lacks confirmed side extents and a seat axis. Their motion registration can be fully constrained while seat geometry remains incomplete. The current boundary extractor primarily keeps depth-discontinuity contours on the seat plane. Boundaries where the seat meets a backrest or armrest need explicit obstacle/plane-junction evidence, or additional views; do not relax the existing gates to mask this gap.',
    'Next implement and validate these supported seat boundaries, then freeze the observer again and test fresh heights, dimensions, offsets, and yaws plus extrinsic-calibration errors. Do this before enabling perception-driven alignment. No perception-driven descent is enabled. The virtual mount has not been physically verified; the added-camera RGB channel remains blank and all visuals use its valid depth channel.', '',
    '## Artifacts', '',
    '- `runs/shared_frame_six_validation_02`: report, per-surface measurements, and depth comparison. Final 24-second common-frame video with revised caption spacing: `runs/shared_frame_six_visuals_03/six_surface_shared_frame.mp4`.',
    '- `runs/shared_frame_component_diagnostic_01`: before/after contour association figure.',
    '- `runs/shared_frame_component_regression_01`: five prior regression audits.',
    '- `runs/shared_frame_six_live_contract_02.json` and `runs/shared_frame_six_integrity_03.json`: live/replay and preservation evidence.',
    '- `scripts/validate_shared_frame_six.sh` preserves the initial experiment; `scripts/validate_shared_frame_component_fix.sh` reproduces the corrected benchmark in new output directories.', '']
(root/'docs/shared-frame-six-surface-results.md').write_text('\n'.join(lines))
print('REPORT_COMPLETE docs/shared-frame-six-surface-results.md')
