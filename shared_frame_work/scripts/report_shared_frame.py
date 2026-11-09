"""Assemble measured results without modifying historical reports."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
runs = {'Clean': 'shared_frame_clean_07', '5 mm noise + 10% dropout': 'shared_frame_noisy_06',
        '0.5 s + 2 s depth losses': 'shared_frame_occlusion_06'}
evaluations = {name: json.loads((root/'runs'/run/'evaluation.json').read_text()) for name, run in runs.items()}
integrity = json.loads((root/'runs/shared_frame_integrity.json').read_text())
contract = json.loads((root/'runs/shared_frame_contract_validation.json').read_text())
lines = ['# Shared 3D body–seat perception: measured results', '',
    'The new observer reconstructs a local floor/seat map from onboard depth and places the G1 skeleton, pelvis, and collision proxies in that same frame using encoders, gyro, gravity, and depth registration. It updates the seat-to-pelvis transform, signed clearance, offsets, orientation, support visibility, and uncertainty. Simulator poses are kept in a separate truth recording and used only for offline error measurement. No controller tuning or training was performed.', '',
    '## Validation scope', '',
    'Six fresh geometry configurations: chair, bench, box, rigid couch, chamfered bench, and a 16-sided elliptical box. Each contains 161 synchronized observations over 16 seconds (966 frame/case samples). These geometries were excluded from earlier controller fitting; this is a bounded development evaluation, not an untouched final perception holdout or hardware test. The pelvis travels roughly 27–31 cm fore/aft and 26–28 cm vertically. Recorded q and simulator body poses independently check the FK overlay. Camera yaw excursions are small; arbitrary turns, walking, moving furniture, and compliant cushions are not validated.', '',
    'The original sensor05/contact02 controller produces the motion and remains unchanged. The new observer has no control authority. A separate one-scene run executes the observer online while the simulator moves, checked against a final-version sensor-only replay (`shared_frame_live_03`); no simulator pose is fed to it.', '',
    '## Errors across every emitted estimate', '',
    'Numbers below are 95th percentiles, including estimates marked invalid or stale. Error uses a single initial XY/yaw gauge alignment; it is not re-aligned every frame. Seat center refers to the observed patch, so hidden full-support extents can produce large initial offset errors. Orientation is an axis modulo 180°, not a proven furniture-facing direction.', '',
    '| Observation condition | Pelvis pose (cm) | Pelvis clearance (cm) | Seat plane height (cm) | Lateral offset (cm) | Yaw axis (deg) | Valid frames |',
    '|---|---:|---:|---:|---:|---:|---:|']
keys = ['pelvis_position_error_cm', 'clearance_error_cm', 'seat_plane_height_error_cm', 'lateral_offset_error_cm', 'yaw_axis_error_deg']
for label, result in evaluations.items():
    values = [f"{result['aggregate'][key]['p95']:.3f}" for key in keys]
    lines.append('| '+label+' | '+' | '.join(values)+f" | {result['valid_frames']}/{result['frames']} |")
lines += ['', 'Validity requires a sufficiently observed footprint, seat age <1 s, and an engineering position-drift budget <3.5 cm. The budget is deliberately labelled heuristic, not calibrated covariance. It must not be interpreted as a certified probability of correctness. Full occlusion retains a static-map prediction; long losses explicitly invalidate the output. A 60 ms receipt delay is evaluated at the measurement timestamp, not passed off as a current-time prediction.', '',
    'Provisional simulation gates: clearance P95 ≤2 cm, plane normal ≤2°, lateral offset ≤3 cm, yaw axis ≤5°. The length tolerances allow 5 mm synthetic depth noise plus approximately 1 cm calibration/proxy allowance. They are not measured hardware tolerances. Per-condition pass/fail:', '']
for label, result in evaluations.items():
    lines.append('- '+label+': '+', '.join(f'{k}={"PASS" if v else "FAIL"}' for k, v in result['gates'].items())+'.')
lines += ['', '## Per-surface clean results', '',
    '| Surface | Clearance P95 (cm) | Lateral P95 (cm) | Yaw P95 (deg) | Valid fraction |', '|---|---:|---:|---:|---:|']
for c in evaluations['Clean']['cases_summary']:
    g = c['geometry']; m = c['metrics']
    lines.append(f"| {g['family']} / {g['shape']} | {m['clearance_error_cm']['p95']:.3f} | {m['lateral_offset_error_cm']['p95']:.3f} | {m['yaw_axis_error_deg']['p95']:.3f} | {100*c['valid_fraction']:.1f}% |")
lines += ['', '## What remains before servo control', '',
    'A camera view containing only a floor/seat plane or a single straight edge cannot observe all translation/heading directions. A two-foot kinematic prior bridges some gaps, conditional on stationary support; common-mode slip remains unobservable. The observer exposes this limitation instead of using true root position. Initial partial silhouettes, conservative body masking, and incomplete landing-patch coverage can also keep estimates invalid. Convex display hulls never authorize unobserved support cells.', '',
    'The six-scene recording demonstrates coherent moving 3D registration and quantifies its limits. It does not establish a general autonomous sitting/alignment solution. Improve view coverage or add independently measured odometry before allowing alignment/descent to depend on uncertain lateral pose. Verify real camera/IMU calibration, timestamp interpolation, foot-slip detection, arbitrary-yaw tests, and dynamic-map handling before hardware use. No additional PPO or touchdown tuning is warranted for these geometric gaps.', '',
    '## Runtime and information-boundary checks', '',
    f"A fixture containing only the sensor file (no truth or geometry manifest) produced exactly identical predictions for {contract['isolated_frames_case_samples']} frame/case samples. All {contract['live_frames']} online/replay frames agreed in pelvis and relative seat pose to floating-point precision; maximum relative-matrix difference was {contract['live_replay_relative_matrix_max_abs_difference']:.2g}. Every one of the {contract['stale_frames_case_samples']} expired-observation samples was invalidated.", '',
    'CPU timings below are per frame/case while other bounded jobs also ran. They do not establish a hard real-time 10 Hz hardware loop. Simulation can slow down to execute the observer; packet outputs remain stamped at measurement time.', '',
    '| Condition | Median observer time (ms) | P95 (ms) |', '|---|---:|---:|']
for label, run in runs.items():
    performance = json.loads((root/'runs'/run/'manifest.json').read_text())['compute_seconds']
    lines.append(f"| {label} | {performance['median']*1000:.1f} | {performance['p95']*1000:.1f} |")
lines += ['',
    '## Tests, preservation, and artifacts', '',
    'Thirteen CPU tests cover SE(3) composition, gravity/optical axes, observable motion registration, planar and noisy-planar degeneracy, contour constraints, curved support axes, rectangle/parallel-edge model rejection, occupancy holes, timestamp rejection/reset, and the privileged-API boundary. The final run folders contain exact source snapshots and hashes. Earlier unsuccessful/incomplete observer iterations and their logs remain preserved.', '',
    f"Integrity: {integrity['original_source_files']} original source/document/calibration files unchanged; archived sensor baseline and selected privileged-student, sensor-student, and observer checkpoint hashes unchanged; Isaac Lab remains clean at `{integrity['isaaclab_revision']}`. New artifacts total {integrity['artifact_total_mib']:.1f} MiB; free disk {integrity['free_gib']:.2f} GiB at final audit. No install or driver change.", '',
    '- [3D video](../runs/shared_frame_clean_07/shared_frame.mp4)',
    '- [Error plots](../runs/shared_frame_clean_07/errors.png)',
    '- [Frame contract and detailed audit](shared-frame-audit.md)',
    '- [Clean metrics](../runs/shared_frame_clean_07/evaluation.json)',
    '- [Noisy metrics](../runs/shared_frame_noisy_06/evaluation.json)',
    '- [Occlusion metrics](../runs/shared_frame_occlusion_06/evaluation.json)',
    '- [Integrity audit](../runs/shared_frame_integrity.json)', '',
    'Run commands are in the audit document and each run manifest. All new work is under `/home/kadir/humanoid-sitting`; simulations are launched directly over SSH in persistent tmux sessions, with no second Codex agent.']
(root/'docs/shared-frame-results.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
