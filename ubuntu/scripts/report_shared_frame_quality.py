"""Generate the final report from immutable evaluation artifacts."""
import json
from pathlib import Path
import numpy as np

root=Path(__file__).resolve().parents[1]
audit=json.loads((root/'runs/shared_frame_observability_audit_final/audit.json').read_text())
contract=json.loads((root/'runs/shared_frame_quality_contract.json').read_text())
integrity=json.loads((root/'runs/shared_frame_quality_integrity.json').read_text())
labels=dict(shared_frame_quality_clean_04='Recorded physics: clean',
            shared_frame_quality_noisy_04='Recorded physics: 5 mm noise + 10% dropout',
            shared_frame_quality_occlusion_03='Recorded physics: 0.5 s + 2 s depth loss',
            shared_frame_stress_04='Analytical motion/slip fixtures',
            shared_frame_holdout_02='Second analytical bank: 2 mm noise + 180° turn')
text=['# Body–seat perception: observability, clearance, and recovery', '',
    'The observer now separates current-plane clearance from forward/lateral seat-center and yaw measurements. It exposes why each quantity is unavailable, rejects blind motion as an alignment measurement, recovers translation after depth loss, and preserves seat geometry when the map reaches its memory limit. The original controller and checkpoints are unchanged; no PPO, soft-landing tuning, software installation, or control feedback was introduced.', '',
    '## What the audit found', '',
    'In the original clean run, 472 of 966 samples were invalid solely because one scalar drift budget grew when any pose direction was weakly observed. All 966 samples still had a visible seat plane. Another 34 samples lacked an observed footprint (one also exceeded the drift budget). Bench/box/chamfered/ellipse views commonly lose a mixture of lateral translation and yaw; masking removes only about 1–2% of voxel points in these recordings. This percentage does not measure physical occlusion, which removes rays before reconstruction.', '',
    'The scalar budget also did not describe hidden seat extent. Original forward-offset P95 error was around 5–6 cm, including frames marked valid. A cropped visible patch is not a measured full-seat center. Opposing physical contour evidence is now required per dimension; an ellipse center remains an explicitly labelled model hypothesis, and heading remains modulo 180°.', '',
    '## Final measured results', '',
    'Errors below are measured only where the corresponding quantity is flagged usable; availability is shown alongside them. All emitted estimates, including rejected ones, remain in the raw error files. These are empirical simulation/fixture results, not calibrated confidence intervals or hardware guarantees.', '',
    '| Condition | Current clearance samples | Clearance P95 (cm) | Lateral samples | Lateral P95 (cm) | Yaw samples | Yaw P95 (°) | All alignment gates |',
    '|---|---:|---:|---:|---:|---:|---:|---:|']
for name,label in labels.items():
    r=audit[name]; n=r['samples']; q=r['quantities']
    val=lambda key:f"{q[key]['error_when_valid']['p95']:.3f}" if q[key]['error_when_valid'] else '—'
    text.append(f"| {label} | {q['clearance']['valid']}/{n} | {val('clearance')} | {q['lateral']['valid']}/{n} | {val('lateral')} | {q['yaw']['valid']}/{n} | {val('yaw')} | {r['alignment_valid']}/{n} |")
text+=['', 'The stricter alignment count is intentional: direct clearance can be measured while a seat boundary or a motion direction remains unknown. The global `valid` flag now requires every alignment gate. The previous rule is available only as `legacy_valid` for comparison. `quality.quantities` and its reason strings are the primary per-measurement interface; `direct_clearance_m` is absent during depth loss. `clearance_m` remains a separately labelled map-based prediction.', '',
       'Current clearance is computed from the current depth-plane equation transformed into the pelvis frame. This cancels the odometry gauge instead of inheriting drift in an unobserved horizontal direction. It is signed distance to an infinite seating plane, not proof that the pelvis is above finite support or that a collision-free sitting trajectory exists.', '',
       '## Failure-driven fixes', '',
       'A 12 cm common-foot slide during a 2.5 s blackout exposed three falsely accepted lateral estimates above 3 cm in the first diagnostic version. Feet can agree perfectly while both slide. The final quality gate requires an observed motion direction; kinematic agreement only affects a labelled prediction allowance. A bounded ±25 cm XY search after prolonged depth loss uses depth contours and gyro orientation, rejects ambiguous/underconstrained matches, and freezes map fusion until reacquisition.', '']
for name in ('shared_frame_stress_04','shared_frame_holdout_02'):
    rows=json.loads((root/'runs'/name/'estimates.json').read_text())
    errors=json.loads((root/'runs'/name/'errors.json').read_text())
    matches=[(r,e) for r,e in zip(rows,errors) if r['case']==2 and r['stamp']==4.5]
    r,e=matches[0]
    text.append(f"- {labels[name]}: first returning depth frame at 4.5 s; recovery={r['registration'].get('recovery')}; lateral error {e['lateral_offset_error_cm']:.3f} cm.")
text+=['', 'The second bank initially exposed a different failure during a 180° turn: a coordinate-sorted map was truncated from one end at 40,000 points. This cropped the stored seat, causing three accepted lateral/yaw outliers despite accurate pelvis odometry. The final map budget preserves support-height geometry and distributes the remaining samples across the scene; the same turn now retains the seat and reacquires usable lateral/yaw measurements. Because these failures informed fixes, both banks are now regression sets, not untouched holdouts.', '',
       'Frame-level checks count any usable clearance >2 cm error, forward/lateral >3 cm, or yaw >5°. These provisional thresholds are diagnostics, not a control-safety certificate:', '',
       '| Condition | Clearance violations | Forward violations | Lateral violations | Yaw violations |',
       '|---|---:|---:|---:|---:|']
for name,label in labels.items():
    q=audit[name]['quantities']
    text.append('| '+label+' | '+' | '.join(str(q[key]['errors_above_threshold_when_valid']) for key in ('clearance','forward','lateral','yaw'))+' |')
text+=['', '## Validation and limitations', '',
    f"Twenty-one CPU regression tests pass. Sensor-only replay matches {contract['contract']['sensor_only_equal_samples']} frame/case samples exactly with no truth archive or geometry manifest present. A fresh 16-second online bench simulation and replay agree across {contract['contract']['live_replay_frames']} frames; pose/clearance differences are below 1e-6 and quality fields agree within 1e-7. All 150 injected missing-depth samples abstain from clearance and alignment. Runtime source hashes match across the five final evaluations.", '',
    'Physics recordings use the original actuators/controller and the proposed pelvis-mounted rear/down camera. Analytical fixtures use pinhole ray casting, prescribed root motion, and frozen encoders to deliberately violate the stationary-foot assumption; they include new seat dimensions/heights/yaws, 8–16 cm slides, and 100°/180° turns. They do not simulate body occlusion, contact dynamics, camera distortion, or hardware IMU/extrinsic bias. Large-turn evidence is therefore sensor-level, not a validated physical turning controller.', '',
    'Directional allowances remain engineering heuristics. Static furniture and correct calibration are assumed. General moving-object detection, probabilistic uncertainty calibration, hardware timing, and arbitrary long-distance relocalization remain unvalidated. Computation is measured per sensor timestamp; this is not a hard real-time or receipt-time predictor. The fixed map budget can still thin very large support surfaces; omitted support remains unknown.', '',
    'The next experiment should validate the promising rear vertical camera across the full surface bank, noise, occlusion, and fresh geometries, before using it for alignment control. Full-seat center should not be the only candidate target: an independently verified interior footprint on observed support may suffice for later alignment. No descent/controller changes were made.', '',
    f"Preservation audit: {integrity['original_source_files']} original source/document/calibration files unchanged, {integrity['archived_baseline_files']} archived baseline files unchanged, selected checkpoints unchanged, Isaac Lab checkout clean. Ubuntu free space: {integrity['free_gib']:.2f} GiB. Existing recordings, intermediate failures, and checkpoints are retained.", '',
    '## Artifacts and reproduction', '',
    '- [Confidence and recovery video](../runs/shared_frame_quality_visuals/measurement_quality.mp4)',
    '- [Complete audit and per-case timelines](../runs/shared_frame_observability_audit_final/audit.md)',
    '- [Machine-readable metrics](../runs/shared_frame_observability_audit_final/audit.json)',
    '- [Runtime contract](../runs/shared_frame_quality_contract.json)',
    '- [Integrity audit](../runs/shared_frame_quality_integrity.json)', '',
    'Canonical Ubuntu root: `/home/kadir/humanoid-sitting`. Final source: `sitting_task/src/humanoid_sitting/shared_frame.py`. Every replay stores its exact source and hashes. All long jobs ran via SSH in persistent tmux sessions; no second agent was started.', '',
    '```bash', 'cd /home/kadir/humanoid-sitting',
    'OPENBLAS_NUM_THREADS=1 bash scripts/run.sh python ../scripts/test_shared_frame.py',
    '# Use a new output directory; existing recordings are never overwritten.',
    'OPENBLAS_NUM_THREADS=1 bash scripts/run.sh python ../scripts/run_shared_frame.py \\',
    '  --capture ../runs/shared_frame_capture_01 --output ../runs/NEW_RUN --mode clean',
    'OPENBLAS_NUM_THREADS=1 bash scripts/run.sh python ../scripts/evaluate_shared_frame.py \\',
    '  --capture ../runs/shared_frame_capture_01 --tracking ../runs/NEW_RUN',
    '```', '']
camera_path=root/'runs/shared_frame_camera_comparison_02/comparison.json'
if camera_path.exists():
    cam=json.loads(camera_path.read_text()); before=cam['Original camera']; after=cam['Rear vertical camera']
    section=['## Controlled camera-placement experiment', '',
        'A new static calibration input lets the runtime use a camera profile without loading the privileged scene manifest. The tested virtual mount is 22 cm behind and 55 cm above the pelvis origin, looking vertically downward. The original camera remains the sole source of the frozen controller’s startup geometry; only the observer sees the experimental view.', '',
        f"On paired 16-second bench simulations, the complete root-position trajectories were identical (maximum difference {cam['trajectory_comparison']['root_position_difference_max_cm']:.6f} cm). Online and replayed observer poses were also identical. Complete alignment availability increased from {before['alignment_valid']}/161 to {after['alignment_valid']}/161 ({100*after['alignment_valid']/161:.1f}%).", '',
        '| Quantity | Original availability | New availability | New P95 error |', '|---|---:|---:|---:|']
    for key in ('clearance','forward','lateral','yaw'):
        q=after['quantities'][key]
        section.append(f"| {key} | {before['quantities'][key]['available']}/161 | {q['available']}/161 | {q['p95']:.3f} {'degrees' if key=='yaw' else 'cm'} |")
    section+=['', 'No accepted measurement exceeded the same provisional thresholds. The pelvis moved 28.3 cm fore/aft and 25.9 cm vertically. This is one actual simulator scene with robot occlusion, not a six-surface result or hardware-tested mount. Full mechanical feasibility, additional surfaces, and placement robustness remain to be evaluated.', '',
        'The first placement probe (`rear_down_01`) accidentally changed the old controller’s startup geometry input, caused a rapid descent outside the tracking regime, and is rejected as a camera-only comparison. It remains preserved. The corrected paired trial (`rear_down_02`) isolates the observer camera and reproduces the baseline trajectory exactly.', '',
        'The added camera’s RGB channel was blank in this multi-camera capture. Its axial depth is populated and is the runtime input. The comparison shows recorded depth at a common scale, and the video uses depth for that view; the RGB capture issue remains separate from depth tracking.', '',
        '![Camera views and measurement availability](../runs/shared_frame_camera_comparison_02/camera_comparison.png)', '',
        '[Paired comparison metrics](../runs/shared_frame_camera_comparison_02/comparison.json) · [Video including the new view](../runs/shared_frame_quality_visuals_camera_02/measurement_quality.mp4)', '']
    index=text.index('## Validation and limitations')
    text[index:index]=section
(root/'docs/shared-frame-observability-results.md').write_text('\n'.join(text))
print('REPORT_COMPLETE',root/'docs/shared-frame-observability-results.md')
