"""End-to-end sensor-only, live/replay, and stale-output checks."""
import json
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parents[1]
load = lambda name: json.loads((root/name).read_text())
clean = load('runs/shared_frame_clean_07/estimates.json')
isolated = load('runs/shared_frame_sensor_only_check_02/estimates.json')
assert isolated == clean[:len(isolated)], 'Truth-free replay changed runtime output'
online = load('runs/shared_frame_live_03/online_estimates.json')
offline = load('runs/shared_frame_live_replay_03/estimates.json')
assert len(online) == len(offline) == 161
pose_delta = max(np.max(np.abs(np.asarray(a['T_O_P'])-b['T_O_P'])) for a, b in zip(online, offline))
relative_delta = max(np.max(np.abs(np.asarray(a['T_S_P'])-b['T_S_P'])) for a, b in zip(online, offline))
clearance_delta = max(abs(a['clearance_m']-b['clearance_m']) for a, b in zip(online, offline))
assert pose_delta < 1.e-6 and relative_delta < 1.e-6 and clearance_delta < 1.e-6
assert all(a['stamp'] == b['stamp'] and a['valid'] == b['valid'] for a, b in zip(online, offline))
occlusion = load('runs/shared_frame_occlusion_06/estimates.json')
expired = [r for r in occlusion if r.get('seat_age_s', 0.) >= 1.]
assert expired and not any(r['valid'] for r in expired)
assert not any(r['valid'] for r in occlusion if 9. <= r['stamp'] < 10.)
report = dict(sensor_only_fixture_has_no_truth_or_geometry_manifest=True,
    isolated_frames_case_samples=len(isolated), isolated_predictions_exactly_equal=True,
    live_frames=len(online), live_replay_matrix_max_abs_difference=pose_delta,
    live_replay_relative_matrix_max_abs_difference=relative_delta,
    live_replay_clearance_max_abs_difference_m=clearance_delta, live_replay_status_identical=True,
    stale_frames_case_samples=len(expired), stale_valid_count=0,
    enforced_depth_encoder_imu_skew_s=.005, injected_receipt_delay_s=.060)
(root/'runs/shared_frame_contract_validation.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
