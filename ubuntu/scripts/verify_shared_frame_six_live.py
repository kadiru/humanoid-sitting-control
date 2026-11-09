"""Verify revised live observer against replay; sensor equality permits reuse."""
import argparse
import json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser()
p.add_argument('--replay',default='shared_frame_rear_down_six_clean_02')
p.add_argument('--replay-capture',default='shared_frame_rear_down_six_capture_01')
a=p.parse_args()
root=Path(__file__).resolve().parents[1]
live=root/'runs/shared_frame_rear_down_six_live_02'
capture=root/'runs'/a.replay_capture;run=root/'runs'/a.replay
keys=['depth','K','q','gravity','delta_rotation','stamp','T_P_C']
same={}
with np.load(live/'sensors.npz') as x,np.load(capture/'sensors.npz') as y:
    for k in keys: same[k]=bool(np.array_equal(x[k],y[k],equal_nan=True))
if not all(same.values()):
    print('INPUTS_DIFFER_REQUIRE_OWN_REPLAY',same,flush=True)
    raise SystemExit(3)
online=json.loads((live/'online_estimates.json').read_text())
offline=json.loads((run/'estimates.json').read_text())
assert len(online)==len(offline)==966
assert all(x['case']==y['case'] and x['stamp']==y['stamp'] for x,y in zip(online,offline))
diff={k:max(float(np.max(np.abs(np.asarray(x[k])-y[k]))) for x,y in zip(online,offline))
      for k in ['T_O_P','T_S_P','direct_clearance_m']}
flags=all(x['valid']==y['valid'] and x['quality']==y['quality'] for x,y in zip(online,offline))
live_hash=json.loads((live/'manifest.json').read_text())['source_sha256']['shared_frame.py']
replay_hash=json.loads((run/'manifest.json').read_text())['source_sha256']['shared_frame.py']
assert live_hash==replay_hash
assert max(diff.values())<1e-7 and flags
report=dict(samples=len(online),live_capture=live.name,replay=a.replay,replay_capture=a.replay_capture,
            runtime_sensor_arrays_exactly_identical=same,maximum_absolute_difference=diff,
            all_nested_quality_values_identical=flags,observer_sha256=live_hash,
            scope='Fresh six-scene online physics capture, same revised estimator. Existing replay reused only after exact equality of every consumed sensor array and calibration was verified; no truth input is used.')
(root/'runs/shared_frame_six_live_contract_02.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2),flush=True)
