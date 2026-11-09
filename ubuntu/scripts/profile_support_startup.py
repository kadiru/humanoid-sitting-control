"""CPU-only startup-estimator profile on recorded sensor inputs; no simulator."""
import argparse
import cProfile
import json
from pathlib import Path
import pstats
import time
import numpy as np
from seat_depth_estimator import estimate_body_frame

p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
source=Path('/data/humanoid-sitting-perception/support_rl_camera_probe_02/startup_capture.npz')
with np.load(source) as capture:
    data={key:capture[key] for key in ('depth','K','mount','quat','gravity')}
def run():
    rows=[]
    for i in range(len(data['depth'])):
        start=time.perf_counter()
        estimate_body_frame(data['depth'][i],data['K'][i],data['mount'],data['quat'],data['gravity'][i])
        rows.append(time.perf_counter()-start)
    return rows
# Warm imports/caches before recording unprofiled timing.
run();times=run()
prof=cProfile.Profile();prof.runcall(run)
with (out/'profile.txt').open('w') as f:pstats.Stats(prof,stream=f).sort_stats('cumulative').print_stats(35)
report=dict(source=str(source),images=len(times),resolution=list(data['depth'].shape[1:]),
    median_image_s=float(np.median(times)),total_batch_s=float(sum(times)),
    maximum_image_s=float(max(times)),per_image_s=times,
    scope='CPU startup geometry fitting only; excludes simulator rendering, physics, live-depth processing and PPO. Measured while the fixed PPO run was active.')
(out/'timing.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
