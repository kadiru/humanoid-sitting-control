"""Package online tracker outputs for the separate truth-based scorer."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--capture',required=True);a=p.parse_args();cap=Path(a.capture);out=cap/'shadow_score';out.mkdir(exist_ok=False)
with np.load(cap/'proprioception.npz') as f:t=f['t'];N=f['q'].shape[1]
sources=['unavailable','current_floor_and_seat','depth_with_kinematic_floor','kinematic_prediction','depth_with_temporal_identity','seat_from_observed_floor']
arrays={k:np.full((len(t),N),np.nan,np.float32) for k in ['clearance','allowance','measurement_age','foot_disagreement']}
arrays['valid']=np.zeros((len(t),N),bool);arrays['source']=np.zeros((len(t),N),np.int16);seen=set()
for line in (cap/'shadow_records.jsonl').read_text().splitlines():
 r=json.loads(line);k=round(r['stamp']/.02)-1;i=r['case'];assert (k,i) not in seen;seen.add((k,i))
 arrays['valid'][k,i]=r['valid'];arrays['source'][k,i]=sources.index(r['source'])
 for field,key in [('clearance','clearance_m'),('allowance','allowance_m'),('measurement_age','measurement_age_s'),('foot_disagreement','foot_disagreement_m')]:arrays[field][k,i]=r.get(key) if r.get(key) is not None else np.nan
assert len(seen)==len(t)*N
np.savez_compressed(out/'tracking.npz',t=t,**arrays)
m=dict(capture=str(cap),sources=sources,tracker_sha256=hashlib.sha256((cap/'source/support_landing_geometry.py').read_bytes()).hexdigest(),fault='none',online=True)
(out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
print(out)
