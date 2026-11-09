"""Offline evaluation only: full-horizon outcomes and protected approach parity."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
r=Path('/data/humanoid-sitting-perception/traction_transfer_01')
base=r.parent/'crouch_rest_bridge_01/safe_env_startup_reference_01'
b=np.load(base/'trace.npz');bs=np.load(base/'startup.npz')
summary={}
for folder in sorted(r.iterdir()):
    if not (folder/'results.json').exists():continue
    rows=json.loads((folder/'results.json').read_text())['episodes']
    t=np.load(folder/'trace.npz');s=np.load(folder/'startup.npz');keep=t['t']<=10.
    assert np.array_equal(t['t'],b['t']) and len(rows)==18
    prefix={k:dict(exact=bool(np.array_equal(t[k][keep],b[k][keep])),max_abs=float(np.max(abs(t[k][keep]-b[k][keep])))) for k in ['q','command','body','torque']}
    starts={k:bool(np.array_equal(s[k],bs[k])) for k in s.files if k in bs.files}
    good=[e for e in rows if e['strict_transfer']]
    summary[folder.name]=dict(stable=sum(e['success'] for e in rows),strict=len(good),physical=sum(any(e['failures'].values()) for e in rows),trials=len(rows),full_20s=all(e['steps']==1000 and e['duration_s']==20 for e in rows),startups_exact=starts,first10s=prefix,failures=[{k:e[k] for k in ['case','geometry_index','success','failures','slip_ticks','traction_bad_ticks','illegal_ticks']} for e in rows if not e['strict_transfer']],clean_final5s_torque_squared_Nm2=[e['metrics_last5s']['total_squared_torque_Nm2'] for e in good])
out=Path(a.output);assert not out.exists();out.write_text(json.dumps(summary,indent=2))
print(json.dumps({k:{x:v[x] for x in ['stable','strict','physical','full_20s','first10s']} for k,v in summary.items()},indent=2))
