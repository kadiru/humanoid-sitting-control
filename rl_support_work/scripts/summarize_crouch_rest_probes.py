"""Report physical outcomes of development probes without selecting failed fragments."""
import argparse,json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path)
a=p.parse_args();rows=[]
for f in sorted(a.root.glob('*/results.json')):
 d=json.loads(f.read_text());episodes=d.get('episodes',[])
 if not episodes:continue
 row=dict(probe=f.parent.name,trials=len(episodes),stable=sum(x['success'] for x in episodes),strict=sum(x['strict_transfer'] for x in episodes),physical_failures=sum(any(x['failures'].values()) for x in episodes),slip_cases=sum(x['slip_ticks']>0 for x in episodes),traction_cases=sum(x['traction_bad_ticks']>0 for x in episodes),illegal_cases=sum(x['illegal_ticks']>0 for x in episodes),all_full20s=all(x['steps']==1000 and x['duration_s']==20. for x in episodes))
 good=[x for x in episodes if x['strict_transfer']]
 row['strict_final5s_torque_squared_Nm2']=float(np.mean([x['metrics_last5s']['total_squared_torque_Nm2'] for x in good])) if good else None
 row['strict_final5s_seat_bw']=float(np.mean([x['metrics_last5s']['vertical_support_load_bw'] for x in good])) if good else None
 rows.append(row)
print(json.dumps(rows,indent=2))
if a.output:
 assert not a.output.exists();a.output.write_text(json.dumps(rows,indent=2))
