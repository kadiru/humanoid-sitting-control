"""Summarize diagnostic candidates without changing criteria or outputs."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('output');a=p.parse_args();r=Path(a.output)
results=json.loads((r/'results.json').read_text());g=json.loads((r/'geometry.json').read_text());d=np.load(r/'contact_diagnostics.npz');trace=np.load(r/'trace.npz')
rows=[]
for e in results['episodes']:
 i=e['case'];k=int(e['geometry_index']);active=d['active'][-250:,i];u=np.where(active,d['utilization'][-250:,i],0)
 row=dict(profile=g[k].get('control_profile'),case=i,stage=e['start_stage'],success=e['success'],failures=e['failures'],hold=e['final_hold_s'],late_traction_max=u.max(0).tolist(),vertical_support_last5=float(d['vertical_support'][-250:,i].mean()),foot_last5=float(d['foot'][-250:,i].mean()),peak_BW=float(d['peak'][:,i].max()),effort=e['metrics_whole_trial']['total_squared_torque_Nm2'])
 rows.append(row)
 print(row['profile']['name'] if row['profile'] else '-',row['stage'],'OK' if row['success'] else 'FAIL',','.join(k for k,v in row['failures'].items() if v),'hold',row['hold'],'traction',round(max(row['late_traction_max']),3),'restBW',round(row['vertical_support_last5'],3),'effort',round(row['effort'],1))
(r/'diagnostic_summary.json').write_text(json.dumps(rows,indent=2))
