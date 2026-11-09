"""Bounds, geometry and foot-anchor diagnostics for coordinated pose probes."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--folder',type=Path,required=True);a=p.parse_args();d=a.folder
read=lambda n:json.loads((d/n).read_text())
cfg=read('landing_config.json');r=read('results.json');plans={x['case']:x for x in read('rolling_plan_audit.json')};geom=read('geometry.json')
records=read('task_feedback_audit.json');t=np.load(d/'trace.npz');e=np.load(d/'every_tick.npz');start=np.load(d/'startup.npz')
names=r['joint_names'];ids=[names.index(s+'_'+j+'_joint') for s in ['left','right'] for j in ['hip_pitch','knee','ankle_pitch']];other=[i for i in range(37) if i not in ids]
fingers=[i for i,n in enumerate(names) if any('_'+s+'_' in n for s in ['zero','one','two','three','four','five','six'])];assert len(fingers)==14
assert np.array_equal(t['command'][:,:,fingers],np.broadcast_to(start['command'][None,:,fingers],t['command'][:,:,fingers].shape))
commands=np.array([x['correction'] for x in records]);assert np.isfinite(commands).all() and np.max(abs(commands))<=cfg['task_joint_cap']+1.e-12
assert not np.count_nonzero(commands[:,:,other]);assert np.isfinite(t['command']).all()
for x in records:
    for row in x['states']:
        assert abs(row['angle_correction'])<=cfg['task_pitch_cap']+1.e-12
        assert np.max(abs(row['translation']))<=cfg['translation_cap']+1.e-12
        assert plans[row['case']]['landing_repair_applied']
rows=[]
for row in r['episodes']:
    i=row['case'];g=geom[row['geometry_index']];plan=plans[i];n=np.array(plan['normal']);true_n=np.array(g['normal']);true_d=-true_n@np.array(g['top_center']);offset_error=float(plan['offset']-true_d);normal_error=float(np.rad2deg(np.arccos(np.clip(n@true_n,-1,1))))
    loaded=np.linalg.norm(e['force'][:,i],axis=-1)>.01*e['weight'][:,i,None];slip=(loaded&(e['slip'][:,i]>.02));events=np.flatnonzero(slip.any(-1)|e['failure'][:,i]|(e['wrong'][:,i]>10)|~e['legal'][:,i])
    first=float(e['t'][events[0]]) if len(events) else None;mask=t['t']<=first if first is not None else np.ones(len(t['t']),dtype=bool)
    rows.append(dict(case=i,height=g['height'],slope=g['pitch_deg'],measured_normal_error_deg=normal_error,measured_plane_offset_error_m=offset_error,geometry_availability_before_event=float(np.mean(t['geometry'][mask,i,11]>0)),first_adverse_event_s=first,foot_slip_ticks=slip[:,5:].sum(0).tolist(),stable=row['success'],slip_ticks=row['slip_ticks'],unwanted_contact=bool(row['failures']['forbidden'] or row['illegal_ticks'])))
out=dict(passed=True,fixed_fingers_exact=True,new_correction_other31joints_zero=True,max_new_correction_rad=float(np.max(abs(commands))),angle_cap_rad=cfg['task_pitch_cap'],translation_cap_m=cfg['translation_cap'],finite_actual_commands=True,scope='New correction bounds only. Reference/load/shear/frozen-parent components remain separate. Inherited translation-only correction is computed for diagnostics but explicitly subtracted; do not report it as an applied controller. Geometry/foot-slip labels are offline audit only.',rows=rows)
dest=d/'task_control_audit.json';assert not dest.exists();dest.write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items() if k!='rows'}))
