"""Standalone offline side-view replay; never feeds a controller or training."""
import argparse,ast,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

def read(p):return json.loads(p.read_text())
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--final',action='store_true');a=p.parse_args();assert not a.output.exists()
root=Path('/data/humanoid-sitting-perception/contact_effort_repair_01');old=Path('/data/humanoid-sitting-perception/transfer_effort_01/assessment/sitting');pre=root/'sitting_preflight_01';study=root/'sitting_ppo_01'
if a.final:
    folders={'fixed_all':study/'holdout_old_reference','reference_all':study/'holdout_repaired_reference'}
    names={}
    for seed in [55211,55217]:
        folders[f'eval_old_{seed}']=study/f'holdout_old_parent_{seed}';names[f'eval_old_{seed}']=f'Starting policy {seed}, old reference'
        folders[f'eval_repaired_{seed}']=study/f'holdout_repaired_parent_{seed}';names[f'eval_repaired_{seed}']=f'Starting policy {seed}, repaired reference'
    for seed in [56211,56217]:folders[f'eval_ppo_{seed}']=study/f'holdout_final_{seed}';names[f'eval_ppo_{seed}']=f'Final PPO {seed}'
else:
    folders={'fixed_all':old/'reference','reference_all':pre/'reference'};names={}
    for seed in [55211,55217]:
        folders[f'eval_repaired_{seed}']=pre/f'parent_{seed}';names[f'eval_repaired_{seed}']=f'Starting policy {seed}, repaired reference'
        folders[f'eval_old_{seed}']=old/f'policy_{seed}';names[f'eval_old_{seed}']=f'Starting policy {seed}, old reference'
chains=[['pelvis','torso_link','head_link']]
for side in ['left','right']:
    chains.append(['pelvis']+[side+s for s in ['_hip_pitch_link','_hip_roll_link','_hip_yaw_link','_knee_link','_ankle_pitch_link','_ankle_roll_link']])
    chains.append(['torso_link']+[side+s for s in ['_shoulder_pitch_link','_shoulder_roll_link','_shoulder_yaw_link','_elbow_pitch_link','_elbow_roll_link','_palm_link']])
payload={};labels=None
for key,folder in folders.items():
    r=read(folder/'results.json');t=np.load(folder/'trace.npz');g=read(folder/'geometry.json')
    rows=[];seen=set()
    for row in r['episodes']:
        if row['surface_id'] not in seen:rows.append(row);seen.add(row['surface_id'])
    cases=[x['case'] for x in rows];labels=[f"{g[x['geometry_index']]['top_center'][2]*100:.2f} cm · {g[x['geometry_index']]['pitch_deg']:+.2f}°" for x in rows]
    body=list(dict.fromkeys(n for chain in chains for n in chain if n in r['body_names']));ids=[r['body_names'].index(n) for n in body];links=[[body.index(n) for n in chain if n in body] for chain in chains]
    xyz=(t['body'][:,:,:,:3]-np.array(r['origins'])[None,:,None,:])[:,cases][:,:,ids];xy=xyz[:,:,:,[0,2]]
    edges=[]
    for row in rows:
        geom=g[row['geometry_index']];center=np.array(geom['top_center']);u=Rotation.from_quat(geom['rotation_xyzw']).as_matrix()[:,0];edge=center+np.array([-.5,.5])[:,None]*geom['depth']*u;edges.append(edge[:,[0,2]].round(4).tolist())
    payload[key]=dict(chains=links,xy=xy.round(4).tolist(),edges=edges,episodes=rows,rms=np.sqrt(t['effort'][:,cases]/37).round(3).tolist(),**{k:t[k][:,cases].tolist() for k in ['good','alive','support','traction','legal']})
tree=ast.parse(Path(__file__).with_name('render_two_surface_replay.py').read_text());template=next(n.value.value for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='template' for t in n.targets))
start=template.index('for(let i=0;i<14;i++)');end=template.index('\nfunction name(key)',start)
template=template[:start]+f"const labels={json.dumps(labels)};for(let i=0;i<labels.length;i++){{let o=document.createElement('option');o.value=i;o.textContent=labels[i];condition.append(o)}}"+template[end:]
start=template.index('function name(key)');end=template.index('\nfor(const key',start);template=template[:start]+f'const names={json.dumps(names)};function name(key){{return names[key]||key}}'+template[end:]
template=template.replace("['fixed_all','Fixed posture']","['fixed_all','Old corrected reference']").replace("['reference_all','Reference only']","['reference_all','Repaired reference']")
template=template.replace('Sitting and leaning: inspect the movement','Sitting: inspect the contact repair'+(' and final PPO' if a.final else ''))
template=template.replace('Compare identical starting conditions. Stage 0 starts standing; stage 6 starts already supported.','All trials start standing. The first repeated placement is shown for each surface. '+('These are reserved holdout evaluations after training and selection.' if a.final else 'These are controller repairs and unchanged starting policies, not new PPO results.'))
template=template.replace('Start <select','Surface <select').replace('PPO checkpoint <select','Comparison policy <select').replace('“No stable support” can include a robot that stays standing.','“No stable support” can include failed friction margin while still in contact.').replace('href="report.html"','href="../findings.html"')
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(template.replace('__DATA__',json.dumps(payload,separators=(',',':'),allow_nan=False)));print(a.output)
