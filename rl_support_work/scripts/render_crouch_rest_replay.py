"""Standalone side-view replay of matched saved evaluations; truth is display only."""
import argparse,ast,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

def read(p):return json.loads(Path(p).read_text())
def export(study,output,preflight=False):
    protocol=read(study/'protocol.json');chains=[['pelvis','torso_link','head_link']]
    for s in ['left','right']:
        chains.append(['pelvis']+[s+n for n in ['_hip_pitch_link','_hip_roll_link','_hip_yaw_link','_knee_link','_ankle_pitch_link','_ankle_roll_link']]);chains.append(['torso_link']+[s+n for n in ['_shoulder_pitch_link','_shoulder_roll_link','_shoulder_yaw_link','_elbow_pitch_link','_elbow_roll_link','_palm_link']])
    banks=['development'] if preflight else ['development','holdout'];payload={};labels=[]
    names={'eval_current_56211':'Current PPO 56211','eval_current_56217':'Current PPO 56217'}
    for s in protocol['seeds']:names[f'eval_{s}']=f'New PPO {s}'
    keys=['fixed_all','reference_all','eval_current_56211','eval_current_56217']+([] if preflight else [f'eval_{s}' for s in protocol['seeds']])
    for key in keys:
        chunks=[]
        for bank in banks:
            if key=='fixed_all':label='current_reference'
            elif key=='reference_all':label='bridge_reference'
            else:label=key.removeprefix('eval_')
            if bank=='development':
                if label=='bridge_reference':folder=Path(protocol['bridge_reference'])
                elif label.isdigit():folder=Path(read(study/f'training_finished_{label}.json')['final_development_evaluation'])
                else:folder=Path(protocol['development_baselines'][label])
            else:folder=study/('holdout_policy_'+label if label.isdigit() else 'holdout_'+label)
            d=read(folder/'results.json');t=np.load(folder/'trace.npz');g=read(folder/'geometry.json');seen=set();rows=[]
            for e in d['episodes']:
                if e['geometry_index'] not in seen:rows.append(e);seen.add(e['geometry_index'])
            cases=[e['case'] for e in rows];body=list(dict.fromkeys(n for ch in chains for n in ch if n in d['body_names']));ids=[d['body_names'].index(n) for n in body];links=[[body.index(n) for n in ch if n in body] for ch in chains]
            xy=(t['body'][:,:,:,:3]-np.array(d['origins'])[None,:,None,:])[:,cases][:,:,ids][:,:,:,[0,2]];edges=[]
            for e in rows:
                geom=g[e['geometry_index']];center=np.array(geom['top_center']);u=Rotation.from_quat(geom['rotation_xyzw']).as_matrix()[:,0];edge=center+np.array([-.5,.5])[:,None]*geom['depth']*u;edges.append(edge[:,[0,2]].round(4).tolist())
                if key==keys[0]:labels.append(f"{bank} · {geom['height']*100:.2f} cm · {geom['pitch_deg']:+.1f}°"+(' · admitted' if bank=='development' and e['geometry_index'] in protocol['eligible_geometry_indices'] else ' · stress test' if bank=='development' else ''))
            chunks.append(dict(chains=links,xy=xy.round(4),edges=edges,episodes=rows,rms=np.sqrt(t['effort'][:,cases]/37).round(3),**{k:t[k][:,cases] for k in ['good','alive','support','traction','legal']}))
        item=dict(chains=chunks[0]['chains'],edges=sum([x['edges'] for x in chunks],[]),episodes=sum([x['episodes'] for x in chunks],[]))
        for k in ['xy','rms','good','alive','support','traction','legal']:item[k]=np.concatenate([x[k] for x in chunks],axis=1).tolist()
        payload[key]=item
    tree=ast.parse(Path(__file__).with_name('render_two_surface_replay.py').read_text());template=next(n.value.value for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='template' for t in n.targets))
    start=template.index('for(let i=0;i<14;i++)');end=template.index('\nfunction name(key)',start)
    template=template[:start]+f"const labels={json.dumps(labels)};for(let i=0;i<labels.length;i++){{let o=document.createElement('option');o.value=i;o.textContent=labels[i];condition.append(o)}}"+template[end:]
    start=template.index('function name(key)');end=template.index('\nfor(const key',start);template=template[:start]+f'const names={json.dumps(names)};function name(key){{return names[key]||key}}'+template[end:]
    template=template.replace("['fixed_all','Fixed posture']","['fixed_all','Current reference']").replace("['reference_all','Reference only']","['reference_all','Repaired bridge reference']")
    template=template.replace('Sitting and leaning: inspect the movement','Crouch-to-rest: inspect the movement').replace('Compare identical starting conditions. Stage 0 starts standing; stage 6 starts already supported.','All trials start standing. First repeated placement per surface shown. '+('Reference repair and existing policies; new PPO results are pending.' if preflight else 'New PPO control timing is frozen by development validation; the first ten seconds remain reference-controlled.'))
    template=template.replace('Start <select','Surface <select').replace('PPO checkpoint <select','Comparison <select').replace('href="report.html"','href="findings.html"').replace('“No stable support” can include a robot that stays standing.','Whole-movement slip/contact checks are additional to this stable-hold score. Three development geometries were excluded from learning and remain stress tests.')
    assert not output.exists();output.write_text(template.replace('__DATA__',json.dumps(payload,separators=(',',':'),allow_nan=False)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--preflight',action='store_true');a=p.parse_args();export(a.study,a.output,a.preflight)
