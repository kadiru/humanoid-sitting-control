"""Render verified transfer assessment and replay; no training or checkpoint selection."""
import argparse,ast,hashlib,html,json,shutil
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def read(p):return json.loads(p.read_text())
def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False)
    fig,axes=plt.subplots(2,3,figsize=(13,8),layout='constrained');summary={};tables=[];payload={};labels=[]
    chains=[['pelvis','torso_link','head_link']]
    for side in ['left','right']:
        chains.append(['pelvis']+[side+x for x in ['_hip_pitch_link','_hip_roll_link','_hip_yaw_link','_knee_link','_ankle_pitch_link','_ankle_roll_link']])
        chains.append(['torso_link']+[side+x for x in ['_shoulder_pitch_link','_shoulder_roll_link','_shoulder_yaw_link','_elbow_pitch_link','_elbow_roll_link','_palm_link']])
    for row,surface in enumerate(['sitting','leaning']):
        folder=a.data/surface;v=read(folder/'verified_01/verification.json');assert v['passed'];summary[surface]=v['by_geometry']
        dest=a.output/surface;dest.mkdir();shutil.copy2(folder/'verified_01/verification.json',dest/'verification.json')
        seeds=[55211,55217] if surface=='sitting' else [55311,55317];keys=['fixed','reference']+[f'policy_{s}' for s in seeds]
        rows=[]
        for key in keys:
            d=folder/key;r=read(d/'results.json');gs=read(d/'geometry.json');target=dest/key;target.mkdir()
            for name in ['results.json','geometry.json','manifest.json','startup.npz','startup_validation.json','sensor_audit.json','first_failures.json','trace.npz']:shutil.copy2(d/name,target/name)
            unseen=[e for e in r['episodes'] if not gs[e['geometry_index']]['known_anchor']];known=[e for e in r['episodes'] if gs[e['geometry_index']]['known_anchor']]
            rows.append(f'<tr><td>{html.escape(key)}</td><td>{sum(e["success"] for e in known)}/{len(known)}</td><td>{sum(e["success"] for e in unseen)}/{len(unseen)}</td><td>{sum(any(e["failures"].values()) for e in unseen)}</td></tr>')
            if key!='fixed':
                col=keys.index(key)-1;ax=axes[row,col];z=v['by_geometry'][key];success=np.array([x['success'] for x in z]).reshape(3,3);fails=np.array([x['physical_failures'] for x in z]).reshape(3,3)
                ax.imshow(success,vmin=0,vmax=2,cmap='RdYlGn',origin='lower')
                for i in range(3):
                    for j in range(3):ax.text(j,i,f'{success[i,j]}/2\nfail {fails[i,j]}'+(' *' if i==1 and j==1 else ''),ha='center',va='center',fontsize=10,color='#142b2f')
                ax.set(title=surface.title()+' · '+key.replace('policy_','PPO '),xticks=range(3),xticklabels=[f'{gs[j]["pitch_deg"]:g}°' for j in range(3)],yticks=range(3),yticklabels=[f'{gs[i*3]["height"]*100:.2f}' for i in range(3)],xlabel='Surface slope',ylabel='Top-center height (cm)')
            # One placement per geometry for side-by-side animation; all repetitions remain in results.
            trace=np.load(d/'trace.npz');cases=[next(e for e in r['episodes'] if e['geometry_index']==j) for j in range(len(gs))];ids=[e['case'] for e in cases]
            names=list(dict.fromkeys(n for ch in chains for n in ch if n in r['body_names']));joints=[r['body_names'].index(n) for n in names]
            xyz=(trace['body'][...,:3]-np.array(r['origins'])[None,:,None,:])[:,ids][:,:,joints]
            edges=[]
            for g in gs:
                u=Rotation.from_quat(g['rotation_xyzw']).as_matrix()[:,0];edge=np.array(g['top_center'])[None]+np.array([-.5,.5])[:,None]*g['depth']*u;edges.append(edge[:,[0,2]].round(4).tolist())
            label={'fixed':'fixed_all','reference':'reference_all',f'policy_{seeds[0]}':'eval_run1',f'policy_{seeds[1]}':'eval_run2'}[key]
            chunk=dict(chains=[[names.index(n) for n in ch if n in names] for ch in chains],xy=xyz[:,:,:,[0,2]].round(4),edges=edges,episodes=cases,**{k:trace[k][:,ids] for k in ['good','alive','support','traction','legal']},rms=np.sqrt(trace['effort'][:,ids]/37).round(3))
            payload.setdefault(label,[]).append(chunk)
        labels += [f'{surface} · {g["height"]*100:.2f} cm · {g["pitch_deg"]:g}°'+(' (known)' if g['known_anchor'] else ' (unseen)') for g in gs]
        tables.append(f'<h2>{surface.title()}</h2><table><tr><th>Controller</th><th>Known anchor success</th><th>Unseen success</th><th>Unseen physical failures</th></tr>'+''.join(rows)+'</table>')
    fig.suptitle('Before new training: standing-to-support transfer\n8 new geometries per task, each at two placements; * = known anchor',fontsize=15);fig.savefig(a.output/'transfer_grid.png',dpi=160);plt.close(fig)
    data={}
    for key,chunks in payload.items():
        item=dict(chains=chunks[0]['chains'],edges=sum([c['edges'] for c in chunks],[]),episodes=sum([c['episodes'] for c in chunks],[]))
        for k in ['xy','good','alive','support','traction','legal','rms']:item[k]=np.concatenate([c[k] for c in chunks],axis=1).tolist()
        data[key]=item
    tree=ast.parse(Path(__file__).with_name('render_two_surface_replay.py').read_text());template=next(n.value.value for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='template' for t in n.targets))
    old="for(let i=0;i<14;i++){let o=document.createElement('option');o.value=i;o.textContent=(i<7?'40 cm seat':'45° leaning surface')+' · stage '+i%7+(i%7===0?' (standing)':i%7===6?' (supported)':'');condition.append(o)}"
    assert old in template;template=template.replace(old,"const labels="+json.dumps(labels)+";labels.forEach((name,i)=>{let o=document.createElement('option');o.value=i;o.textContent=name;condition.append(o)});")
    template=template.replace('Compare identical starting conditions. Stage 0 starts standing; stage 6 starts already supported.','Frozen policies before new training. Every trial starts standing; first placement shown. Heights and slopes change; reference motion stays fixed.').replace("['reference_all','Reference only']","['reference_all','Corrected reference']").replace("return key.replace('eval_','Seed ').replace('_last_passing',' · last passed').replace('_last',' · last trained')","return key==='eval_run1'?'Previous PPO seed 1':'Previous PPO seed 2'")
    (a.output/'motion_replay.html').write_text(template.replace('__DATA__',json.dumps(data,separators=(',',':'),allow_nan=False)))
    (a.output/'report.html').write_text('<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Unseen height and slope assessment</title><style>body{font:17px/1.5 system-ui;background:#f5f6f2;color:#193438;max-width:1200px;margin:36px auto;padding:0 24px}img{width:100%}table{border-collapse:collapse;width:100%;background:white}td,th{padding:10px;border-bottom:1px solid #ccd8d2;text-align:left}a{color:#216b50}</style><h1>Unseen heights and slopes: before new training</h1><p>Current policies and the corrected reference use their original motion and start standing. Each task tests eight new surface geometries plus its known anchor, repeated at two simulator placements. No geometry-specific retuning.</p><p><a href="motion_replay.html">Inspect motion, support and physical failures</a></p><img src="transfer_grid.png" alt="Support success and physical failure by new surface height and slope">'+''.join(tables)+'<p>Full 20-second trials; success requires the final 5 seconds continuously meeting the original support/contact criteria, with no physical failure. Fixed standing is a retention check. Two placements are repetitions, not independent surfaces. The leaning height is the inclined slab’s top-center height.</p><p>These development cases become training data for the next experiment. A separate eight-surface holdout per task was reserved before testing and remains unused for training or checkpoint selection. This assessment does not contain new effort-training results.</p><p>Exact controller-harness regression, isolated resets, matched physical/sensor startups, loaded parent checkpoints, command bounds, sources, geometry observations and prior artifact preservation were independently checked. All results and validation are included beside this page.</p>')
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2))
    (a.output/'SHA256.json').write_text(json.dumps({str(f.relative_to(a.output)):hashlib.sha256(f.read_bytes()).hexdigest() for f in a.output.rglob('*') if f.is_file()},indent=2))
    print(a.output/'report.html')
if __name__=='__main__':main()
