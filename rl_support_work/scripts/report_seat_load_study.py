"""Package only independently verified seat-load results and offline replay."""
import argparse,ast,csv,hashlib,html,json,shutil
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_seat_load_study import DATA,OLD,read,sha

def replay(paths,out):
    chains=[['pelvis','torso_link','head_link']]
    for side in ['left','right']:
        chains.append(['pelvis']+[side+s for s in ['_hip_pitch_link','_hip_roll_link','_hip_yaw_link','_knee_link','_ankle_pitch_link','_ankle_roll_link']])
        chains.append(['torso_link']+[side+s for s in ['_shoulder_pitch_link','_shoulder_roll_link','_shoulder_yaw_link','_elbow_pitch_link','_elbow_roll_link','_palm_link']])
    payload={};labels=[]
    for key,path in paths.items():
        r=read(path/'results.json');t=np.load(path/'trace.npz');gs=read(path/'geometry.json');rows=sorted(r['episodes'],key=lambda x:x['case']);body=list(dict.fromkeys(n for chain in chains for n in chain if n in r['body_names']));ids=[r['body_names'].index(n) for n in body];links=[[body.index(n) for n in chain if n in body] for chain in chains]
        xy=(t['body'][:,:,ids,:3]-np.array(r['origins'])[None,:,None,:])[:,:,:,[0,2]];edges=[];labels=[]
        for row in rows:
            g=gs[row['geometry_index']];c=np.array(g['top_center']);u=Rotation.from_quat(g['rotation_xyzw']).as_matrix()[:,0];edge=c+np.array([-.5,.5])[:,None]*g['depth']*u;edges.append(edge[:,[0,2]].tolist());labels.append(f"{100*g['height']:.1f}cm · {g['pitch_deg']:+.1f}° · case{row['case']}")
        payload[key]=dict(chains=links,xy=xy.round(4).tolist(),edges=edges,episodes=rows,rms=np.sqrt(t['effort']/37).round(3).tolist(),support=t['vertical_support'].round(4).tolist(),**{k:t[k].tolist() for k in ['good','alive','traction','legal']})
    tree=ast.parse(Path(__file__).with_name('render_two_surface_replay.py').read_text());template=next(n.value.value for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='template' for t in n.targets))
    start=template.index('for(let i=0;i<14;i++)');end=template.index('\nfunction name(key)',start)
    template=template[:start]+f"const labels={json.dumps(labels)};for(let i=0;i<labels.length;i++){{let o=document.createElement('option');o.value=i;o.textContent=labels[i];condition.append(o)}}"+template[end:]
    names={'eval_56811':'New PPO 56811','eval_56817':'New PPO 56817','eval_new_reference':'New reference without PPO','eval_current_56217':'Current policy56217'}
    start=template.index('function name(key)');end=template.index('\nfor(const key',start);template=template[:start]+f'const names={json.dumps(names)};function name(key){{return names[key]||key}}'+template[end:]
    template=template.replace("['fixed_all','Fixed posture']","['fixed_all','Current reference']").replace("['reference_all','Reference only']","['reference_all','Current policy 56211']")
    template=template.replace('Sitting and leaning: inspect the movement','Seat weight transfer: standing-start holdout replay').replace('Compare identical starting conditions. Stage 0 starts standing; stage 6 starts already supported.','All trials start standing and run 20 seconds. These reserved surfaces were evaluated after both final checkpoints were frozen.')
    template=template.replace('Start <select','Surface <select').replace('PPO checkpoint <select','Comparison <select').replace('Contact forces are summarized by the normal support load, so an oblique support load is not the vertical fraction of body weight.','The displayed load is the measured vertical seat force divided by bodyweight.').replace('“No stable support” can include a robot that stays standing.','“No stable support” can include a seated robot with insufficient friction margin.').replace('href="report.html"','href="findings.html"')
    out.write_text(template.replace('__DATA__',json.dumps(payload,separators=(',',':'),allow_nan=False)))

def main():
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--verification',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    v=read(a.verification/'verification.json');assert v['passed'];a.output.mkdir(exist_ok=False);out=a.output
    shutil.copytree(a.verification,out/'verification');shutil.copytree(DATA/'rest_verified_02',out/'rest_validation');shutil.copy2(DATA/'rest_geometry.png',out/'rest_geometry.png')
    protocol=read(a.study/'protocol.json');shutil.copy2(a.study/'protocol.json',out/'protocol.json');shutil.copy2(a.study/'holdout_barrier.json',out/'holdout_barrier.json');shutil.copytree(a.study/'frozen',out/'frozen')
    allpaths={f'development_{k}':Path(p) for k,p in protocol['development_baselines'].items()}
    for seed in protocol['seeds']:
        for bank in ['development','holdout']:allpaths[f'{bank}_{seed}']=a.study/f'{bank}_{seed}'
    for label in ['current_reference','current_56211','current_56217','new_reference']:allpaths['holdout_'+label]=a.study/('holdout_'+label)
    evidence=out/'evidence';evidence.mkdir()
    for label,path in allpaths.items():
        target=evidence/label;target.mkdir()
        for name in ['results.json','geometry.json','manifest.json','startup.npz','startup_validation.json','trace.npz','first_failures.json','sensor_audit.json','initial.pt','parent_identity.json','action_contract.json','cartesian_paths.npz','cartesian_manifest.json']:
            if (path/name).exists():shutil.copy2(path/name,target/name)
    for seed in protocol['seeds']:
        target=out/f'training_{seed}';target.mkdir()
        for name in ['initial.pt','critic_warmup.pt','final.pt','manifest.json','progress.json','verification.json','results.json','first_failures.json','action_contract.json']:
            shutil.copy2(a.study/f'train_{seed}'/name,target/name)
    for name in ['admission.json','optimizer_regression_02.json','optimizer_recovery_01.json','reporting_recovery_01.json']:
        shutil.copy2(DATA/name,out/name)
    rows=[]
    for label,baselines in v['comparisons'].items():
        for baseline,comp in baselines.items():
            for group,windows in comp['grouped'].items():
                for window,metrics in windows.items():
                    for metric,stats in metrics.items():rows.append(dict(policy_label=label,baseline_label=baseline,group=group,window=window,metric=metric,**stats))
    with (out/'paired_metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    labels=['current_reference','current_56211','current_56217','new_reference','56811','56817'];names=['Current\nreference','Current\nPPO56211','Current\nPPO56217','New\nreference','New\nPPO56811','New\nPPO56817']
    fig,axs=plt.subplots(1,2,figsize=(13,4.8));xx=np.arange(6)
    for i,bank in enumerate(['development','holdout']):
        counts=[v['evaluations'][f'{bank}_{x}']['successes'] for x in labels];failed=[v['evaluations'][f'{bank}_{x}']['physical_failures'] for x in labels];n=18 if bank=='development' else 16
        axs[i].bar(xx,counts,color=['#526f84']*3+['#b08a5d','#267e79','#4b9c72']);axs[i].set_ylim(0,n+3);axs[i].set_xticks(xx,names,fontsize=9);axs[i].set_title(f'{bank.title()}: full 20-second standing trials');axs[i].set_ylabel('Stable successes');axs[i].grid(axis='y',alpha=.2);axs[i].set_axisbelow(True)
        for x,c,b in zip(xx,counts,failed):axs[i].text(x,c+.2,f'{c}/{n}\n{b} physical',ha='center',fontsize=9)
    fig.suptitle('Standing-start stability and physical failures');fig.tight_layout();fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
    summaryrows=[];comparisonrows=[]
    fmt=lambda x:'—' if x is None else f'{x:+.1f}%'
    for label,name in zip(labels,names):
        e=v['evaluations']['holdout_'+label];summaryrows.append(f"<tr><td>{html.escape(name.replace(chr(10),' '))}</td><td>{e['successes']}/16</td><td>{e['physical_failures']}</td><td>{v['strong_rest_goal'].get('holdout_'+label,{}).get('count','—')}</td></tr>")
    for seed in protocol['seeds']:
        for baseline in ['current_reference','current_56211','current_56217','new_reference']:
            comp=v['comparisons'][f'holdout_{seed}'][baseline]['grouped']['all'];whole=comp['metrics_whole_trial']['total_squared_torque_Nm2'];last=comp['metrics_last5s']['total_squared_torque_Nm2'];load=comp['metrics_last5s']['vertical_support_load_bw'];n=whole['common_successes']
            comparisonrows.append(f"<tr><td>{seed}</td><td>{html.escape(baseline.replace('_',' '))}</td><td>{n}</td><td>{fmt(whole['change_percent'])}</td><td>{fmt(last['change_percent'])}</td><td>{'—' if load['policy'] is None else f'{100*load["policy"]:.1f}%'}</td></tr>")
    conclusions=[]
    for seed in protocol['seeds']:
        e=v['evaluations'][f'holdout_{seed}'];n=v['strong_rest_goal'][f'holdout_{seed}']['count'];conclusions.append(f"Seed {seed}: {e['successes']}/16 stable, {e['physical_failures']} physical failures, {n}/16 meet the stronger low-effort/load goal.")
    text=' '.join(conclusions)
    (out/'findings.html').write_text(f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Seat load transfer — verified results</title><style>body{{font:17px/1.55 system-ui;color:#20383d;max-width:1120px;margin:38px auto;padding:0 22px;background:#fbfcfa}}h1{{font-size:34px;line-height:1.2}}h2{{margin-top:34px;font-size:24px}}table{{border-collapse:collapse;width:100%;font-size:15px}}th,td{{text-align:left;padding:10px;border-bottom:1px solid #dce3e0}}img{{width:100%;height:auto}}.note{{color:#52676b}}a{{color:#126f75}}.lead{{font-size:20px;background:#eaf1ed;padding:20px;border-radius:8px}}</style>
<h1>More weight on the seat, less joint effort</h1><p class="lead">The resting target passed 18/18 full 20-second seated tests:73–74% of bodyweight on the seat and91–96Nm² total squared joint torque, with no physical failures. Reaching that target from standing is a separate test.</p><p>{text}</p>
<p><a href="motion_replay.html">Inspect the matched standing-start replay</a> · <a href="paired_metrics.csv">All paired metrics and height/slope groups</a> · <a href="verification/verification.json">Independent verification</a></p>
<h2>Standing stability on reserved surfaces</h2><table><thead><tr><th>Controller</th><th>Stable</th><th>Physical failures</th><th>Low-effort goal</th></tr></thead><tbody>{''.join(summaryrows)}</tbody></table><p class="note">Stable means full 20seconds without physical failure and continuous qualifying support for the final 5 seconds. The stronger goal also requires at least 50% vertical seat load and at most 35% of the matched current-reference final 5s squared torque. “Not stable” can mean insufficient friction margin while still seated.</p><img src="comparison.png" alt="Development and holdout stability with physical failures">
<h2>Effort on common full-duration successes</h2><table><thead><tr><th>New seed</th><th>Baseline</th><th>Common successes</th><th>Whole20s torque²</th><th>Final 5s torque²</th><th>Final 5s seat weight</th></tr></thead><tbody>{''.join(comparisonrows)}</tbody></table><p class="note">Percentages compare summed squared physical torque over all 37 joints. Failed fragments are excluded; a saving on a small successful subset does not offset lost stability. CSV also includes final 10s, absolute torque, RMS Nm averaged over time, leg/upper-body contributions and foot load.</p>
<h2>Validated target, already seated</h2><img src="rest_geometry.png" alt="Offline collision geometry of current posture and upright target"><img src="rest_validation/rest_validation.png" alt="Low-effort seated target across heights and slopes"><p>Upright posture and removal of the crouching leg preload reduce the resting effort. These tests start seated and therefore are not matched trajectory comparisons with standing controllers.</p>
<h2>What was controlled</h2><p>Two fresh seeds, 64 updates each: 368,640 physics transitions total, including indexed-reset warmup. First 8 updates train the critic only. All episode starts are standing. A new active-only PPO update excludes warmup samples and prevents disabled-actor Adam drift. Runtime actor and critic use sensors, an explicit motion reference and a clock; simulator truth is restricted to resets, reward and evaluation.</p><p>The new zero-residual approach was incomplete before training:4/18development successes and2physical failures. This was preserved as a learning baseline. Both current policies and the current reference remain comparison arms. Both final checkpoints finished before reserved evaluations; no checkpoint selection or policy promotion occurred.</p><p class="note">Two seeds, repeated placements and nearby rigid slabs do not establish arbitrary furniture generalization. This is reference-assisted PPO, not unassisted motion discovery. Joint torque is an effort proxy, not electrical energy or a global minimum. Prior studies remain unchanged.</p></html>''')
    replay({'fixed_all':a.study/'holdout_current_reference','reference_all':a.study/'holdout_current_56211','eval_current_56217':a.study/'holdout_current_56217','eval_new_reference':a.study/'holdout_new_reference','eval_56811':a.study/'holdout_56811','eval_56817':a.study/'holdout_56817'},out/'motion_replay.html')
    shutil.copy2(__file__,out/'report_source.py');files=sorted(x for x in out.rglob('*') if x.is_file());(out/'SHA256.json').write_text(json.dumps({str(x.relative_to(out)):sha(x) for x in files},indent=2));print(text)

if __name__=='__main__':main()
