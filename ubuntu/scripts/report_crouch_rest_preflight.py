"""Export validated reference gains, clearly separate from pending main PPO."""
import argparse,json,shutil,hashlib,html
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from verify_seat_load_study import evaluation,starts,compare
from verify_progressive_seat_study import quality
from render_crouch_rest_replay import export

p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False)
protocol=json.loads((a.study/'protocol.json').read_text());paths={k:Path(v) for k,v in protocol['development_baselines'].items()};paths['bridge_reference']=Path(protocol['bridge_reference']);rows={};summaries={};geometry={};matched={}
for name,path in paths.items():
    s,r,g=evaluation(path,18);quality(s,r);summaries[name]=s;rows[name]=r;geometry[name]=g;matched[name]=starts(paths['current_reference'],path);assert matched[name]['exact']
    dest=a.output/'evidence'/name;dest.mkdir(parents=True)
    for fn in ['results.json','first_failures.json','manifest.json','startup.npz','bridge_startup.npz','sensor_audit.json','bridge_contract.json','bridge_sources.json']:
        if (path/fn).exists():shutil.copy2(path/fn,dest/fn)
clean=[e for e in rows['bridge_reference'] if e['strict_transfer']];assert len(clean)==12
paired={k:compare(clean,r,geometry['bridge_reference']) for k,r in rows.items() if k!='bridge_reference'}
caseids=[e['case'] for e in clean];labels=['Current reference','Current PPO 56211','Current PPO 56217','Repaired reference'];keys=list(paths)
fig,axes=plt.subplots(1,3,figsize=(15,5));x=np.arange(4)
axes[0].bar(x-.18,[summaries[k]['successes'] for k in keys],.36,label='Stable final hold');axes[0].bar(x+.18,[summaries[k]['physical_failures'] for k in keys],.36,color='#bd433d',label='Physical failures');axes[0].set_ylim(0,19);axes[0].set_ylabel('Trials out of18');axes[0].set_title('All nine development surfaces');axes[0].legend(fontsize=8)
for j,(metric,title,ylabel) in enumerate([('total_squared_torque_Nm2','Resting effort, matched12 successes','Sum of squared torque (Nm²)'),('vertical_support_load_bw','Seat weight share, matched12 successes','Fraction of body weight')],1):
    means=[np.mean([r['metrics_last5s'][metric] for r in rows[k] if r['case'] in caseids]) for k in keys];axes[j].bar(x,means,color=['#8493a2','#7593b1','#51748f','#23856b']);axes[j].set_title(title);axes[j].set_ylabel(ylabel)
for ax in axes:ax.set_xticks(x,labels,rotation=30,ha='right');ax.grid(axis='y',alpha=.15)
fig.suptitle('Reference repair: twelve clean20-second trials; main PPO results pending',fontsize=14);fig.tight_layout();fig.savefig(a.output/'comparison.png',dpi=150);plt.close(fig)
table='<tr><th>Surface</th><th>Stable / clean</th><th>Physical failures</th><th>Training domain</th></tr>'
for gi,g in enumerate(geometry['bridge_reference']):
    rr=[r for r in rows['bridge_reference'] if r['geometry_index']==gi];table+=f"<tr><td>{g['height']*100:.0f}cm / {g['pitch_deg']:+.0f}°</td><td>{sum(r['success'] for r in rr)}/2 stable; {sum(r['strict_transfer'] for r in rr)}/2 clean</td><td>{sum(any(r['failures'].values()) for r in rr)}</td><td>{'Admitted' if gi in protocol['eligible_geometry_indices'] else 'Excluded stress test'}</td></tr>"
change=paired['current_reference']['grouped']['all'];whole=change['metrics_whole_trial']['total_squared_torque_Nm2']['change_percent'];late=change['metrics_last5s']['total_squared_torque_Nm2']['change_percent']
doc=f'''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Validated crouch-to-rest bridge</title><style>body{{font:17px/1.55 system-ui;max-width:1200px;margin:35px auto;padding:0 24px;color:#19363b}}h1{{line-height:1.2}}img{{width:100%}}table{{border-collapse:collapse;width:100%}}td,th{{padding:9px;border-bottom:1px solid #cad6d8;text-align:left}}.note{{padding:18px;background:#fff4d6;border-left:5px solid #c79936}}a{{color:#176f77}}</style><h1>Validated crouch-to-rest bridge</h1><p class="note">Six of nine development surfaces now pass both placements with no slipping, grip-margin violation or unwanted contact. Three conditions remain unresolved. Two main PPO runs are in progress; the results below are controller-repair gains, not learned PPO gains.</p><p>On the12 clean matched trials, the repaired reference reduces total squared joint torque by <b>{-whole:.1f}% over the whole20-second motion</b> and <b>{-late:.1f}% during the final5 seconds</b> versus the current reference. All37 joints are included. These are torque proxies, not measured energy.</p><p>The original first10 seconds match exactly at all saved samples. The new transfer uses predicted motor preload as the seat takes weight, keeps the feet flat using encoders and IMU, and shifts balance correction to the hips once supported. The previous crouch reset is avoided: every training episode starts standing.</p><img src="comparison.png" alt="Stability and paired resting effort"><table>{table}</table><p>Admitted seat-load fractions range approximately49–73%, averaging63%. The minimum is not a majority-weight claim. Excluded cases remain in the final comparison. The six-update optimizer smoke kept12/12 admitted trials clean and introduced no new physical-failure condition.</p><p>Main pilot: seeds58111/58117,64 scheduled updates each,368640 total physics transitions including teacher-only and reset warmup samples. Learned control starts at15 seconds and can move to13,11 and10 seconds only after passing a separate earlier-control evaluation. The approach remains reference-controlled. Eight withheld surface geometries are reserved for final evaluation after both checkpoints freeze.</p><p><a href="motion_replay.html">Inspect the matched motion replay</a> · <a href="results.json">Validation and paired metrics</a> · <a href="protocol.json">Frozen protocol</a></p><p>Repeated placements on known rigid surfaces are limited evidence. No policy is promoted, and the three failed development conditions are not solved.</p>'''
(a.output/'findings.html').write_text(doc);(a.output/'results.json').write_text(json.dumps(dict(summaries=summaries,exact_startups=matched,reference_repair_comparisons=paired,clean_cases=caseids),indent=2));shutil.copy2(a.study/'protocol.json',a.output/'protocol.json')
for fn in ['safe_ppo_admission_01.json','safe_env_regression_verification_01.json','prior_verification_01.json','approach_parity_01.json','holdout_reservation_01.json']:
    shutil.copy2(a.study.parent/fn,a.output/fn)
shutil.copy2(a.study.parent/'safe_env_reset_memory_02/reset_probe.json',a.output/'reset_memory_validation.json');shutil.copy2(__file__,a.output/'report_source.py');export(a.study,a.output/'motion_replay.html',True)
(a.output/'SHA256.json').write_text(json.dumps({str(p.relative_to(a.output)):hashlib.sha256(p.read_bytes()).hexdigest() for p in a.output.rglob('*') if p.is_file()},indent=2));print(a.output)
