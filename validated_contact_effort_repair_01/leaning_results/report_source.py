"""Summarize the bounded leaning pilot after independent verification."""
import argparse,csv,json,hashlib,html,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def read(p):return json.loads(p.read_text())
def csvfile(p,rows):
    with p.open('w') as f:
        if rows:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

p=argparse.ArgumentParser();p.add_argument('--view',type=Path,required=True);p.add_argument('--verification',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
verified=read(args.verification);assert verified['passed'];out=args.output;out.mkdir(exist_ok=False)
shutil.copy2(args.verification,out/'verification.json');shutil.copy2(__file__,out/'report_source.py')
summary=[];paired=[];failures=[];headline=[];pre12={}
windows=['metrics_whole_trial','metrics_last10s','metrics_last5s'];metrics=['total_squared_torque_Nm2','total_absolute_torque_Nm','all_joint_rms_Nm','leg_squared_torque_Nm2','upper_squared_torque_Nm2']
for seed,parent in [(56111,55311),(56117,55317)]:
    state=read(args.view/f'seed_{seed}.json')
    for bank in ['development','holdout']:
        label=f'dev_{seed}_block1' if bank=='development' else f'holdout_final_{seed}'
        folder=args.view/label;policy=read(folder/'results.json')['episodes'];g=read(folder/'geometry.json')
        previous=Path('/data/humanoid-sitting-perception/transfer_effort_01/assessment/leaning')
        refs={'corrected_reference':previous/'reference' if bank=='development' else args.view/'holdout_reference','starting_policy':previous/f'policy_{parent}' if bank=='development' else args.view/f'holdout_parent_{parent}'}
        for row in policy:
            geom=g[row['geometry_index']]
            failures.append(dict(seed=seed,bank=bank,case=row['case'],geometry=geom['label'],height_m=geom['top_center'][2],slope_deg=geom['pitch_deg'],success=row['success'],final_hold_s=row['final_hold_s'],**row['failures']))
        for baseline,refdir in refs.items():
            reference=read(refdir/'results.json')['episodes'];common=[(a,b) for a,b in zip(policy,reference) if a['success'] and b['success']]
            for window in windows:
                for metric in metrics:
                    pv=np.mean([a[window][metric] for a,b in common]) if common else None;bv=np.mean([b[window][metric] for a,b in common]) if common else None
                    row=dict(seed=seed,bank=bank,baseline=baseline,window=window,metric=metric,successes=sum(x['success'] for x in policy),count=len(policy),physical_failures=sum(any(x['failures'].values()) for x in policy),common_successes=len(common),policy=pv,reference=bv,change_pct=100*(pv/bv-1) if bv else None)
                    summary.append(row)
                    if bank=='holdout' and metric==metrics[0] and window in [windows[0],windows[2]]:headline.append(row)
                    for a,b in common:
                        geom=g[a['geometry_index']];av=a[window][metric];rv=b[window][metric]
                        paired.append(dict(seed=seed,bank=bank,baseline=baseline,case=a['case'],geometry=geom['label'],height_m=geom['top_center'][2],slope_deg=geom['pitch_deg'],window=window,metric=metric,policy=av,reference=rv,change_pct=100*(av/rv-1) if rv else None))
        a=np.load(folder/'trace.npz');b=np.load(refs['starting_policy']/'trace.npz')
        pre12[label]={k:float(abs(a[k][:120].astype(float)-b[k][:120].astype(float)).max()) for k in ['q','torque','command','reference','geometry','body']}
        assert max(pre12[label].values())==0
csvfile(out/'summary.csv',summary);csvfile(out/'paired_cases.csv',paired);csvfile(out/'case_outcomes.csv',failures)
(out/'approach_identity.json').write_text(json.dumps(pre12,indent=2))

fig,ax=plt.subplots(1,3,figsize=(14,4.8),layout='constrained')
seeds=[56111,56117];x=np.arange(2)
for i,seed in enumerate(seeds):
    rows=[r for r in failures if r['seed']==seed and r['bank']=='holdout'];n=sum(r['success'] for r in rows)
    ax[0].bar(i,n,color='#266b7c');ax[0].text(i,n+.25,f'{n}/{len(rows)}',ha='center')
ax[0].set(title='Stable full-duration holdout trials',xticks=x,xticklabels=seeds,ylim=(0,18),ylabel='Successful trials');ax[0].axhline(16,ls=':',color='gray')
for panel,baseline in zip(ax[1:],['corrected_reference','starting_policy']):
    for j,window in enumerate([windows[0],windows[2]]):
        for i,seed in enumerate(seeds):
            row=next(r for r in headline if r['seed']==seed and r['baseline']==baseline and r['window']==window)
            if row['change_pct'] is not None:panel.bar(i+(j-.5)*.34,row['change_pct'],.34,color=['#266b7c','#d49339'][j],label=['Whole 20 s','Final 5 s'][j] if i==0 else None)
            else:panel.text(i+(j-.5)*.34,0,'N/A',ha='center',va='bottom',rotation=90,fontsize=8)
    panel.axhline(0,color='black',lw=.8);panel.set(title='Effort vs '+baseline.replace('_',' '),xticks=x,xticklabels=seeds,ylabel='Squared torque change (%)');panel.grid(axis='y',alpha=.2)
ax[1].legend(fontsize=8);fig.suptitle('New leaning PPO: final checkpoints, reserved surfaces\nEffort comparisons use common full-duration successes only')
fig.savefig(out/'comparison.png',dpi=170);plt.close(fig)
table=[]
for r in headline:
    table.append('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in [r['seed'],r['baseline'].replace('_',' '),r['window'].replace('metrics_',''),f"{r['successes']}/{r['count']}",r['physical_failures'],r['common_successes'],'N/A' if r['change_pct'] is None else f"{r['change_pct']:+.2f}%"] )+'</tr>')
selection='; '.join(str(s)+': '+('starting-policy fallback' if verified['selection'][str(s)]['parent_fallback'] else 'new checkpoint selected using development only') for s in seeds)
doc='''<!doctype html><html><meta charset="utf-8"><title>Leaning effort repair</title><style>body{font:17px/1.5 system-ui;max-width:1160px;margin:40px auto;padding:0 24px;color:#21333c}h1{font-size:30px}table{border-collapse:collapse;font-size:15px}th,td{border-bottom:1px solid #cbd5da;padding:9px;text-align:left}img{width:100%}p{max-width:1000px}a{color:#16657c}</style><h1>Leaning effort repair: verified comparison</h1>'''
doc+='<p>The final policies are evaluated against both the corrected reference and their own starting policies. Success requires a complete 20-second trial, no physical failure, and five continuous seconds of stable supported contact. Friction margin failures count against success even without a fall or visible slip.</p>'
doc+='<img src="comparison.png" alt="Stability and paired effort comparison"><table><tr>'+''.join('<th>'+x+'</th>' for x in ['Seed','Baseline','Window','Success','Physical failures','Common successes','Squared torque change'])+'</tr>'+''.join(table)+'</table>'
doc+='<p><b>Selection:</b> '+html.escape(selection)+'. No policy has been promoted.</p>'
doc+='<p><b>What changed:</b> a frozen starting policy still controls approach. A small PPO policy learns two coordinated knee/hip/ankle adjustments, held for 200 ms and enabled only after 12 seconds. The first 12 seconds of every final development and holdout trajectory match the starting policy exactly. Training uses a direct whole-body torque objective, a critic-only warmup and unchanged support/failure criteria. This changes several design choices together, so it does not isolate one cause of the previous regression.</p>'
doc+='<p><b>Budget:</b> two seeds, each eight critic-only plus 32 actor-enabled optimizer updates; 115,200 physical transitions each, including reset warmup. Actor-enabled updates containing only the approach phase have masked actor advantages. The separate preflight used 46,080 physical transitions. No extra training was added for evaluation recovery.</p>'
doc+='<p><b>Limits:</b> eight holdout geometries are each repeated twice. They are nearby rigid slabs, not arbitrary furniture. Two seeds are limited evidence. Torque is an effort proxy, not electrical energy or proof of a global minimum. No effort-saving claim uses failed trajectory fragments.</p>'
doc+='<p><a href="summary.csv">All paired metrics</a> · <a href="paired_cases.csv">Per-surface paired results</a> · <a href="case_outcomes.csv">Every success and physical failure</a> · <a href="verification.json">Independent validation</a> · <a href="approach_identity.json">Exact approach comparison</a></p></html>'
(out/'findings.html').write_text(doc)
files={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file()};(out/'SHA256.json').write_text(json.dumps(files,indent=2));print(out)
