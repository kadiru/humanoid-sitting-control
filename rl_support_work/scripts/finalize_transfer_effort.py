"""Wait for independently verified workers and export an honest final comparison."""
import argparse,hashlib,html,json,shutil,time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA=Path('/data/humanoid-sitting-perception/transfer_effort_01')
def read(p):return json.loads(p.read_text())
def copy_tree(src,dest):shutil.copytree(src,dest,ignore=shutil.ignore_patterns('events.out.*','model_*.pt'))
def main():
    p=argparse.ArgumentParser();p.add_argument('--wait',action='store_true');a=p.parse_args()
    while True:
        states={s:read(DATA/'ppo'/s/'status.json')['phase'] if (DATA/'ppo'/s/'status.json').exists() else 'not_started' for s in ['sitting','leaning']}
        if all(x=='verified_complete' for x in states.values()):break
        if any('failed' in x for x in states.values()):raise RuntimeError(states)
        if not a.wait:raise RuntimeError(states)
        time.sleep(20)
    out=DATA/'final_export_01';out.mkdir(exist_ok=False);summary={};sections=[];fig,axes=plt.subplots(2,3,figsize=(16,9),layout='constrained')
    for row,s in enumerate(['sitting','leaning']):
        folder=DATA/'ppo'/s;v=read(folder/'verified_01/verification.json');assert v['passed'];protocol=read(folder/'protocol.json');summary[s]=v;dest=out/s;dest.mkdir()
        for name in ['protocol.json','status.json']:shutil.copy2(folder/name,dest/name)
        shutil.copy2(folder/'verified_01/verification.json',dest/'verification.json');copy_tree(folder/'frozen',dest/'frozen')
        for d in folder.iterdir():
            if d.is_dir() and (d.name.startswith('train_') or d.name.startswith('dev_') or d.name.startswith('holdout_')):copy_tree(d,dest/d.name)
            elif d.name.startswith('seed_') and d.suffix=='.json':shutil.copy2(d,dest/d.name)
        copy_tree(DATA/'assessment'/s/'verified_01',dest/'assessment_verification')
        for label in ['reference']+[f'policy_{n}' for n in protocol['parent_seeds']]:copy_tree(DATA/'assessment'/s/label,dest/('assessment_'+label))
        copy_tree(DATA/'preflight'/s,dest/'preflight')
        rows=[]
        for col,bank in enumerate(['development','holdout']):
            labels=['reference']+[f'parent_{n}' for n in protocol['parent_seeds']]+[f'final_{n}' for n in protocol['seeds']]
            results=[]
            for label in labels:
                if bank=='holdout':r=read(folder/('holdout_'+label)/'results.json')
                elif label=='reference':r=read(DATA/'assessment'/s/'reference/results.json')
                elif label.startswith('parent_'):r=read(DATA/'assessment'/s/label.replace('parent_','policy_')/'results.json')
                else:r=read(folder/f'dev_{label[6:]}_block3/results.json')
                results.append(r);rows.append(f'<tr><td>{bank}</td><td>{label}</td><td>{r["successes"]}/{r["count"]}</td><td>{sum(any(e["failures"].values()) for e in r["episodes"])}</td></tr>')
            x=np.arange(len(labels));ax=axes[row,col];ax.bar(x-.17,[r['successes'] for r in results],.34,color='#287951',label='Stable support');ax.bar(x+.17,[sum(any(e['failures'].values()) for e in r['episodes']) for r in results],.34,color='#b64840',label='Physical failure');ax.set(title=s.title()+' · '+bank,xticks=x,xticklabels=[l.replace('_','\n') for l in labels],ylim=(0,(18 if bank=='development' else 16)+2),ylabel='Trials');ax.tick_params(axis='x',labelsize=8);ax.legend(fontsize=8)
        ax=axes[row,2];vals=[];names=[]
        for seed in protocol['seeds']:
            for bank in ['development','holdout']:
                key=f'dev_{seed}_block3' if bank=='development' else f'holdout_final_{seed}'
                x=v['comparisons'][key]['corrected_reference']['grouped']['metrics_whole_trial']['total_squared_torque_Nm2'];vals.append(x['change_percent'] if x['change_percent'] is not None else np.nan);names.append(f'{seed}\n{bank}\nn={x["common_successes"]}')
        ax.bar(range(len(vals)),vals,color=['#287951' if x<0 else '#ae7839' for x in vals]);ax.axhline(0,color='black',lw=.7);ax.set(title=s.title()+' · final effort vs corrected reference',xticks=range(len(vals)),xticklabels=names,ylabel='Mean sum of squared torque change (%)');ax.tick_params(axis='x',labelsize=8)
        details=[]
        for label,baselines in v['comparisons'].items():
            for baseline,comparison in baselines.items():
                details.append('<details><summary>'+html.escape(label+' vs '+baseline)+'</summary><table><tr><th>Window</th><th>Metric</th><th>Common successes</th><th>Policy / baseline</th><th>Change</th></tr>')
                for window,metrics in comparison['grouped'].items():
                    for metric,x in metrics.items():
                        values='—' if x['policy'] is None else f'{x["policy"]:.4f} / {x["baseline"]:.4f}';change='—' if x['change_percent'] is None else f'{x["change_percent"]:+.2f}%'
                        details.append(f'<tr><td>{window}</td><td>{metric}</td><td>{x["common_successes"]}</td><td>{values}</td><td>{change}</td></tr>')
                details.append('</table></details>')
        sections.append('<h2>'+s.title()+'</h2><table><tr><th>Bank</th><th>Controller</th><th>Success</th><th>Physical failures</th></tr>'+''.join(rows)+'</table><p>Development-only checkpoint selection: '+html.escape(json.dumps(v['selection']))+'</p>'+''.join(details))
    fig.suptitle('Effort-focused PPO: stability first, then matched whole-body torque',fontsize=17);fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    shutil.copy2('/home/kadir/humanoid-sitting/docs/TRANSFER_EFFORT_PROTOCOL.md',out/'PROTOCOL.md')
    copy_tree(DATA/'banks',out/'banks');copy_tree(DATA/'calibration',out/'calibration')
    (out/'report.html').write_text('<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Effort PPO on changed heights and slopes</title><style>body{font:16px/1.5 system-ui;max-width:1400px;margin:32px auto;padding:0 22px;background:#f5f7f3;color:#1b3537}table{border-collapse:collapse;background:white;width:100%}td,th{padding:9px;border-bottom:1px solid #ccd5d0;text-align:left}img{width:100%}summary{cursor:pointer;font-weight:600;margin:16px 0}</style><h1>Effort PPO on changed heights and slopes</h1><p>The final checkpoints are reported whether or not they improved. Selected checkpoints are separate development-only candidates; fallback means no trained candidate passed the stability-retention rule. No automatic policy promotion.</p><img src="comparison.png" alt="Final support success, physical failures and common-success torque comparisons">'+''.join(sections)+'<h2>Scope and limits</h2><p>Each task trains two seeds for 128 new updates from standing, 147,456 transitions per seed including indexed-reset warmup, total 589,824. Development: 9 surfaces per task, twice repeated; reserved holdout: 8 surfaces per task, twice repeated. Repeated placements are not independent surface geometries. The reference motion and physical criteria remain unchanged; actor/critic inputs are sensor/reference data.</p><p>Compare failures and support success before effort. All effort values use common successful full 20-second trials; no failed-fragment savings. Squared/absolute torque and mean instantaneous joint RMS are effort proxies, not electrical energy or a global optimum. Reference assistance, narrow rigid slabs, nearby heights/slopes and two seeds do not establish arbitrary-surface sitting or leaning.</p><p>Per-case height/slope, perception errors/availability, all validation, active/warmup accounting, checkpoint hashes, selection, and paired leg/upper-body metrics are in summary.json and the copied validation/results. Human review of plots and final conclusions remains required.</p>')
    (out/'SHA256.json').write_text(json.dumps({str(f.relative_to(out)):hashlib.sha256(f.read_bytes()).hexdigest() for f in out.rglob('*') if f.is_file()},indent=2))
    status=dict(phase='verified_results_ready_for_review',output=str(out),note='Agent must independently inspect, copy to Mac, visually review and deliver; no policy promoted.')
    (DATA/'status.json').write_text(json.dumps(status,indent=2));print(out)
if __name__=='__main__':main()
