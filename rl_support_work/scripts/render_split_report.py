"""Render verified results; produces no training and changes no study artifacts."""
import argparse,json,html,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
def read(p):return json.loads(p.read_text())
def save(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False))
def paired(a,b,window,metric,standing=False):
 z=[(x[window][metric],y[window][metric]) for x,y in zip(a['episodes'],b['episodes']) if x['success'] and y['success'] and (not standing or x['start_stage']==0)]
 return (len(z),100*(sum(x for x,y in z)/sum(y for x,y in z)-1)) if z else (0,None)
def main():
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--preflight',action='store_true');a=p.parse_args();out=a.output;out.mkdir(exist_ok=False)
 docs=[];summary={};fig,axes=plt.subplots(2,3,figsize=(15,8.4),constrained_layout=True)
 for row,surface in enumerate(['sitting','leaning']):
  if a.preflight:
   folder=a.data;suffix='02' if surface=='leaning' else '01';v=read(folder/f'{surface}_preflight_verified_{suffix}/verification.json');assert v['passed']
   labels=['fixed','old_reference','reference','smoke_PPO'];paths=[folder/f'{surface}_{x}' for x in ['fixed_01','old_reference_01',f'reference_validation_{suffix}',f'smoke_evaluation_{suffix}']]
  else:
   folder=a.data/surface;v=read(folder/'verified_01/verification.json');assert v['passed'];protocol=read(folder/'protocol.json');labels=['fixed','old_reference','reference']+[f'PPO_{s}' for s in protocol['seeds']];paths=[folder/x for x in ['fixed','old_reference','reference']+[f'policy_{s}' for s in protocol['seeds']]]
  results={l:read(p/'results.json') for l,p in zip(labels,paths)};summary[surface]={}
  for l,p in zip(labels,paths):
   dest=out/surface/l;dest.mkdir(parents=True);shutil.copy2(p/'results.json',dest/'results.json');shutil.copy2(p/'manifest.json',dest/'manifest.json');shutil.copy2(p/'startup_validation.json',dest/'startup_validation.json');shutil.copy2(p/'sensor_audit.json',dest/'sensor_audit.json');shutil.copy2(p/'first_failures.json',dest/'first_failures.json')
  save(out/surface/'verification.json',v)
  ax=axes[row,0];success=[r['successes'] for r in results.values()];count=next(iter(results.values()))['count'];fail=[sum(any(e['failures'].values()) for e in r['episodes']) for r in results.values()];x=np.arange(len(labels));ax.bar(x-.17,success,.34,label='Stable support',color='#237a55');ax.bar(x+.17,fail,.34,label='Physical failure',color='#c24b48');ax.set(ylim=(0,count+2),title=f'{surface.title()}: all starting stages',ylabel=f'Trials (of {count})',xticks=x,xticklabels=[l.replace('_','\n') for l in labels]);ax.legend(fontsize=8);ax.tick_params(axis='x',labelsize=9)
  ax=axes[row,1]
  for l,r in results.items():
   vals=[sum(e['success'] and e['start_stage']==s for e in r['episodes'])/sum(e['start_stage']==s for e in r['episodes']) for s in range(7)];ax.plot(range(7),vals,'o-',label=l)
  ax.set(title=f'{surface.title()}: support by start',xlabel='Starting stage (0 = standing, 6 = supported)',ylabel='Success fraction',xticks=range(7),ylim=(-.05,1.05));ax.legend(fontsize=8)
  ax=axes[row,2];comp_labels=[l for l in labels if l not in ['fixed','old_reference']];changes=[];textlabels=[]
  for l in comp_labels:
   b='old_reference' if l=='reference' else 'reference';n,pct=paired(results[l],results[b],'metrics_whole_trial','total_squared_torque_Nm2');changes.append(pct if pct is not None else np.nan);textlabels.append(l.replace('_','\n')+f'\nvs {b.replace("_"," ")}\nn={n}')
  ax.bar(range(len(changes)),changes,color=['#237a55' if np.isfinite(x) and x<0 else '#ba7540' for x in changes]);ax.axhline(0,c='black',lw=.8);ax.set(title=f'{surface.title()}: paired joint effort',ylabel='Change in mean total squared torque (%)',xticks=range(len(changes)),xticklabels=textlabels);ax.tick_params(axis='x',labelsize=8)
  table=[]
  for l,r in results.items():
   stand=[e for e in r['episodes'] if e['start_stage']==0];entry=dict(support_successes=r['successes'],count=r['count'],physical_failures=sum(any(e['failures'].values()) for e in r['episodes']),standing_successes=sum(e['success'] for e in stand),standing_count=len(stand),comparisons={})
   for baseline in ['fixed','old_reference','reference']:
    if l==baseline:continue
    entry['comparisons'][baseline]={w:{m:dict(zip(['common_successes','mean_change_percent'],paired(r,results[baseline],w,m))) for m in ['total_squared_torque_Nm2','total_absolute_torque_Nm','all_joint_rms_Nm','leg_squared_torque_Nm2','upper_squared_torque_Nm2']} for w in ['metrics_whole_trial','metrics_last10s','metrics_last5s']}
   summary[surface][l]=entry
   table.append(f'<tr><td>{html.escape(l)}</td><td>{r["successes"]}/{r["count"]}</td><td>{entry["physical_failures"]}</td><td>{entry["standing_successes"]}/{len(stand)}</td></tr>')
  docs.append('<h2>'+surface.title()+'</h2><table><tr><th>Controller</th><th>Support success</th><th>Physical failures</th><th>From standing</th></tr>'+''.join(table)+'</table>')
 title='Corrected references and two-update smoke — not final PPO results' if a.preflight else 'Separate sitting and leaning PPO — verified comparison'
 fig.suptitle(title,fontsize=17);fig.savefig(out/'comparison.png',dpi=160);plt.close(fig);save(out/'summary.json',summary)
 caution='The reference repairs are hand-engineered sensor/command corrections. Two-update smoke confirms execution and retention, not learned improvement.' if a.preflight else 'Each policy had128 PPO updates, including64 fromstanding. Check per-seed results; do not pool away failures.'
 (out/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>'+title+'</title><style>body{font:17px/1.55 system-ui;background:#f5f3ee;color:#1c2a30;max-width:1250px;margin:40px auto;padding:0 24px}h1{font-size:32px}table{border-collapse:collapse;width:100%;background:white}th,td{padding:10px;border-bottom:1px solid #ddd;text-align:left}img{width:100%;background:white}.note{padding:18px;background:#e5eee9;border-radius:8px}a{color:#17624a}</style><h1>'+title+'</h1><p class="note">'+caution+'</p><img src="comparison.png" alt="Stability, standing starts, and paired effort comparison">'+''.join(docs)+'<p>Each trial runs20 seconds; support requires the final5 seconds continuously qualifying with no physical failure. Fixed standing retention is not a sitting/leaning success. Repetitions use known states and two rigid surfaces; these are not unseen-surface tests. Effort includes only common full-duration successful trials. Torque is an effort proxy, not measured electrical energy.</p><p><a href="summary.json">Detailed effort comparisons</a> · Per-controller results, manifests and validation are stored beside this report.</p>')
 print(out/'report.html')
if __name__=='__main__':main()
