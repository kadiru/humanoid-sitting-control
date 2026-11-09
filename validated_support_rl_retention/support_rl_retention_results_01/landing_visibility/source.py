"""Offline visibility audit; ground-truth support is an event label only."""
import csv
import json
from pathlib import Path
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

store=Path('/data/humanoid-sitting-perception');suite=store/'support_rl_retention_results_01'
out=suite/'landing_visibility';out.mkdir(exist_ok=False);shutil.copy2(__file__,out/'source.py')
read=lambda p:json.loads(p.read_text())
records=[];summaries={};fig,axes=plt.subplots(1,3,figsize=(13,6),sharey=True,constrained_layout=True)
labels={'parent':'Preserved policy','control':'Continued PPO','anchored':'PPO + parent penalty'}
for col,arm in enumerate(labels):
    path=store/f'support_rl_retention_easy_{arm}_01'
    z=np.load(path/'rollout.npz');t=z['t'];valid=z['live'][...,8]>0;rows=read(path/'results.json')['episodes']
    for i in range(16):
        # Event is first sampled support above 25% body weight, not the exact
        # onset of touch. The original safety flags are checked at 50 Hz.
        touch=np.flatnonzero(z['support'][:,i]>.25);fail=np.flatnonzero(~z['alive'][:,i])
        if not len(touch):continue
        k=int(touch[0]);window=(t>=t[k]-1.-1e-6)&(t<=t[k]+.5+1e-6)
        before=k;after=k
        if not valid[k,i]:
            while before>0 and not valid[before-1,i]:before-=1
            while after<len(t)-1 and not valid[after+1,i]:after+=1
        record=dict(arm=arm,case=i,success=rows[i]['success'],support_event_s=float(t[k]),
            current_depth_valid_at_event=bool(valid[k,i]),near_contact_valid_fraction=float(valid[window,i].mean()),
            invalid_run_first_sample_s=float(t[before]) if not valid[k,i] else None,
            invalid_run_last_sample_s=float(t[after]) if not valid[k,i] else None,
            invalid_run_reaches_end=bool(not valid[k,i] and after==len(t)-1),
            first_latched_failure_s=float(t[fail[0]]) if len(fail) else None)
        records.append(record)
        axes[col].scatter(t,np.full(len(t),i),c=np.where(valid[:,i],'#0f766e','#cbd5e1'),s=17,marker='s',linewidths=0)
        axes[col].scatter([t[k]],[i],color='black',marker='|',s=140)
        if len(fail):axes[col].scatter([t[fail[0]]],[i],color='#dc2626',marker='x',s=60,zorder=4)
    part=[r for r in records if r['arm']==arm]
    summaries[arm]=dict(cases_with_support_event=len(part),invalid_at_event=sum(not r['current_depth_valid_at_event'] for r in part),
        mean_valid_fraction_in_event_window=float(np.mean([r['near_contact_valid_fraction'] for r in part])),
        failed_cases=[r['case'] for r in part if not r['success']])
    axes[col].set(xlim=(10,15),ylim=(15.6,-.6),xlabel='Time (seconds)',title=labels[arm],yticks=range(16))
    axes[col].grid(axis='y',alpha=.15)
axes[0].set_ylabel('Known easier test case')
fig.suptitle('Current depth availability around landing\nEvent marker: first sampled seat support above 25% of body weight',fontsize=13)
axes[1].legend(handles=[Patch(color='#0f766e',label='Usable current depth'),Patch(color='#cbd5e1',label='Unavailable'),
    Line2D([0],[0],color='black',marker='|',linestyle='None',label='Support event'),
    Line2D([0],[0],color='#dc2626',marker='x',linestyle='None',label='Latched failure')],loc='upper center',bbox_to_anchor=(.5,-.12),ncol=2,fontsize=8)
fig.savefig(out/'visibility.png',dpi=150);plt.close(fig)
report=dict(summaries=summaries,cases=records,trace_frequency_Hz=10,
    event='First sampled seat support >25% body weight, offline simulator label only',
    window='One second before through half a second after the event',
    limitations='No causal intervention. Parent succeeds despite gaps. This audits the live ten-channel depth feature, not all robot perception or its proprioceptive contact estimate. Stored traces omit per-frame rejection reasons; do not infer the exact occlusion cause from validity alone.')
(out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
with (out/'cases.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
lines=['# Landing visibility diagnostic','',report['event']+'. '+report['window']+'.','',
    '| Policy | Current depth unavailable at event |','|---|---:|']
for arm,s in summaries.items():lines.append(f"| {labels[arm]} | {s['invalid_at_event']}/{s['cases_with_support_event']} |")
lines+=['',report['limitations'],'',
    'Case 15 is lost by both trained candidates. All three have unavailable current depth at the support event. Accepted measurements remain accurate, but accuracy on accepted frames does not establish continuous availability during landing.','',
    'Next diagnostic: record raw sensor frames and rejection reasons through the landing phase; evaluate a short-term kinematic/geometry estimate with explicit uncertainty when a fresh plane cannot be fitted. Keep simulator truth as evaluation labels, and test on training geometries before any new policy experiment. Also test parent sensitivity to small command changes, since the preserved policy can succeed with the same validity gaps.','',
    '![Availability during landing](visibility.png)','']
(out/'report.md').write_text('\n'.join(lines))
print(json.dumps(summaries,indent=2))
