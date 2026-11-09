from pathlib import Path
import json,hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');O=D/'progress_report_349'
t=json.loads((O/'timing_348.json').read_text());phases=[dict(phase=k,**v) for k,v in t['phase_time'].items()]
plt.rcParams.update({'font.size':11,'axes.spines.right':False,'axes.spines.top':False})
fig,axs=plt.subplots(1,2,figsize=(14,6),layout='constrained')
labels=['Global 0.15\ncommand metric','Reserved ankle rates','Reserved + normal\ncap + 0.5s hover'];vals=[sum(v['global015_s'] for v in t['phase_time'].values()),sum(v['reserved_s'] for v in t['phase_time'].values()),t['continuous_reserved_normal_hover_time_s']]
bars=axs[0].bar(labels,vals,color=['#387f91','#ca8c32','#a94344']);axs[0].axhline(13.5,color='#333',ls='--',label='13.5s gate');axs[0].bar_label(bars,fmt='%.3f',label_type='center',color='white',fontsize=12);axs[0].set(ylabel='Seconds',ylim=(0,20),title='Static command metric omits required timing');axs[0].legend(loc='upper left')
x=np.arange(4);rv=[p['reserved_s'] for p in phases];nv=[p['normal_extra_s'] for p in phases];hv=[p['hover_s'] for p in phases]
axs[1].bar(x,rv,label='Reserved rates',color='#387f91');axs[1].bar(x,nv,bottom=rv,label='Additional normal-descent time',color='#ca8c32');axs[1].bar(x,hv,bottom=np.array(rv)+nv,label='Synthetic quiet hold',color='#a94344');axs[1].set(xticks=x,xticklabels=[p['phase'].title() for p in phases],ylabel='Seconds',title='Fixed continuous-polyline timing by phase');axs[1].legend(loc='upper center',bbox_to_anchor=(.5,-.13),ncol=1)
fig.suptitle('Conditional offline timing models — no measured movement or physical admission',fontsize=13);fig.savefig(O/'timing_budget_review.png',dpi=160);plt.close(fig)
original=(O/'findings.html').read_text();review=original.replace('src="timing_budget.png"','src="timing_budget_review.png"');assert review.replace('src="timing_budget_review.png"','src="timing_budget.png"')==original
(O/'review.html').write_text(review)
(O/'review_audit.json').write_text(json.dumps(dict(formatting_only=True,change='Move plot numeric bar labels inside bars to avoid threshold-line overlap; one HTML image source changed. Values, original plot/HTML/data retained.',numeric_data_changed=False),indent=2))
(O/'review_SHA256.json').write_text(json.dumps({n:hashlib.sha256((O/n).read_bytes()).hexdigest() for n in ['timing_budget_review.png','review.html','review_audit.json']},indent=2))
