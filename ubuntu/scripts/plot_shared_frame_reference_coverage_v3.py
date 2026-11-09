"""Offline availability overview; withheld outputs are never plotted as measurements."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parents[1]
out=Path('/data/humanoid-sitting-perception/shared_frame_reference_results_03')
summary=json.loads((out/'results.json').read_text())
runs=summary['runs']; tags=[name for name in runs if 'baseline' in runs[name]]
fig,(ax,timeline)=plt.subplots(1,2,figsize=(15,7),gridspec_kw={'width_ratios':[1.1,1]})
y=np.arange(len(tags));before=[100*runs[n]['baseline']['alignment_valid']/runs[n]['samples'] for n in tags]
after=[100*runs[n]['alignment_valid']/runs[n]['samples'] for n in tags]
ax.barh(y+.16,before,height=.30,label='Initial scan only',color='#8896a8')
ax.barh(y-.16,after,height=.30,label='Verified reference views',color='#128267')
ax.set_yticks(y,tags);ax.invert_yaxis();ax.set(xlim=(0,102),xlabel='Relative-pose availability (% of all frames)')
ax.set_title('Original 17 regression conditions');ax.legend(loc='lower right');ax.grid(axis='x',alpha=.15)
labels=[]
for i,(case,title) in enumerate([(0,'chair'),(1,'couch'),(2,'ellipse')]):
    for offset,(prefix,color,label) in enumerate([('integrity','#8896a8','initial'),('reference','#128267','views')]):
        version='03' if prefix=='integrity' else '03'
        rows=json.loads((root/f'runs/shared_frame_{prefix}_holdout_occlusion_{version}/estimates.json').read_text())
        rows=[r for r in rows if r['case']==case]
        t=np.array([r['stamp'] for r in rows]);valid=np.array([r['valid'] for r in rows]); yy=2*i+offset
        timeline.scatter(t,np.full(len(t),yy),s=8,color='#e5e5e5',marker='s')
        timeline.scatter(t[valid],np.full(valid.sum(),yy),s=9,color=color,marker='s')
        bank=np.array([r['registration']['reference']=='direct_anchor_view_consensus' for r in rows])
        timeline.scatter(t[valid&bank],np.full((valid&bank).sum(),yy),s=20,color='#de9b20',marker='|')
        labels.append(title+' / '+label)
timeline.set_yticks(range(6),labels);timeline.invert_yaxis()
for start,end in [(6,6.5),(8,10)]:timeline.axvspan(start,end,color='#cc5442',alpha=.10)
timeline.set(xlim=(0,16),xlabel='Sensor time (s)',title='Previous three-surface bank / depth blackout')
timeline.grid(axis='x',alpha=.15)
fig.suptitle('Reference transitions recover coverage while retaining unknown intervals',fontsize=16)
fig.text(.52,.035,'Light gray: unavailable. Orange ticks: reference consensus.\nShaded intervals: all depth removed. Other accepted frames use the initial scan.',fontsize=10)
fig.tight_layout(rect=(0,.085,1,.95));fig.savefig(out/'coverage.png',dpi=140);plt.close(fig)
print('COVERAGE_PLOT_COMPLETE',out/'coverage.png',flush=True)
