"""Formatting-only supplemental plot review; originals remain untouched."""
from pathlib import Path
import json,hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');O=D/'progress_report_292'
r=json.loads((O/'results.json').read_text());late=[s for s in r['summaries'] if s['window']==[11,14.98]];labels=[f"{s['run']} / case {s['case']}" for s in late];x=np.arange(4);colors=['#2878a5','#bd7722']
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(1,2,figsize=(14,5.5))
for j in range(2):
 ax[0].bar(x+(j-.5)*.3,[s['raw_roll_median'][j] for s in late],.3,label=['Left','Right'][j],color=colors[j]);ax[1].bar(x+(j-.5)*.3,[s['modeled_roll_dv_bound_median'][j] for s in late],.3,label=['Left','Right'][j],color=colors[j])
ax[0].axhline(.05,color='#444',ls='--',label='Quiet limit');ax[0].set_title('Recorded raw velocity');ax[0].set_ylabel('Median ankle-roll qd (rad/s)');ax[0].set_ylim(0,.37);ax[0].legend(ncol=3,loc='upper center',fontsize=10)
ax[1].set_title('Ideal sticking model: one-step sensitivity envelope');ax[1].set_ylabel('Maximum incremental |Δqd| (rad/s)');ax[1].set_ylim(0,.0023);ax[1].legend(ncol=2,loc='upper center')
for aa in ax:aa.set_xticks(x,labels)
fig.suptitle('Different quantities: not a prediction of future settling');fig.tight_layout();fig.savefig(O/'velocity_and_model_review.png',dpi=150,bbox_inches='tight');plt.close(fig)
fig,ax=plt.subplots(1,2,figsize=(14,5.5));ax[0].bar(x,[s['pelvis_roll_dq_bound_median']*1000 for s in late],color='#76629a');ax[0].set_ylabel('Pelvis roll envelope (mrad)');ax[1].bar(x,[s['pelvis_lateral_dq_bound_median']*1e6 for s in late],color='#548b70');ax[1].set_ylabel('Lateral displacement envelope (µm)')
for aa in ax:aa.set_xticks(x,labels)
fig.suptitle('Conditional model response: pelvis free, both soles constrained — not executed');fig.tight_layout();fig.savefig(O/'coupled_response_review.png',dpi=150,bbox_inches='tight');plt.close(fig)
s=(O/'findings.html').read_text();t=s.replace('src="velocity_and_model.png"','src="velocity_and_model_review.png"').replace('src="coupled_response.png"','src="coupled_response_review.png"');assert t.replace('_review.png','.png')==s
(O/'review.html').write_text(t)
files=['velocity_and_model_review.png','coupled_response_review.png','review.html'];audit=dict(change='Plot legend positioning, plot margins and shorter axis labels only; plotted numeric arrays and HTML numeric/text content unchanged. Original report/plots preserved.',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest());(O/'review_audit.json').write_text(json.dumps(audit,indent=2));files+=['review_audit.json'];(O/'review_SHA256.json').write_text(json.dumps({n:hashlib.sha256((O/n).read_bytes()).hexdigest() for n in files},indent=2))
