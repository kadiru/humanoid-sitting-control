import json,numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
out=Path('/data/humanoid-sitting-perception/contact_effort_repair_01/postcontact_unloading_probe_01_review')
report=json.loads((out/'review.json').read_text())
fig,ax=plt.subplots(1,3,figsize=(14,4.6),layout='constrained');names=['Repaired\nreference','Upright\ncontinuation1','Continuation2']
for i,row in enumerate(report['profiles']):
    ax[0].bar(i,row['successes'],color='#247d79');ax[0].bar(i,row['physical_failures'],bottom=row['successes'],color='#c66e50');ax[0].text(i,row['successes']+.15,f"{row['successes']}/9",ha='center')
    effort=row['paired']['metrics_last5s']['total_squared_torque_Nm2']['change_pct'];ax[1].bar(i,effort,color='#247d79');ax[1].text(i,effort,f"{effort:+.1f}%\nn={row['common_successes']}",ha='center',va='bottom')
    load=row['paired']['metrics_last5s']['vertical_support_load_bw'];ax[2].bar(i-.16,load['matched_reference']*100,.3,color='#bbc9cd',label='Matched reference' if i==0 else None);ax[2].bar(i+.16,load['probe']*100,.3,color='#247d79',label='Probe' if i==0 else None)
ax[0].set(title='Success / physical failure',ylim=(0,10),ylabel='Trials');ax[1].set(title='Final5s effort on common successes',ylabel='Squared torque change (%)');ax[1].set_ylim(-15,43);ax[1].axhline(0,color='#333',lw=.8);ax[2].set(title='Final5s weight supported by seat',ylabel='Percent body weight',ylim=(0,45));ax[2].legend(fontsize=8)
for a in ax:a.set_xticks(range(3),names)
fig.suptitle('Exploratory post-contact unloading — prescribed reference, not learned PPO\nNeither candidate passes the full development grid')
fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
