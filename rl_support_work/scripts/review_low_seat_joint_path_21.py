"""Formatting-only supplemental figure; preserve original exported hashes."""
import json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');out=D/'progress_report_21'
def main():
    p=D/'joint_path_17/candidate';t=np.load(p/'trace.npz');r=json.loads((p/'results.json').read_text());g=json.loads((p/'geometry.json').read_text());plans={a['case']:a for a in json.loads((p/'rolling_plan_audit.json').read_text())}
    audit=json.loads((out/'physical_audit.json').read_text());orig=np.asarray(r['origins']);pelvis=r['body_names'].index('pelvis')
    fig,ax=plt.subplots(1,2,figsize=(12,4.5));keys=['stable','clean','slipping','unwanted_contacts'];x=np.arange(4)
    for j,(name,data,color) in enumerate([('Best normal3s',audit['best_probe'],'#237a57'),('Rejected joint path',audit['candidate'],'#b94b43')]):
        bars=ax[0].bar(x+(j-.5)*.34,[data[k] for k in keys],.34,label=name,color=color);ax[0].bar_label(bars,padding=3)
    ax[0].set_xticks(x,['Stable','Clean','Slip cases','Unwanted contacts'],rotation=12);ax[0].set_ylim(0,26);ax[0].legend();ax[0].set_title('24 matched development probe episodes')
    mask=(t['t']>=8)&(t['t']<=16)
    for i in [0,6,10]:
        a=plans[i];roots=np.asarray(a['root_path']);u=np.clip((t['t']-8)/7*24,0,24);lo=np.minimum(u.astype(int),23);f=u-lo
        target=(1-f[:,None])*roots[lo,:3]+f[:,None]*roots[lo+1,:3]
        error=(t['body'][:,i,pelvis,:3]-orig[i]-target)@np.asarray(a['normal']);geo=g[r['episodes'][i]['geometry_index']]
        ax[1].plot(t['t'][mask],1000*error[mask],label=f"{geo['height']*100:g} cm / +{geo['pitch_deg']:g}°")
    ax[1].set_xlim(8,16);ax[1].axhline(0,color='black',lw=.7);ax[1].set_ylabel('Root tracking error along seat normal (mm)');ax[1].set_xlabel('Episode time (s)');ax[1].legend(fontsize=9);ax[1].set_title('Tracking errors near touchdown')
    fig.suptitle('Rejected physical repair — no PPO or fresh holdout',fontsize=14);fig.tight_layout();assert not (out/'comparison_review.png').exists();fig.savefig(out/'comparison_review.png',dpi=160);plt.close(fig)
    (out/'review.html').write_text((out/'findings.html').read_text().replace('src="comparison.png"','src="comparison_review.png"'))
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    (out/'review_audit.json').write_text(json.dumps(dict(scope='Formatting only: root tracking plot autoscale now uses the displayed8–16s interval instead of hidden pre8s samples. All original18 export hashes remain unchanged; no data or conclusion changed.',hashes={n:sha(out/n) for n in ['findings.html','comparison.png','review.html','comparison_review.png']}),indent=2))
if __name__=='__main__':main()
