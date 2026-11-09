"""Read-only progress export for completed low-seat physical probes."""
import json,hashlib,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')
out=D/'normal_probe_export_03'
out.mkdir(exist_ok=False)
names=['damped_original_01','clearance_damped_01','load_feedback_original_02',
       'load_feedback_clearance_02','normal_descent_03','hover_rotate_normal_03']
audits=[json.loads((D/n/'independent_audit_v2.json').read_text()) for n in names]
fig,ax=plt.subplots(figsize=(12,5),layout='constrained')
x=np.arange(len(names));w=.25
for j,(key,color,label) in enumerate([('clean','#278461','Clean support'),('slipping','#d18c13','Any slipping'),('unwanted_contacts','#bd3843','Unwanted contact')]):
    ax.bar(x+(j-1)*w,[a['candidate'][key] for a in audits],w,color=color,label=label)
ax.axhline(8,color='#278461',ls='--',label='Matched parent: 8 clean')
ax.set(xticks=x,xticklabels=['Damping','Deeper path','Load feedback','Load + deeper','Normal descent','Rotate + normal'],ylim=(0,24),ylabel='Episodes out of 24',title='Repair remains incomplete: normal descent helps some cases; all candidates fail admission')
ax.legend(ncols=4,loc='upper center',fontsize=9)
ax.text(.01,-.20,'Slipping/contact counts may overlap. These are development probes, not new PPO or unseen tests.',transform=ax.transAxes,fontsize=10)
fig.savefig(out/'comparison.png',dpi=150);plt.close(fig)

p=D/'normal_descent_03/candidate';r=json.loads((p/'results.json').read_text());t=np.load(p/'trace.npz');e=np.load(p/'every_tick.npz');plans={a['case']:a for a in json.loads((p/'rolling_plan_audit.json').read_text())};origin=np.array(r['origins']);pelvis=r['body_names'].index('pelvis')
fig,axes=plt.subplots(3,2,figsize=(11,8),sharex=True,layout='constrained')
for col,i in enumerate([2,4]):
    a=plans[i];path=np.array(a['root_path']);u=np.clip((t['t']-10)/5*24,0,24);lo=np.minimum(u.astype(int),23);f=u-lo;target=(1-f[:,None])*path[lo,:3]+f[:,None]*path[lo+1,:3];root=t['body'][:,i,pelvis,:3]-origin[i]
    axes[0,col].plot(t['t'],1000*(root[:,0]-target[:,0]),label='Inward position error')
    axes[0,col].plot(t['t'],1000*(root[:,2]-target[:,2]),label='Vertical position error')
    pitch=Rotation.from_quat(t['body'][:,i,pelvis,3:]).as_euler('xyz')[:,1]
    axes[1,col].plot(t['t'],np.rad2deg(pitch),label='Actual pelvis pitch');axes[1,col].axhline(np.rad2deg(a['pitch']),ls='--',label='Planned pitch')
    load=(e['force'][:,i,:5]*e['normal'][:,i,:5]).sum(-1).sum(-1)/e['weight'][:,i]
    axes[2,col].plot(e['t'],load,label='Measured seat-normal force / BW')
    loaded=np.linalg.norm(e['force'][:,i],axis=-1)>.01*e['weight'][:,i,None];slip=(loaded&(e['slip'][:,i]>.02)).any(-1)
    axes[2,col].scatter(e['t'][slip],load[slip],s=10,color='#d18c13',label='Slip tick')
    wrong=e['wrong'][:,i]>10;axes[2,col].scatter(e['t'][wrong],load[wrong],s=12,color='#bd3843',label='Forbidden contact')
    for ax in axes[:,col]:
        ax.axvspan(10.5,11.5,color='#999999',alpha=.15);ax.set_xlim(10,16);ax.grid(alpha=.2);ax.legend(fontsize=7)
    geometry=json.loads((p/'geometry.json').read_text())[r['episodes'][i]['geometry_index']]
    axes[0,col].set_title(f"{100*geometry['height']:g} cm / +{geometry['pitch_deg']:g}° — "+('housing contact + slip' if i==2 else 'clean support'))
    axes[2,col].set_xlabel('Episode seconds')
axes[0,0].set_ylabel('Root error (mm)');axes[1,0].set_ylabel('Pelvis pitch (degrees)');axes[2,0].set_ylabel('Seat-normal load / BW')
fig.suptitle('Recorded normal-descent diagnosis; grey band = inherited balance handoff\nAssociation only: two separately versioned balance probes test this hypothesis',fontsize=12)
fig.savefig(out/'normal_diagnosis.png',dpi=150);plt.close(fig)
for name in ['normal_diagnosis_03.json','normal_foot_balance_screen_03.json','normal_revision_03.json','balance_revision_04.json','retention_targets.json','reservation.json']:
    shutil.copy2(D/name,out/name)
for name,a in zip(names,audits):(out/(name+'_audit.json')).write_text(json.dumps(a,indent=2))
rows=''.join(f'<tr><td>{n}</td><td>{a["candidate"]["stable"]}</td><td>{a["candidate"]["clean"]}</td><td>{a["candidate"]["slipping"]}</td><td>{a["candidate"]["unwanted_contacts"]}</td><td>{a["lost_stable_cases"]}</td></tr>' for n,a in zip(names,audits))
(out/'findings.html').write_text('''<!doctype html><html><meta charset="utf-8"><title>Low-seat repair progress</title><style>body{font:17px system-ui;max-width:1100px;margin:35px auto;padding:20px;color:#172231}table{border-collapse:collapse;width:100%}td,th{padding:9px;border-bottom:1px solid #ccc;text-align:left}img{width:100%}</style><h1>Physical repair remains unresolved</h1><p>Normal descent gained four clean successes but still has ten sliding cases, two housing contacts and two lost stable-support cases. Rotating before descent was worse. Neither is admitted for PPO. Two new balance-handoff probes are separate development tests; their results are not included here.</p><p>All counts use 24 complete 20-second tests. Contacts include forbidden body or illegal support patch. Conservative force-margin warnings remain separate. Exact startups and first ten seconds passed; bounded-command audits are not a physical-success certificate.</p><table><tr><th>Candidate</th><th>Stable</th><th>Clean</th><th>Slip</th><th>Contact union</th><th>Lost stable cases</th></tr>'''+rows+'''</table><img src="comparison.png"><img src="normal_diagnosis.png"><p>The grey interval is the old timed ankle-to-hip balance handoff. The timing is a hypothesis, not a single-factor causal conclusion. Sensor geometry was available. Strong backward root motion and pelvic rotation precede the housing contact.</p><p>PPO and the fresh holdout remain closed. A repair must remove targeted failures and preserve all 60 current successes across 72 development trials. This is reference-assisted simulation, not real-robot validation or sitting/leaning anywhere.</p></html>''')
(out/'SHA256.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()},indent=2))
print(out)
