"""Audit the separate prescribed unloading probe; never a learned-policy result."""
import json,hashlib,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path('/data/humanoid-sitting-perception/contact_effort_repair_01');p=ROOT/'postcontact_unloading_probe_01';out=ROOT/'postcontact_unloading_probe_01_review';out.mkdir(exist_ok=False)
def read(p):return json.loads(p.read_text())
r=read(p/'results.json');t=np.load(p/'trace.npz');start=np.load(p/'startup.npz');g=read(p/'geometry.json');rows=r['episodes'];assert len(rows)==27
assert all((x['steps'],x['duration_s'])==(1000,20.) for x in rows)
assert all(np.isfinite(t[k]).all() for k in t.files)
assert all(x['t']>12 for x in read(p/'first_failures.json'))
assert hashlib.sha256((p/'unloading_source.py').read_bytes()).hexdigest()==read(p/'unloading_manifest.json')['source_sha256']
report=dict(scope='Reference-only prescribed post-contact continuation, not PPO; frozen PPO studies unchanged. Development surfaces only.',profiles=[],first12_max_differences={},startup_max_differences={})
origins=np.array(r['origins'])
for b in range(3):
    es=rows[9*b:9*(b+1)];common=[(e,rows[i]) for i,e in enumerate(es) if e['success'] and rows[i]['success']]
    result=dict(profile=es[0]['profile'],successes=sum(e['success'] for e in es),count=9,physical_failures=sum(any(e['failures'].values()) for e in es),common_successes=len(common),paired={},outcomes=[])
    for e in es:
        geo=g[e['geometry_index']];result['outcomes'].append(dict(case=e['case'],height_m=geo['top_center'][2],slope_deg=geo['pitch_deg'],success=e['success'],final_hold_s=e['final_hold_s'],failures=e['failures']))
    for w in ['metrics_whole_trial','metrics_last10s','metrics_last5s']:
        result['paired'][w]={}
        for k in ['total_squared_torque_Nm2','total_absolute_torque_Nm','all_joint_rms_Nm','leg_squared_torque_Nm2','upper_squared_torque_Nm2','vertical_support_load_bw','foot_load_bw']:
            av=float(np.mean([a[w][k] for a,z in common]));bv=float(np.mean([z[w][k] for a,z in common]));result['paired'][w][k]=dict(probe=av,matched_reference=bv,change_pct=100*(av/bv-1) if bv else None)
    report['profiles'].append(result)
    report['first12_max_differences'][es[0]['profile']]={k:float(abs(t[k][:120,9*b:9*(b+1)]-t[k][:120,:9]).max()) for k in ['q','torque','command','geometry']}
    report['startup_max_differences'][es[0]['profile']]={k:float(abs(start[k][9*b:9*(b+1)]-start[k][:9]).max()) for k in ['q','command','geometry']}
report['interpretation']='Neither continuation is validated across the development grid. The8/9 candidate transfers more weight to the seat and lowers effort on common successes but introduces one forbidden-contact failure. Do not promote or count as a learned improvement. Parallel placement differences are reported, not asserted bitwise equal.'
(out/'review.json').write_text(json.dumps(report,indent=2));shutil.copy2(__file__,out/'review_source.py')
fig,ax=plt.subplots(1,3,figsize=(14,4.6),layout='constrained');names=['Repaired\nreference','Upright\ncontinuation1','Continuation2']
for i,row in enumerate(report['profiles']):
    ax[0].bar(i,row['successes'],color='#247d79');ax[0].bar(i,row['physical_failures'],bottom=row['successes'],color='#c66e50');ax[0].text(i,row['successes']+.15,f"{row['successes']}/9",ha='center')
    effort=row['paired']['metrics_last5s']['total_squared_torque_Nm2']['change_pct'];ax[1].bar(i,effort,color='#247d79');ax[1].text(i,effort,f"{effort:+.1f}%\nn={row['common_successes']}",ha='center',va='bottom')
    load=row['paired']['metrics_last5s']['vertical_support_load_bw'];ax[2].bar(i-.16,load['matched_reference']*100,.3,color='#bbc9cd',label='Matched reference' if i==0 else None);ax[2].bar(i+.16,load['probe']*100,.3,color='#247d79',label='Probe' if i==0 else None)
ax[0].set(title='Success / physical failure',ylim=(0,10),ylabel='Trials');ax[1].set(title='Final5s effort on common successes',ylabel='Squared torque change (%)');ax[1].axhline(0,color='#333',lw=.8);ax[2].set(title='Final5s weight supported by seat',ylabel='Percent body weight',ylim=(0,45));ax[2].legend(fontsize=8)
for a in ax:a.set_xticks(range(3),names)
fig.suptitle('Exploratory post-contact unloading — prescribed reference, not learned PPO\nNeither candidate passes the full development grid')
fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
(out/'SHA256.json').write_text(json.dumps({str(x.relative_to(out)):hashlib.sha256(x.read_bytes()).hexdigest() for x in out.rglob('*') if x.is_file()},indent=2));print(out);print(json.dumps({k:report[k] for k in ['first12_max_differences','startup_max_differences']},indent=2))
