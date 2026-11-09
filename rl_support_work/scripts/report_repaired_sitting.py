"""Read-only report/export for the frozen sitting repair and effort comparison."""
import argparse,csv,hashlib,html,json,shutil,subprocess
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def read(p):return json.loads(Path(p).read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False))
def csvfile(p,rows):
    if rows:
        with p.open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def table(headers,rows):
    return '<table><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</table>'

p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--verification',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
v=read(a.verification);assert v['passed'];out=a.output;out.mkdir(exist_ok=False)
root=a.study.parent;pre=root/'sitting_preflight_01';old=Path('/data/humanoid-sitting-perception/transfer_effort_01/assessment/sitting')
windows=['metrics_whole_trial','metrics_last10s','metrics_last5s']
metrics=['total_squared_torque_Nm2','total_absolute_torque_Nm','all_joint_rms_Nm','leg_squared_torque_Nm2','upper_squared_torque_Nm2','foot_load_bw','vertical_support_load_bw']
dirs={
 'development':{'old_reference':old/'reference','repaired_reference':pre/'reference',**{f'old_parent_{s}':old/f'policy_{s}' for s in [55211,55217]},**{f'repaired_parent_{s}':pre/f'parent_{s}' for s in [55211,55217]},**{f'final_{s}':a.study/f'dev_{s}' for s in [56211,56217]}},
 'holdout':{'old_reference':a.study/'holdout_old_reference','repaired_reference':a.study/'holdout_repaired_reference',**{f'old_parent_{s}':a.study/f'holdout_old_parent_{s}' for s in [55211,55217]},**{f'repaired_parent_{s}':a.study/f'holdout_repaired_parent_{s}' for s in [55211,55217]},**{f'final_{s}':a.study/f'holdout_final_{s}' for s in [56211,56217]}}}
outcomes=[];counts=[];summary=[];pairs=[];absolute=[];audit={};all_rows={}
for bank,folders in dirs.items():
    all_rows[bank]={}
    for label,folder in folders.items():
        result=read(folder/'results.json');rows=result['episodes'];g=read(folder/'geometry.json');all_rows[bank][label]=rows
        counts.append(dict(bank=bank,policy=label,successes=sum(x['success'] for x in rows),count=len(rows),physical_failures=sum(any(x['failures'].values()) for x in rows)))
        for row in rows:
            geo=g[row['geometry_index']]
            outcomes.append(dict(bank=bank,policy=label,case=row['case'],geometry=geo['label'],height_m=geo['top_center'][2],slope_deg=geo['pitch_deg'],success=row['success'],final_hold_s=row['final_hold_s'],**row['failures']))
            if row['success']:
                for w in windows:
                    for metric in metrics:
                        absolute.append(dict(bank=bank,policy=label,case=row['case'],geometry=geo['label'],height_m=geo['top_center'][2],slope_deg=geo['pitch_deg'],window=w,metric=metric,value=row[w][metric]))
        trace=np.load(folder/'trace.npz');sensor=read(folder/'sensor_audit.json');fail={x['case']:x['t'] for x in read(folder/'first_failures.json')}
        truth=[x for x in sensor['truth_validation'] if x['valid'] and x['t']<fail.get(x['case'],20.01)]
        missing=sum(int(((trace['geometry'][:,r['case'],11]==0)&(trace['t']<fail.get(r['case'],20.01))).sum()) for r in rows)
        moving_missing=sum(int(((trace['geometry'][:,r['case'],11]==0)&(trace['t']<fail.get(r['case'],20.01))&((trace['root_speed'][:,r['case']]>.02)|(trace['gyro'][:,r['case']]>.05))).sum()) for r in rows)
        invalid_samples=[dict(case=int(c),time_s=float(trace['t'][tick]),age_s=float(trace['geometry'][tick,c,12])) for tick,c in np.argwhere(trace['geometry'][:,:,11]==0) if trace['t'][tick]<fail.get(int(c),20.01)]
        audit[bank+'/'+label]=dict(sample_count=int(trace['geometry'].shape[0]*trace['geometry'].shape[1]),missing_pre_failure_samples=missing,moving_missing_pre_failure_samples=moving_missing,invalid_confidence_samples=invalid_samples,geometry_values_remain_finite=bool(np.isfinite(trace['geometry']).all()),max_offset_mm=max((abs(x['offset_error_m'])*1000 for x in truth),default=None),max_normal_deg=max((x['normal_error_deg'] for x in truth),default=None))
        if 'estimated_root_x' in trace:
            # Offline truth audit only. Runtime estimator is encoder/IMU foot-anchored.
            pelvis=result['body_names'].index('pelvis');origins=np.asarray(result['origins']);err=trace['estimated_root_x']-(trace['body'][:,:,pelvis,0]-origins[None,:,0]);valid=np.stack([trace['t']<fail.get(r['case'],20.01) for r in rows],1)
            audit[bank+'/'+label]['foot_anchored_root_x_error_mm']=dict(rms=float(np.sqrt(np.mean(err[valid]**2))*1000),maximum=float(abs(err[valid]).max()*1000))
        evidence=out/'evidence'/bank/label;evidence.mkdir(parents=True)
        for name in ['results.json','geometry.json','manifest.json','startup.npz','startup_validation.json','trace.npz','first_failures.json','sensor_audit.json','adapted_paths.npz','measured_adaptation.json','parent_identity.json']:
            if (folder/name).exists():shutil.copy2(folder/name,evidence/name)
    comparisons=[('repaired_reference','old_reference')]+[(f'repaired_parent_{s}',f'old_parent_{s}') for s in [55211,55217]]
    comparisons += [(f'final_{s}',baseline) for s in [56211,56217] for baseline in ['old_reference','repaired_reference','old_parent_55211','old_parent_55217','repaired_parent_55211','repaired_parent_55217']]
    for policy,baseline in comparisons:
        rows=all_rows[bank][policy];reference=all_rows[bank][baseline];g=read(folders[policy]/'geometry.json');common=[(x,y) for x,y in zip(rows,reference) if x['success'] and y['success']]
        for w in windows:
            for metric in metrics:
                pv=float(np.mean([x[w][metric] for x,y in common])) if common else None;bv=float(np.mean([y[w][metric] for x,y in common])) if common else None
                summary.append(dict(bank=bank,policy=policy,baseline=baseline,window=w,metric=metric,common_successes=len(common),policy_value=pv,baseline_value=bv,change_pct=100*(pv/bv-1) if bv else None))
                for x,y in common:
                    geo=g[x['geometry_index']];pv=x[w][metric];bv=y[w][metric]
                    pairs.append(dict(bank=bank,policy=policy,baseline=baseline,case=x['case'],geometry=geo['label'],height_m=geo['top_center'][2],slope_deg=geo['pitch_deg'],window=w,metric=metric,policy_value=pv,baseline_value=bv,change_pct=100*(pv/bv-1) if bv else None))
for name,rows in [('counts',counts),('case_outcomes',outcomes),('summary',summary),('paired_cases',pairs),('successful_case_metrics',absolute)]:csvfile(out/(name+'.csv'),rows)
write(out/'geometry_audit.json',audit);shutil.copy2(a.verification,out/'verification.json');shutil.copy2(a.study/'protocol.json',out/'protocol.json');shutil.copy2(a.study/'selection_frozen.json',out/'selection_frozen.json');shutil.copy2(__file__,out/'report_source.py')
for seed in [56211,56217]:
    dest=out/'training'/str(seed);dest.mkdir(parents=True)
    for name in ['verification.json','manifest.json','progress.json','parent_identity.json','initial.pt','critic_warmup.pt','final.pt','results.json']:
        shutil.copy2(a.study/f'train_{seed}'/name,dest/name)
    shutil.copy2(a.study/f'seed_{seed}.json',dest/'selection.json')
shutil.copytree(a.study/'frozen',out/'frozen');shutil.copytree(a.verification.parent,out/'independent_validation')

def metric(policy,baseline,window):return next(x for x in summary if x['bank']=='holdout' and x['policy']==policy and x['baseline']==baseline and x['window']==window and x['metric']==metrics[0])
def pct(x):return 'N/A' if x is None else f'{x:+.2f}%'
names=['old_reference','repaired_reference','old_parent_55211','repaired_parent_55211','old_parent_55217','repaired_parent_55217','final_56211','final_56217']
labels=['Old ref','Repaired ref','Old 55211','Repaired 55211','Old 55217','Repaired 55217','PPO 56211','PPO 56217']
fig,ax=plt.subplots(2,2,figsize=(14,9),layout='constrained');x=np.arange(len(names))
for i,name in enumerate(names):
    c=next(c for c in counts if c['bank']=='holdout' and c['policy']==name)
    ax[0,0].bar(i,c['successes'],color='#217a77');ax[0,0].bar(i,c['physical_failures'],bottom=c['successes'],color='#c96a50');ax[0,0].text(i,c['successes']+.2,str(c['successes']),ha='center',fontsize=8)
ax[0,0].set(title='Unseen surfaces: success (teal), physical failure (red)',xticks=x,xticklabels=labels,ylim=(0,18),ylabel='Trials out of 16');ax[0,0].tick_params(axis='x',rotation=40)
for panel,baseline in zip([ax[0,1],ax[1,0]],['repaired_reference','repaired_parent_55211']):
    for i,seed in enumerate([56211,56217]):
        for j,w in enumerate([windows[0],windows[2]]):
            m=metric(f'final_{seed}',baseline,w);val=m['change_pct']
            if val is not None:panel.bar(i+(j-.5)*.32,val,.32,color=['#217a77','#bc8935'][j],label=['Whole 20 s','Final 5 s'][j] if i==0 else None)
            else:panel.text(i+(j-.5)*.32,0,'No common successes',rotation=90,fontsize=8)
        panel.text(i,0,f"n={m['common_successes']}",ha='center',va='bottom',fontsize=8)
    panel.set(title='Learned effort vs '+baseline.replace('_',' '),xticks=[0,1],xticklabels=[56211,56217],ylabel='Squared torque change (%)');panel.axhline(0,color='#222',lw=.8);panel.legend(fontsize=9)
for i,name in enumerate(names):
    vals=[r['value'] for r in absolute if r['bank']=='holdout' and r['policy']==name and r['window']==windows[2] and r['metric']=='vertical_support_load_bw']
    if vals:ax[1,1].bar(i,100*np.mean(vals),color='#217a77')
ax[1,1].set(title='Final 5 s: body weight carried by the seat',xticks=x,xticklabels=labels,ylabel='Percent body weight (successful trials)',ylim=(0,100));ax[1,1].tick_params(axis='x',rotation=40)
fig.suptitle('Sitting repair and bounded effort PPO — reserved holdout\nEffort and load bars use successful full-duration trials; paired effort uses common successes')
fig.savefig(out/'comparison.png',dpi=165);plt.close(fig)

doc='''<!doctype html><html><meta charset="utf-8"><title>Sitting contact repair and effort PPO</title><style>body{font:17px/1.5 system-ui;max-width:1160px;margin:38px auto;padding:0 24px;color:#21343b}h1{font-size:31px}h2{font-size:24px;margin-top:35px}img{width:100%}table{border-collapse:collapse;font-size:14px}th,td{padding:8px;border-bottom:1px solid #cad5d8;text-align:left}.note{border-left:4px solid #bc8935;padding:10px 18px;background:#faf7ef}a{color:#176777}</style><h1>Sitting: contact repair, then a small effort comparison</h1>'''
doc+='<p>The contact repair and the learned PPO adjustments are assessed separately. Every trial starts standing and lasts 20 seconds. A successful trial has no physical failure and ends with five continuous seconds of stable support.</p><img src="comparison.png" alt="Stability, learned effort and seat load">'
doc+='<h2>Stability comes first</h2>'+table(['Policy','Development success /18','Unseen success /16','Unseen physical failures'],[[label,str(next(c['successes'] for c in counts if c['bank']=='development' and c['policy']==name)),str(next(c['successes'] for c in counts if c['bank']=='holdout' and c['policy']==name)),str(next(c['physical_failures'] for c in counts if c['bank']=='holdout' and c['policy']==name))] for name,label in zip(names,labels)])
doc+='<h2>Effort on common successful unseen trials</h2>'+table(['Policy','Baseline','Common trials','Whole 20 s','Final 10 s','Final 5 s'],[[str(seed),base.replace('_',' '),metric(f'final_{seed}',base,windows[0])['common_successes'],*[pct(metric(f'final_{seed}',base,w)['change_pct']) for w in windows]] for seed in [56211,56217] for base in ['old_reference','old_parent_55211','old_parent_55217','repaired_reference','repaired_parent_55211','repaired_parent_55217']])
m=metric('repaired_reference','old_reference',windows[0]);m5=metric('repaired_reference','old_reference',windows[2])
doc+=f'<p class="note"><b>The repair has an effort cost.</b> Against the old reference on {m["common_successes"]} common successful unseen trials, the repaired reference changes whole-motion squared torque by {pct(m["change_pct"])} and final-five-second torque by {pct(m5["change_pct"])}. Passing the support test does not mean relaxed sitting: its minimum support load is only 5% of body weight. Seat-load and leg-effort measurements remain essential.</p>'
doc+='<p><b>Development-only selection:</b> '+ '; '.join(str(s)+(': repaired55211 parent fallback' if v['selection'][str(s)]['parent_fallback'] else ': final checkpoint eligible') for s in [56211,56217])+'. Full-budget final checkpoints are reported above regardless of selection. Nothing has been promoted.</p>'
doc+='<h2>What was repaired and what was learned</h2><p>Depth, joint encoders and IMU determine the seat plane and a foot-anchored pelvis position. Foot-preserving inverse kinematics and ankle feedback establish weight transfer above the seat before lowering. This removed the development contact and impact failures. The original second parent still failed the friction margin in two development trials, so both new seeds start from the same validated55211 parent;55217 remains a benchmark.</p>'
doc+='<p>The frozen parent controls the approach. PPO adds two coordinated knee/hip/ankle adjustments held for 200 ms after 12 seconds. First 12-second traces match the repaired starting policy exactly. Both seeds receive eight critic-only and 32 actor-enabled updates,115,200 physical transitions each including indexed-reset warmup. The initial global 0.5-second settling interval is outside that budget. No extra training or altered physical criteria were used.</p>'
doc+='<p>Independent checks cover source/prior hashes, finite changed weights, actor identity during critic warmup, loaded checkpoints, rollout accounting, exact starts, full durations, bounded commands, fixed fingers, sensor-derived path reconstruction and development/holdout separation. Geometry audit links below include truth-based error measurement only; simulator truth does not drive runtime control.</p>'
ga=audit['holdout/repaired_reference']
doc+=f'<p><b>Tracking caveat:</b> the repaired reference has {ga["missing_pre_failure_samples"]}/{ga["sample_count"]} sampled geometry-confidence flags marked invalid before failure, of which {ga["moving_missing_pre_failure_samples"]} occur above the specified motion thresholds (root speed0.02m/s or angular speed0.05rad/s). Geometry values remain finite. Maximum measured plane-offset error while valid is {ga["max_offset_mm"]:.3f}mm. Foot-anchored pelvis error is measured using ideal simulated sensors and stationary-foot assumptions; it is not a hardware accuracy claim.</p>'
doc+='<p><b>Limits:</b> eight reserved nearby rigid-slab geometries repeated at two placements are not arbitrary furniture. Only two seeds were trained. This is reference-assisted control with known horizontal placement, not independent motion discovery. Squared torque is an effort proxy, not electrical energy or proof of a global optimum. No savings are claimed from failed fragments.</p>'
doc+='<p><a href="motion_replay.html">Motion replay</a> · <a href="summary.csv">All paired metrics</a> · <a href="paired_cases.csv">Each height and slope</a> · <a href="case_outcomes.csv">All failures and successes</a> · <a href="successful_case_metrics.csv">Effort and seat/foot loads</a> · <a href="geometry_audit.json">Geometry and pelvis tracking</a> · <a href="verification.json">Independent verification</a></p></html>'
(out/'findings.html').write_text(doc)
subprocess.run(['/home/kadir/humanoid-sitting/IsaacLab/.venv/bin/python','/home/kadir/humanoid-sitting/scripts/render_contact_repair_replay.py','--final','--output',str(out/'motion_replay.html')],check=True)
write(out/'SHA256.json',{str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file()})
print(out)
