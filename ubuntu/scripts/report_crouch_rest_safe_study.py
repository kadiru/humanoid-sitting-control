"""Export verified bridge/PPO results; no training or checkpoint selection."""
import argparse,csv,html,json,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_crouch_rest_safe_study import read,sha
from render_crouch_rest_replay import export

NAMES={'current_reference':'Current reference','current_56211':'Current PPO 56211','current_56217':'Current PPO 56217','bridge_reference':'Repaired reference','58111':'New PPO 58111','58117':'New PPO 58117'}
def copied(source,target):
    target.mkdir(parents=True,exist_ok=False)
    for name in ['results.json','geometry.json','manifest.json','startup.npz','bridge_startup.npz','trace.npz','first_failures.json','sensor_audit.json','initial.pt','parent_identity.json','action_contract.json','bridge_contract.json','bridge_sources.json','initial_settling.json','control_accounting.json','verification.json','progress.json','final.pt','critic_warmup.pt']:
        if (source/name).exists():shutil.copy2(source/name,target/name)
def pct(x):return '—' if x is None else f'{x:+.2f}%'
def table(headers,rows):
    return '<table><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</table>'
def write_csv(path,rows):
    if not rows:return
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def main():
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--verification',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    v=read(a.verification/'verification.json');assert v['passed'];protocol=read(a.study/'protocol.json');out=a.output;out.mkdir(exist_ok=False)
    shutil.copytree(a.verification,out/'verification')
    for name in ['protocol.json','protocol.sha256','holdout_barrier.json']:shutil.copy2(a.study/name,out/name)
    root=a.study.parent
    for name in ['safe_ppo_admission_01.json','safe_env_regression_verification_01.json','approach_parity_01.json','holdout_reservation_01.json']:
        shutil.copy2(root/name,out/name)
    shutil.copy2(root/'safe_env_reset_memory_02/reset_probe.json',out/'reset_memory_validation.json')
    frozen=out/'frozen_inputs';frozen.mkdir();index={}
    for i,(name,h) in enumerate(protocol['hashes'].items()):
        src=Path(name);assert sha(src)==h;dest=frozen/f'{i:03d}_{src.name}';shutil.copy2(src,dest);index[name]=dict(file=str(dest.relative_to(out)),sha256=h)
    (out/'frozen_input_index.json').write_text(json.dumps(index,indent=2))
    paths={}
    for bank in ['development','holdout']:
        for label in NAMES:
            if label.isdigit():
                done=read(a.study/f'training_finished_{label}.json');path=Path(done['final_development_evaluation']) if bank=='development' else a.study/f'holdout_policy_{label}'
            elif bank=='holdout':path=a.study/f'holdout_{label}'
            elif label=='bridge_reference':path=Path(protocol['bridge_reference'])
            else:path=Path(protocol['development_baselines'][label])
            paths[f'{bank}_{label}']=path;copied(path,out/'evidence'/f'{bank}_{label}')
    for seed in protocol['seeds']:
        folder=a.study/f'seed_{seed}';dest=out/f'training_{seed}';dest.mkdir();shutil.copy2(folder/'curriculum.json',dest/'curriculum.json');shutil.copy2(a.study/f'training_finished_{seed}.json',dest/'training_finished.json')
        for record in read(folder/'curriculum.json'):
            src=Path(record['checkpoint']).parent;copied(src,dest/src.name)
            src=Path(record['evaluation']);copied(src,dest/src.name)
            if record['earlier_gate'] is not None:
                candidates=list(folder.glob(f"advance_{record['block']:02d}_control*s"));assert len(candidates)==1;copied(candidates[0],dest/candidates[0].name)
    flat=[]
    for category in ['comparisons','reference_repair_comparisons']:
        for label,baselines in v[category].items():
            for baseline,c in baselines.items():
                for group,windows in c['grouped'].items():
                    for window,metrics in windows.items():
                        for metric,stats in metrics.items():flat.append(dict(category=category,policy_label=label,baseline_label=baseline,group=group,window=window,metric=metric,**stats))
    write_csv(out/'paired_metrics.csv',flat)
    case_rows=[]
    for label,path in paths.items():
        data=read(path/'results.json');geom=read(path/'geometry.json')
        for e in data['episodes']:
            g=geom[e['geometry_index']];case_rows.append(dict(evaluation=label,case=e['case'],height_m=g['height'],slope_deg=g['pitch_deg'],stable=e['success'],strict=e.get('strict_transfer'),failures=';'.join(k for k,b in e['failures'].items() if b),slip_ticks=e.get('slip_ticks'),traction_bad_ticks=e.get('traction_bad_ticks'),illegal_ticks=e.get('illegal_ticks'),last5_torque_squared=e['metrics_last5s']['total_squared_torque_Nm2'],last5_seat_load_bw=e['metrics_last5s']['vertical_support_load_bw']))
    write_csv(out/'trial_results.csv',case_rows)
    labels=list(NAMES);x=np.arange(len(labels));colors=['#83949b']*3+['#c89555','#267a78','#67a7a0']
    fig,axes=plt.subplots(2,2,figsize=(14,10),layout='constrained')
    for ax,bank,n in zip(axes[0],['development','holdout'],[18,16]):
        stable=[v['evaluations'][f'{bank}_{s}']['successes'] for s in labels];strict=[v['evaluations'][f'{bank}_{s}'].get('strict_successes',np.nan) for s in labels];physical=[v['evaluations'][f'{bank}_{s}']['physical_failures'] for s in labels]
        ax.bar(x-.17,stable,.34,color=colors,label='Stable final hold');ax.bar(x+.17,strict,.34,color='#254d65',label='Clean whole transfer');ax.set_ylim(0,n+3);ax.set_xticks(x,[NAMES[s].replace(' ','\n',1) for s in labels],rotation=20,ha='right');ax.set_title(f'{bank.title()}: all {n} trials');ax.set_ylabel('Successful trials');ax.legend(fontsize=8)
        for xx,ss,pp in zip(x,stable,physical):ax.text(xx,ss+.3,f'{pp} physical',ha='center',fontsize=8)
    for ax,window in zip(axes[1],['metrics_whole_trial','metrics_last5s']):
        for j,seed in enumerate(protocol['seeds']):
            vals=[];counts=[]
            for base in labels[:4]:
                m=v['comparisons'][f'holdout_{seed}'][base]['grouped']['all'][window]['total_squared_torque_Nm2'];vals.append(m['change_percent'] if m['change_percent'] is not None else np.nan);counts.append(m['common_successes'])
            xx=np.arange(4)+(j-.5)*.34;ax.bar(xx,vals,.34,label=str(seed),color=colors[4+j])
            for px,yy,count in zip(xx,vals,counts):
                if np.isfinite(yy):ax.annotate(f'n={count}',(px,yy),xytext=(0,4 if yy>=0 else -12),textcoords='offset points',ha='center',fontsize=8)
        ax.axhline(0,color='#777',lw=.8);ax.set_xticks(np.arange(4),[NAMES[s] for s in labels[:4]],rotation=20,ha='right');ax.set_title('Holdout torque²: '+('whole 20 s' if window=='metrics_whole_trial' else 'final 5 s'));ax.set_ylabel('Change on common successes (%)');ax.legend()
    for ax in axes.flat:ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('Stability first; effort comparisons exclude failed trials',fontsize=15);fig.savefig(out/'comparison.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4.5),layout='constrained')
    for seed,t in v['training'].items():
        xx=[];yy=[];total=0
        for block in t['blocks']:xx.extend([total,total+block['updates']]);yy.extend([block['learning_start_s']]*2);total+=block['updates']
        ax.plot(xx,yy,label=seed,marker='.',alpha=.8)
    ax.set_yticks([10,11,13,15]);ax.set_ylim(9.5,15.5);ax.set_xlabel('Scheduled rollout updates');ax.set_ylabel('Learned control begins (seconds)');ax.set_title('Earlier control requires a separate clean-transfer validation');ax.grid(alpha=.2);ax.legend();fig.savefig(out/'curriculum.png',dpi=150);plt.close(fig)
    summary=[];domain=[];effort=[];budget=[];sentences=[];gates=[]
    for label in labels:
        e=v['evaluations']['holdout_'+label];summary.append([NAMES[label],f"{e['successes']}/16",e.get('strict_successes','not recorded'),e['physical_failures']])
        d=v['evaluations']['development_'+label];domain.append([NAMES[label],f"{d['admitted']['stable']}/12",d['admitted']['strict'],d['admitted']['physical'],f"{d['excluded']['stable']}/6",d['excluded']['physical']])
    for seed,t in v['training'].items():
        e=v['evaluations']['holdout_'+seed];b=v['evaluations']['holdout_bridge_reference'];c=v['comparisons']['holdout_'+seed]['bridge_reference']['grouped']['all']
        whole=c['metrics_whole_trial']['total_squared_torque_Nm2'];late=c['metrics_last5s']['total_squared_torque_Nm2']
        sentences.append(f"Seed {seed}: {e['successes']}/16 stable and {e.get('strict_successes',0)}/16 clean holdout trials, with {e['physical_failures']} physical failures. Repaired reference: {b['successes']}/16 stable and {b.get('strict_successes',0)}/16 clean. On {whole['common_successes']} common successes, PPO changes whole-motion torque² by {pct(whole['change_percent'])} and resting torque² by {pct(late['change_percent'])}. Final learned control begins at {t['final_control_start_s']} s.")
        budget.append([seed,t['updates'],t['actual_actor_Adam_steps'],t['actual_critic_Adam_steps'],t['transitions'],t['control_ticks'],t['teacher_ticks'],t['excluded_ticks'],t['warmup']])
        for block in t['blocks']:
            g=block['gate'];early=block['earlier_gate'];gates.append([seed,block['block']+1,block['learning_start_s'],f"{g['included_strict']}/12",'not attempted' if early is None else f"{early['included_strict']}/12",block['next_start_s']])
        for baseline in labels[:4]:
            c=v['comparisons'][f'holdout_{seed}'][baseline]['grouped']['all'];stats=[c[w]['total_squared_torque_Nm2'] for w in ['metrics_whole_trial','metrics_last10s','metrics_last5s']];effort.append([seed,NAMES[baseline],stats[0]['common_successes']]+[pct(s['change_percent']) for s in stats])
    geometry={k:e['geometry_before_failure'] for k,e in v['evaluations'].items()};(out/'geometry_audit.json').write_text(json.dumps(geometry,indent=2))
    (out/'conclusions.txt').write_text('\n\n'.join(sentences)+'\n\nNo policy is promoted. All standing approaches remain reference-assisted.\n')
    style='body{font:17px/1.55 system-ui;max-width:1200px;margin:35px auto;padding:0 24px;color:#19363b}h1{line-height:1.2}img{width:100%}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:8px;border-bottom:1px solid #cad6d8;text-align:left}.note{padding:18px;background:#fff4d6;border-left:5px solid #c79936}a{color:#176f77}.scroll{overflow:auto}'
    doc=f'''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Crouch-to-rest PPO — verified comparison</title><style>{style}</style><h1>Crouch-to-rest PPO: stability and whole-body effort</h1><p class="note">The physical repair was admitted on six of nine development surfaces. Three remained stress tests. Every episode starts standing, but the first ten seconds always use the reference controller. This is reference-assisted residual learning, not unassisted discovery of sitting.</p>{''.join('<p>'+html.escape(s)+'</p>' for s in sentences)}<p><a href="motion_replay.html">Matched motion replay</a> · <a href="paired_metrics.csv">All paired metrics</a> · <a href="trial_results.csv">Every trial</a> · <a href="verification/verification.json">Independent verification</a></p><h2>Reserved surfaces</h2>{table(['Controller','Stable','Clean whole transfer','Physical failures'],summary)}<p>Stable requires a full 20-second trial without physical failure and continuous qualifying support in the final five seconds. Clean additionally requires zero slipping, friction-margin or illegal-contact violations at every active tick. Older baselines lack the complete clean-transfer summary; missing values are not zero.</p><img src="comparison.png" alt="Success, physical failures and paired torque changes"><h2>Development domain and unresolved cases</h2>{table(['Controller','Admitted stable','Admitted clean','Admitted physical','Stress stable','Stress physical'],domain)}<p>The excluded cases are 40 cm/−5°, 42 cm/−5° and 42 cm/0°, each at two placements. They are excluded from optimization, not hidden from final evaluation. The policy receives no geometry-admission selector during evaluation.</p><h2>Actual progression</h2><img src="curriculum.png" alt="Actual learned-control start times">{table(['Seed','Block','Control begins','Current window clean','Earlier window clean','Next start'],gates)}<p>Advancement requires all twelve admitted trials to pass a separate evaluation at the proposed earlier time. No failed gate is bypassed. The final full-budget checkpoint is evaluated, without policy promotion or best-checkpoint substitution.</p><h2>Effort on common full-duration successes</h2>{table(['Seed','Comparator','Common successes','Whole 20 s','Final 10 s','Final 5 s'],effort)}<p>Negative torque² changes mean lower effort. Comparisons against the repaired reference isolate incremental learning; comparisons against older baselines include controller-repair gains. A reduction among surviving trials does not offset lost reliability. The CSV also includes strict-policy subsets, height/slope groups, absolute torque, time-averaged instantaneous joint RMS Nm, leg/upper-body contributions and seat load. Failed fragments never contribute to savings claims.</p><h2>Budget and verification</h2><div class="scroll">{table(['Seed','Rollouts','Actor Adam','Critic Adam','Physics ticks','Learned ticks','Teacher ticks','Excluded ticks','Warmup ticks'],budget)}</div><p>Each seed has 64 scheduled rollouts, eight critic-only and 56 actor-enabled. Rollouts without eligible samples take no optimizer step. Main budget: 368,640 physics transitions, including indexed-reset warmup and teacher/excluded rows. Initial settling adds 3,600 excluded ticks; optimizer smoke used 17,280 separate transitions. Both final checkpoints and control timings froze before holdout evaluation.</p><p>Verified {v['source_hashes']} frozen inputs and {v['prior_hashes']} protected prior files, finite changed weights, optimizer continuity, exact loaded checkpoints and startups, teacher-prefix identity, complete horizons, fixed fingers and command limits. <a href="geometry_audit.json">Geometry availability and error before failure</a> remain separate from physical outcome. Inherited action_contract.json describes the older base harness; bridge_contract.json and frozen bridge sources specify the effective five-second motion and residual spans.</p><h2>Limits</h2><p>Runtime uses 406 sensor/reference channels, depth geometry, joint kinematics, IMU and the static robot model; simulator truth supplies reset/reward/evaluation labels only. Explicit horizontal placement and the motion reference remain assistance. The reserved eight nearby rigid surfaces, each repeated twice, were withheld from this pilot, not necessarily from the entire project. Two seeds do not establish arbitrary-surface generalization. Total torque over 37 joints is an effort proxy, not measured electrical energy or proof of a global optimum. Leaning is unchanged by this sitting-only pilot.</p>'''
    (out/'findings.html').write_text(doc);export(a.study,out/'motion_replay.html');shutil.copy2(__file__,out/'report_source.py')
    (out/'SHA256.json').write_text(json.dumps({str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()},indent=2));print('\n'.join(sentences))
if __name__=='__main__':main()
