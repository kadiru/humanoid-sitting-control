"""Physical outcomes first, paired effort second, every episode retained in video."""
import argparse,csv,html,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from earlier_control_common import *
from render_earlier_control_video import render,name,label
from verify_earlier_control_video import verify as verify_video

def table(headers,rows):
    esc=lambda x:html.escape(str(x))
    return '<table><thead><tr>'+''.join('<th>'+esc(x)+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+esc(x)+'</td>' for x in r)+'</tr>' for r in rows)+'</tbody></table>'
def pct(x):return '—' if x is None else f'{x:+.2f}%'
def main():
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False)
    v=read(a.study/'verified_02/verification.json');assert v['passed'];out=a.output
    shutil.copy2(a.study/'verified_02/verification.json',out/'verification.json');shutil.copy2(a.study/'protocol.json',out/'protocol.json');shutil.copy2(ROOT/'docs/EARLIER_CONTROL_PROTOCOL.md',out/'protocol.md')
    records=[];outcomes=[];efforts=[]
    for bank,items in v['paths'].items():
        for controller,path in items.items():
            path=Path(path);dest=out/'results'/bank/controller;dest.mkdir(parents=True)
            for file in ['results.json','geometry.json','first_failures.json','sensor_audit.json','manifest.json','early_contract.json','episode_contract.json']:
                if (path/file).exists():shutil.copy2(path/file,dest/file)
            rows=read(path/'results.json')['episodes'];geoms=read(path/'geometry.json')
            for r in rows:
                g=geoms[r['geometry_index']];status,reasons=label(r)
                item=dict(bank=bank,controller=controller,case=r['case'],height_cm=g['height']*100,slope_deg=g['pitch_deg'],outcome=status,stable=r['success'],physical=any(r['failures'].values()),slip_ticks=r['slip_ticks'],illegal_ticks=r['illegal_ticks'],margin_ticks=r['traction_bad_ticks'],reason='; '.join(reasons))
                for window in ['metrics_whole_trial','metrics_last10s','metrics_last5s']:
                    for key,value in r[window].items():item[window+'_'+key]=value
                records.append(item)
            outcomes.append([bank,name(controller),f"{sum(r['success'] for r in rows)}/{len(rows)}",sum(any(r['failures'].values()) for r in rows),sum(r['slip_ticks']>0 for r in rows),sum(r['illegal_ticks']>0 for r in rows),sum(r['traction_bad_ticks']>0 for r in rows)])
            if controller.startswith(('final_','selected_')):
                for baseline,c in v['comparisons'][bank+'_'+controller].items():
                    group=c['grouped']['all'];metric='total_squared_torque_Nm2'
                    efforts.append([bank,name(controller),name(baseline),len(c['common_success_cases'])]+[pct(group[w][metric]['change_percent']) for w in ['metrics_whole_trial','metrics_last10s','metrics_last5s']])
    with (out/'episodes.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    curriculum=[];selections=[]
    for s in SEEDS:
        dest=out/f'seed_{s}';dest.mkdir();folder=a.study/f'seed_{s}'
        for file in ['curriculum.json','selection.json']:shutil.copy2(folder/file,dest/file)
        shutil.copy2(a.study/f'training_finished_{s}.json',dest/'final_identity.json')
        selected=read(folder/'selection.json');final=read(a.study/f'training_finished_{s}.json')
        selections.append([s,final['start_s'],selected['start_s'],'Current parent fallback' if selected.get('parent_fallback') else f"Block {selected['block']+1} checkpoint",'No eligible new candidate' if selected.get('parent_fallback') else 'Passed development stability, margin and effort gates'])
        for r in read(folder/'curriculum.json'):
            curriculum.append([s,r['block']+1,r['trained_start_s'],r['current_gate']['passed'],r.get('proposed_start_s','—'),r.get('earlier_gate',{}).get('passed','—'),'; '.join(r['current_gate']['reasons']+r.get('earlier_gate',{}).get('reasons',[])) or 'passed'])
    fig,axes=plt.subplots(3,2,figsize=(14,13),constrained_layout=True)
    for bi,bank in enumerate(['development','retention','wide']):
        names=list(v['paths'][bank]);x=np.arange(len(names));counts=[v['evaluations'][bank+'_'+n]['successes'] for n in names];total=[v['evaluations'][bank+'_'+n]['trials'] for n in names]
        axes[bi,0].bar(x,counts,color=['#788f9c' if not n.startswith(('final','selected')) else '#257c68' for n in names]);axes[bi,0].set_ylim(0,max(total)+2);axes[bi,0].set_ylabel('Stable full-duration successes');axes[bi,0].set_title(bank+' — stability first')
        axes[bi,0].set_xticks(x,[name(n).replace(' PPO ','\n') for n in names],rotation=25,ha='right')
        for i,(n,d) in enumerate(zip(counts,total)):axes[bi,0].text(i,n+.2,f'{n}/{d}',ha='center')
        learned=[n for n in names if n.startswith(('final_','selected_'))];xx=np.arange(len(learned));width=.25
        for j,baseline in enumerate(['reference','parent_59111','parent_59117']):
            c=[v['comparisons'][bank+'_'+n][baseline]['grouped']['all']['metrics_last5s']['total_squared_torque_Nm2'] for n in learned];vals=[z['change_percent'] if z['change_percent'] is not None else np.nan for z in c]
            bars=axes[bi,1].bar(xx+(j-1)*width,vals,width,label=name(baseline))
            for bar,z in zip(bars,c):
                if z['change_percent'] is not None:axes[bi,1].text(bar.get_x()+bar.get_width()/2,bar.get_height(),f"n={z['common_successes']}",ha='center',va='bottom',fontsize=8)
        axes[bi,1].axhline(0,color='black',lw=.8);axes[bi,1].set_xticks(xx,[name(n) for n in learned],rotation=25,ha='right');axes[bi,1].set_ylabel('Final5s torque² change (%)');axes[bi,1].set_title(bank+' — common full-duration successes');axes[bi,1].legend(fontsize=8)
    fig.suptitle('Earlier PPO control: stability and paired resting effort\nNegative effort change is lower; failed fragments excluded',fontsize=16);fig.savefig(out/'comparison.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4),constrained_layout=True)
    for s in SEEDS:
        rr=read(a.study/f'seed_{s}/curriculum.json');ax.plot([6*(i+1) for i in range(4)],[r['trained_start_s'] for r in rr],marker='o',label=str(s))
    ax.set(xlabel='Completed PPO updates',ylabel='Early policy starts (seconds)',title='Actual trained takeover times — failed gates never bypassed',xticks=[6,12,18,24],yticks=[6,8,9,10,11]);ax.legend();fig.savefig(out/'curriculum.png',dpi=150);plt.close(fig)
    render(v['paths'],out/'video');video=verify_video(out/'video',v['paths']);write(out/'video/verification.json',video)
    caveats='<p>Every trial starts standing, but the repaired feedback reference remains active. A new bounded early action precedes the frozen resting policy at11s. This is reference-assisted control. Retention cases were previously tested; the wider bank was withheld from all new training and selection. Two seeds and repeated rigid surfaces do not establish arbitrary-surface generalization. Torque is an effort proxy, not electrical energy.</p>'
    body='<h1>Earlier PPO sitting control</h1><p>Independent budget, checkpoint, optimizer, startup and source checks passed. Full-budget finals and any distinct development-selected policies are shown separately. No policy automatically promoted.</p>'+caveats
    body+='<p>Pre-main setup revision: the initial0.004rad bound was rejected, and0.001rad retained. Exploration admission was explicitly revised to permit one20ms conservative grip-margin warning below modeled friction capacity, with zero physical failures/slips/unwanted contacts. Reporting still uses the original0.90 warning threshold. Advancement and selection still require no added per-case warning ticks and retained effort savings. Failed preflights and original definitions are preserved; this was an adaptive setup decision before main training and holdout testing.</p>'
    body+='<h2>Every final test episode in under one minute</h2><p>Recorded-state schematic. Green SUCCESS requires stable support with no slipping or unwanted contact. Red FAILURE includes unsupported or physically failed episodes. Amber conservative grip-margin warnings remain separate. Training exploration is excluded from this final-comparison video.</p><video controls preload="metadata" src="video/summary.mp4"></video>'
    body+=f'<p>{video["episodes"]} episodes across {video["conditions"]} conditions; {video["seconds"]:.2f}s. Coverage audit passed. <a href="video/coverage.json">Coverage manifest</a></p>'
    body+='<h2>Stability and physical failures</h2>'+table(['Bank','Controller','Stable','Physical failure','Slip cases','Unwanted contact cases','Margin-flag cases'],outcomes)
    body+='<h2>Paired total effort</h2><p>Torque² percent change; negative means less effort. Only common full-duration stable successes with zero all-tick slipping or unwanted contacts in both controllers are compared. See verification.json for absolute torque, mean instantaneous all-joint RMS Nm, leg/upper-body components, strict-clean subsets and height/slope groups.</p>'+table(['Bank','Policy','Baseline','Common successes','Whole20s','Final10s','Final5s'],efforts)
    body+='<img src="comparison.png"><h2>Curriculum and selection</h2><img src="curriculum.png">'+table(['Seed','Block','Trained start s','Current gate','Proposed start s','Earlier gate','Reasons'],curriculum)
    body+=table(['Seed','Full-budget trained start s','Selected validated start s','Selection','Reason'],selections)
    body+='<p><a href="episodes.csv">Every episode and metric</a> · <a href="verification.json">Full audit and paired comparisons</a> · <a href="protocol.md">Frozen protocol</a></p>'
    page='<!doctype html><html><head><meta charset="utf-8"><title>Earlier PPO sitting comparison</title><style>body{font:16px system-ui;margin:36px auto;max-width:1450px;color:#243b49;padding:0 24px}table{border-collapse:collapse;width:100%;font-size:14px;margin:24px 0}th,td{padding:9px;border-bottom:1px solid #d4e0e5;text-align:left}th{background:#eaf1f4}img,video{width:100%;border-radius:8px}h2{margin-top:42px}p{line-height:1.6}</style></head><body>'+body+'</body></html>'
    (out/'findings.html').write_text(page);(out/'review.html').write_text(page)
    # Full video is the saved-state replay; no second selected-only animation.
    (out/'motion_replay.html').write_text('<!doctype html><meta charset="utf-8"><title>All episodes replay</title><style>body{font:20px system-ui;background:#edf3f5;margin:24px}video{width:100%}</style><h1>All final comparison episodes</h1><video controls src="video/summary.mp4"></video><p><a href="findings.html">Outcome report and full coverage audit</a></p>')
    hashes={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()};write(out/'SHA256.json',hashes);print(dict(export=str(out),video=video))
if __name__=='__main__':main()
