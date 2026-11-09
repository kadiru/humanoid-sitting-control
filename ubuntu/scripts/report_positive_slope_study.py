"""Report full comparisons and all-episode video without promoting any policy."""
import argparse,shutil,csv,html
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from positive_slope_common_v2 import *
from render_positive_slope_video_v3 import render,label
from verify_positive_slope_video import verify as verify_video

def title(k):
    names={'reference':'Previous reference','repaired_reference':'Repaired reference','repaired_parent':'Frozen59117 / repaired handoff','current_60117':'Current early PPO60117','parent_59111':'Current PPO59111','parent_59117':'Current PPO59117'}
    return names.get(k,k.replace('final_','Final PPO').replace('selected_','Selected PPO'))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--study',type=Path,required=True);ap.add_argument('--verification',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    v=read(a.verification);assert v['passed'];out=a.output;out.mkdir(exist_ok=False)
    shutil.copy2(a.verification,out/'verification.json');shutil.copy2(a.study/'protocol.json',out/'protocol.json')
    records=[]
    for bank,paths in v['paths'].items():
        for name,folder in paths.items():
            p=Path(folder);rows=read(p/'results.json')['episodes'];g=read(p/'geometry.json');dest=out/'results'/bank/name;dest.mkdir(parents=True)
            for f in ['results.json','geometry.json','manifest.json','first_failures.json','sensor_audit.json','early_contract.json','episode_contract.json','rolling_plan_audit.json','control_accounting.json','material_audit.json']:
                if (p/f).exists():shutil.copy2(p/f,dest/f)
            for x in rows:
                gg=g[x['geometry_index']];status,reasons=label(x)
                records.append(dict(bank=bank,controller=name,case=x['case'],height_cm=100*gg['height'],slope_deg=gg['pitch_deg'],outcome=status,stable=x['success'],physical=any(x['failures'].values()),slip_ticks=x['slip_ticks'],illegal_ticks=x['illegal_ticks'],margin_ticks=x['traction_bad_ticks'],reason='; '.join(reasons)))
    for seed in SEEDS:
        p=out/f'training_{seed}';p.mkdir()
        for f in ['curriculum.json','selection.json']:shutil.copy2(a.study/f'seed_{seed}'/f,p/f)
        shutil.copy2(a.study/f'training_finished_{seed}.json',p/'final_checkpoint.json')
        for block in range(4):
            target=p/f'block_{block:02d}';target.mkdir()
            for f in ['early_contract.json','learning_audit.json','control_accounting.json','progress.json']:shutil.copy2(a.study/f'seed_{seed}/block_{block:02d}/train'/f,target/f)
    with (out/'episodes.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    banks=list(v['paths']);fig,axes=plt.subplots(len(banks),2,figsize=(16,6*len(banks)),constrained_layout=True)
    for bi,bank in enumerate(banks):
        keys=list(v['paths'][bank]);vals=[v['evaluations'][bank+'_'+k]['clean'] for k in keys];ax=axes[bi,0]
        ax.bar(range(len(keys)),vals,color=['#52768a' if not k.startswith('final') else '#21896d' for k in keys]);ax.set(ylim=(0,26),ylabel='Full20s stable / no slip or unwanted contact',title=bank+' stability (24 trials)');ax.set_xticks(range(len(keys)),[title(k) for k in keys],rotation=35,ha='right')
        for i,x in enumerate(vals):ax.text(i,x+.25,str(x),ha='center')
        names=[k for k in keys if k.startswith(('final_','selected_'))];ax=axes[bi,1];positions=np.arange(len(names));width=.22
        for j,base in enumerate(['repaired_reference','repaired_parent','parent_59117']):
            stats=[v['comparisons'][bank+'_'+k][base]['grouped']['all']['metrics_last5s']['total_squared_torque_Nm2'] for k in names]
            values=[x['change_percent'] if x['change_percent'] is not None else np.nan for x in stats]
            ax.bar(positions+(j-1)*width,values,width,label=title(base))
            for x,y,z in zip(positions+(j-1)*width,values,stats):
                if np.isfinite(y):ax.text(x,y,f"n={z['common_successes']}",fontsize=8,ha='center',va='bottom')
        ax.axhline(0,color='black',lw=.7);ax.set_xticks(positions,[title(k) for k in names],rotation=25,ha='right');ax.set(ylabel='Final5s all-joint torque² change (%)',title='Resting effort: common clean trials only');ax.legend(fontsize=9)
    fig.suptitle('Positive slopes: physical repair and incremental PPO are separate comparisons');fig.savefig(out/'comparison.png',dpi=135);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4),constrained_layout=True)
    for seed in SEEDS:
        r=read(a.study/f'seed_{seed}/curriculum.json');ax.step([1,4,7,10],[x['trained_start_s'] for x in r],where='post',label=str(seed));ax.scatter([3,6,9,12],[x['trained_start_s'] for x in r])
    ax.set(xlabel='PPO update',ylabel='Learned intervention begins (s)',yticks=[6,8,9],title='Actual trained takeover time; earlier times require passing gates');ax.legend();fig.savefig(out/'curriculum.png',dpi=150);plt.close(fig)
    render(v['paths'],out/'video');vv=verify_video(out/'video',v['paths']);write(out/'video/verification.json',vv)
    content='<h1>Positive-slope sitting: repaired transfer and earlier PPO</h1><p>Positive slope means the surface descends toward the robot’s feet. Negative slopes were excluded. This is bounded reference-assisted control, not discovery of an unassisted sitting motion.</p>'
    content+='<video controls preload="metadata" src="video/summary.mp4"></video>'+f'<p>Every final comparison episode is included: {vv["episodes"]} episodes, {vv["seconds"]:.2f} seconds, H264. Green requires stable support without any slipping or unwanted contacts. Failed trials remain red; conservative margin warnings are separate amber labels. Offline schematic from recorded body transforms and model COM.</p>'
    for bank,pp in v['paths'].items():
        content+='<h2>'+html.escape(bank)+'</h2><table><tr><th>Controller</th><th>Clean</th><th>Stable</th><th>Physical failure</th><th>Slip</th><th>Unwanted contact</th><th>Strict margin-clean</th></tr>'
        for k in pp:
            r=dict(v['evaluations'][bank+'_'+k]);rr=read(Path(pp[k])/'results.json')['episodes'];r['unwanted']=sum(bool(x['illegal_ticks'] or x['failures']['forbidden']) for x in rr);content+='<tr><td>'+html.escape(title(k))+'</td>'+''.join('<td>'+str(r[z])+'</td>' for z in ['clean','stable','physical','slip','unwanted','strict'])+'</tr>'
        content+='</table>'
    content+='<img src="comparison.png"><img src="curriculum.png"><h2>What was actually learned</h2>'
    for seed in SEEDS:
        t=v['training'][str(seed)];s=v['selections'][str(seed)];content+='<p>'+html.escape(f'Seed{seed}: trained updates by takeover time {t["stage_updates"]}; full-budget final begins at {t["final_start_s"]}s. Selection: '+('explicit frozen-parent fallback, with repaired handoff' if s.get('parent_fallback') else str(s['start_s'])+'s, block'+str(s['block'])))+'</p>'
    content+='<p>Training used nine development geometries, each repeated twice. Three known difficult geometries (36cm/+15°,36cm/+20°,40cm/+20°) stayed excluded from optimization and selection and remain in the 24-trial development stress comparison. The 24 withheld trials use 12 separately reserved nearby geometries, each repeated twice, and were not simulated until both finals and selections were fixed.</p><p>The original 0.90 margin warning and physical slip/contact definitions are unchanged. The legacy force ratio divides by0.6; this is not the actual static friction limit. Simulator material tensors report robot static/dynamic0.5/0.5 and seat/floor0.8/0.6. Admission uses observed no-slip/no-contact outcomes, while margin warnings remain reported and penalized.</p><p>Compare new PPO to the repaired reference AND frozen59117 on the same repaired handoff to isolate learning from controller repair. All effort comparisons require BOTH controllers to finish the same20-second trial cleanly; no savings are claimed from failed fragments. Torque², absolute torque, instantaneous all-joint RMS Nm, leg/upper-body contributions, height/slope groups and whole20/final10/final5 windows are in verification.json. Torque is a proxy, not measured energy or a global optimum. Two seeds and finite rigid slabs do not demonstrate arbitrary-surface generalization. No policy has been promoted.</p><p>Source/prior preservation, optimizer CPU replay, exact checkpoint loads, budgets, matched physical and414-channel starts, bounded actions, fixed fingers, full horizons and holdout timing are independently verified.</p><p><a href="verification.json">All checks and paired metrics</a> · <a href="episodes.csv">Every episode</a> · <a href="video/coverage.json">Video coverage</a></p>'
    (out/'findings.html').write_text('<!doctype html><meta charset="utf-8"><title>Positive-slope PPO comparison</title><style>body{font:16px system-ui;max-width:1450px;margin:32px auto;padding:0 24px;color:#243b49}p{line-height:1.6}video,img{width:100%}table{border-collapse:collapse;width:100%;margin:24px 0}th,td{padding:9px;border-bottom:1px solid #ccc;text-align:left}</style>'+content)
    write(out/'SHA256.json',{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()});print(vv)
if __name__=='__main__':main()
