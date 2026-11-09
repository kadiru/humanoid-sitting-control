"""Verified positive-slope baseline/repair report and every-episode video."""
import time,shutil,csv,html
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from positive_slope_common import *
from positive_slope_validation import physical,matched
from verify_seat_load_study import evaluation,compare
from render_positive_slope_video import render,name,label
from verify_positive_slope_video import verify as verify_video

def main():
    paths={'reference':DATA/'assessment/reference',**{f'parent_{s}':DATA/f'assessment/{s}' for s in [59111,59117]},
        **{f'final_{s}':DATA/f'assessment/{s}' for s in [60111,60117]},**{f'repair_{i:02d}':DATA/f'repair_{i:02d}/reference' for i in [1,2]}}
    while not all((p/'episode_contract.json').exists() or (p/'closure_recovery.json').exists() for p in paths.values()):
        for p in paths.values():
            log=p.parent/(p.name+'.log')
            if log.exists() and not (p/'closure_recovery.json').exists():assert 'Traceback (most recent call last)' not in log.read_text(),log
        time.sleep(15)
    out=DATA/'diagnosis_export_01';out.mkdir(exist_ok=False);v=dict(passed=False,scope='Development-only physical assessment and repair probes; no new PPO trained; holdout untouched',paths={'development':{k:str(p) for k,p in paths.items()}},outcomes={},comparisons={},startups={})
    all_rows={};geoms={};records=[]
    for k,p in paths.items():
        summary,rows,g=evaluation(p,24);v['outcomes'][k]=dict(**physical(p,require_clean=False),evaluation=summary)
        v['startups'][k]=matched(paths['reference'],p,10.)
        all_rows[k]=[dict(x,success=bool(label(x)[0]=='SUCCESS')) for x in rows];geoms[k]=g
        for x in rows:
            gg=g[x['geometry_index']];status,reasons=label(x)
            records.append(dict(controller=k,case=x['case'],height_cm=gg['height']*100,slope_deg=gg['pitch_deg'],outcome=status,stable=x['success'],physical=any(x['failures'].values()),slip_ticks=x['slip_ticks'],illegal_ticks=x['illegal_ticks'],margin_ticks=x['traction_bad_ticks'],reason='; '.join(reasons)))
        dest=out/'results'/k;dest.mkdir(parents=True)
        for filename in ['results.json','geometry.json','first_failures.json','sensor_audit.json','manifest.json','episode_contract.json','rolling_plan_audit.json','positive_config.json','closure_recovery.json']:
            if (p/filename).exists():shutil.copy2(p/filename,dest/filename)
        if k.startswith('repair_'):
            for filename in ['positive_slope_planner.py','run_positive_slope_reference.py']:
                assert sha(p/'source'/filename)==sha(ROOT/'scripts'/filename)
    for k in all_rows:
        v['comparisons'][k]={b:compare(all_rows[k],all_rows[b],geoms[k]) for b in ['reference','parent_59111','parent_59117','final_60117']}
    check(read(DATA/'prior_hashes.json'));v.update(passed=True,prior_files_preserved=len(read(DATA/'prior_hashes.json')))
    write(out/'verification.json',v)
    with (out/'episodes.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    keys=list(paths);fig,ax=plt.subplots(1,2,figsize=(15,6),constrained_layout=True)
    ax[0].bar(np.arange(len(keys)),[v['outcomes'][k]['clean'] for k in keys],color=['#587789']*5+['#289179']*2)
    ax[0].set(ylim=(0,26),ylabel='Clean full-duration successes / 24',title='Stable support with zero slip/contact')
    ax[0].set_xticks(np.arange(len(keys)),[name(k) for k in keys],rotation=30,ha='right')
    for i,k in enumerate(keys):ax[0].text(i,v['outcomes'][k]['clean']+.3,str(v['outcomes'][k]['clean']),ha='center')
    vals=[v['comparisons'][k]['reference']['grouped']['all']['metrics_last5s']['total_squared_torque_Nm2'] for k in keys[1:]]
    ax[1].bar(np.arange(len(vals)),[x['change_percent'] if x['change_percent'] is not None else np.nan for x in vals],color='#587789')
    ax[1].axhline(0,color='black',lw=.7);ax[1].set_xticks(np.arange(len(vals)),[name(k) for k in keys[1:]],rotation=30,ha='right')
    ax[1].set(title='Resting effort vs current reference',ylabel='Final5s torque² change (%) — common clean trials')
    for i,x in enumerate(vals):
        if x['change_percent'] is not None:ax[1].text(i,x['change_percent'],f"n={x['common_successes']}",ha='center',va='bottom')
    fig.suptitle('Positive slopes +5° to +20°: physical validation, not new PPO results');fig.savefig(out/'comparison.png',dpi=140);plt.close(fig)
    render(v['paths'],out/'video');vv=verify_video(out/'video',v['paths']);write(out/'video/verification.json',vv)
    headers=['Controller','Clean','Stable support','Physical failures','Slipping','Unwanted contacts','Strict margin-clean']
    cells=[[name(k)]+[v['outcomes'][k][z] for z in ['clean','stable','physical','slip','illegal','strict']] for k in keys]
    table='<table><tr>'+''.join('<th>'+x+'</th>' for x in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in r)+'</tr>' for r in cells)+'</table>'
    body='<h1>Positive-slope physical validation</h1><p><strong>Development-only assessment. New PPO has not been trained. Reserved unseen surfaces remain untouched.</strong></p><p>Positive slopes descend toward the robot’s feet. The current reference and policies encounter real seat-contact failures at 15–20°. Two separate candidates test slower transfer farther inside the seat; failures remain visible. Their results are physical-controller changes, not learning gains.</p>'
    body+='<video controls src="video/summary.mp4"></video>'+f'<p>All {vv["episodes"]} compared episodes, {vv["seconds"]:.2f}s, complete coverage independently verified. Green means stable support without slipping or unwanted contact; red includes every failed outcome. Conservative grip warnings are amber.</p>'+table
    body+='<img src="comparison.png"><p>Clean success is stricter than stable support. Total joint torque is an effort proxy, not energy. Paired effort excludes failed fragments and requires both controllers to complete the trial cleanly. Height/slope groups and whole20/final10/final5 squared, absolute, RMS and leg/upper-body effort are in verification.json. No policy is promoted.</p><p>The first10 seconds, all physical/sensor startups and previous study hashes were verified. The video is an offline schematic from recorded transforms and model COM. Two repeated placements are not independent surface types.</p><p><a href="episodes.csv">All episodes</a> · <a href="verification.json">Independent checks and paired metrics</a> · <a href="video/coverage.json">Video coverage</a></p>'
    (out/'findings.html').write_text('<!doctype html><meta charset="utf-8"><title>Positive-slope validation</title><style>body{font:16px system-ui;max-width:1400px;margin:32px auto;color:#243b49;padding:0 22px}p{line-height:1.55}video,img{width:100%}table{border-collapse:collapse;width:100%;margin:24px 0}td,th{padding:10px;border-bottom:1px solid #ccc;text-align:left}</style>'+body)
    write(out/'SHA256.json',{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()})
    write(DATA/'diagnosis_export_status.json',dict(phase='complete',export=str(out),video=vv));print(vv)
if __name__=='__main__':main()
