"""Wait for both finite workers, verify twice, then create a reviewable export."""
import time,traceback,shutil,html
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from low_positive_common import *
from positive_slope_common_v5 import write,sha
from run_transfer_assessment import status
from verify_low_positive_study import verify

def report(v,out):
    text=[];fig,axs=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    colors=['#6b7280','#64748b','#7c3aed','#a78bfa','#059669','#14b8a6']
    for k,seed in enumerate([62111,62117]):
        items=[(name.removeprefix(f'{seed}_holdout_'),value) for name,value in v['evaluations'].items() if name.startswith(f'{seed}_holdout_')]
        labels=[a for a,b in items];clean=[b['clean'] for a,b in items]
        axs[k,0].bar(labels,clean,color=colors[:len(items)]);axs[k,0].set_ylim(0,25);axs[k,0].set_ylabel('Clean successes /24');axs[k,0].set_title(f'Seed{seed}: fresh unseen heights/slopes');axs[k,0].tick_params(axis='x',rotation=25)
        for i,(a,b) in enumerate(items):axs[k,0].text(i,b['clean']+.3,str(b['clean']),ha='center')
        pairs=v['comparisons'][f'{seed}_holdout_final'];bases=list(pairs);changes=[]
        for b in bases:
            value=pairs[b]['grouped']['all']['metrics_whole_trial']['total_squared_torque_Nm2']['change_percent'];changes.append(np.nan if value is None else value)
        axs[k,1].bar(bases,changes,color='#059669');axs[k,1].axhline(0,color='black',lw=.8);axs[k,1].set_ylabel('Whole20s torque² change (%)');axs[k,1].set_title('Final policy: common clean pairs only; lower is better');axs[k,1].tick_params(axis='x',rotation=25)
        text.append(f'<h2>Seed {seed}</h2><table><tr><th>Controller</th><th>Stable</th><th>Clean</th><th>Slipping</th><th>Unwanted contact</th><th>Physical failure</th></tr>')
        for name,b in items:text.append(f"<tr><td>{html.escape(name)}</td><td>{b['stable']}/24</td><td>{b['clean']}/24</td><td>{b['slipping']}</td><td>{b['unwanted_contact']}</td><td>{b['physical']}</td></tr>")
        text.append('</table><p>Final-policy effort differences on common full-duration clean successes:</p><table><tr><th>Baseline</th><th>Pairs</th><th>Whole20s</th><th>Final10s</th><th>Final5s</th></tr>')
        for b in bases:
            values=[pairs[b]['grouped']['all'][w]['total_squared_torque_Nm2'] for w in ['metrics_whole_trial','metrics_last10s','metrics_last5s']]
            pretty=lambda x:'n/a' if x is None else f'{x:+.3f}%'
            text.append(f"<tr><td>{b}</td><td>{values[0]['common_successes']}</td>"+''.join('<td>'+pretty(x['change_percent'])+'</td>' for x in values)+'</tr>')
        chosen=v['selections'][str(seed)]['selected']
        description='Unchanged parent fallback; no learned checkpoint passed selection' if chosen.get('parent_fallback') else f"Checkpoint after update{chosen['update']}, control starting at{chosen['start_s']:g}s"
        text.append('</table><p>Selection: '+html.escape(description)+'. A selected checkpoint identical to the final10s policy uses that same evaluation; distinct selections are reported separately in verification.json.</p>')
    fig.savefig(out/'comparison.png',dpi=150);plt.close(fig)
    page='''<!doctype html><meta charset="utf-8"><title>Continuous sitting feedback: verified comparison</title><style>body{font:17px system-ui;max-width:1100px;margin:40px auto;padding:0 24px;color:#17212c}table{border-collapse:collapse;width:100%;margin:20px 0}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:left}img{width:100%}.note{padding:16px;background:#f0f4f8;line-height:1.5}</style><h1>Continuous sitting feedback</h1><p class="note">'''+html.escape(v['scope'])+'''</p><p>All six manually adjusted transfer variants failed to remove the targeted slips and were rejected. This pilot kept the existing repaired reference and trained directly on the problem cases. Failed episodes are never used to claim effort savings. Unwanted contact includes both forbidden-body and illegal support-patch contact. Conservative force-ratio warnings remain separate.</p><img src="comparison.png">'''+''.join(text)+'''<p><a href="verification.json">Full grouped results, source checks, budgets, geometry accuracy and strict-margin comparisons</a></p>'''
    (out/'findings.html').write_text(page)

def main():
    state=DATA/'finalization_status.json';study=DATA/'ppo_01'
    try:
        status(state,'waiting_workers')
        while True:
            states=[read(study/f'status_{s}.json') if (study/f'status_{s}.json').exists() else {} for s in [62111,62117]]
            if any(s.get('phase')=='failed' for s in states):raise RuntimeError(states)
            if all(s.get('phase')=='complete' for s in states):break
            time.sleep(30)
        status(state,'verifying')
        for i in [1,2]:
            out=study/f'verified_{i:02d}';out.mkdir(exist_ok=False);write(out/'verification.json',verify(study))
        assert read(study/'verified_01/verification.json')==read(study/'verified_02/verification.json')
        out=DATA/'final_export_01';out.mkdir(exist_ok=False);shutil.copy2(study/'verified_02/verification.json',out/'verification.json');shutil.copy2(study/'protocol.json',out/'protocol.json')
        for seed in [62111,62117]:
            d=out/f'seed_{seed}';d.mkdir();f=study/f'seed_{seed}'
            for name in ['selection.json','development_candidates.json','training_validation.json','holdout_barrier.json']:shutil.copy2(f/name,d/name)
            shutil.copy2(f/'train/final.pt',d/'final.pt')
            for p in f.iterdir():
                if p.is_dir() and p.name.startswith(('development_old_','development_transfer_','holdout_')):
                    dest=d/p.name;dest.mkdir()
                    for name in ['results.json','manifest.json','startup.npz','traction_startup.npz','trace.npz','every_tick.npz','geometry.json','sensor_audit.json','first_failures.json']:
                        if (p/name).exists():shutil.copy2(p/name,dest/name)
        report(read(out/'verification.json'),out)
        write(out/'SHA256.json',{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()})
        status(state,'complete',export=str(out),independent_verification_and_visual_review_still_required=True)
        status(DATA/'status.json','export_complete',main_training_started=True,holdout_started=True,export=str(out),independent_review_pending=True,no_policy_promotion=True)
    except Exception as e:status(state,'failed',error=repr(e),traceback=traceback.format_exc());raise

if __name__=='__main__':main()
