"""Export audited physical development results, explicitly separate from PPO."""
import json,hashlib,shutil,html
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit_low_seat_contact_probe_v2 import clean,summarize
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');OUT=D/'progress_report_14'
def read(p):return json.loads(Path(p).read_text())
def pairs(b,c):
    common=[i for i in range(len(c)) if clean(b[i]) and clean(c[i]) and b[i]['steps']==c[i]['steps']==1000]
    out={'common_cases':common,'n':len(common),'windows':{}}
    if not common:return out
    for window in ['metrics_whole_trial','metrics_last10s','metrics_last5s']:
        out['windows'][window]={}
        for key in b[0][window]:
            if not any(s in key for s in ['torque','rms']):continue
            bv=float(np.mean([b[i][window][key] for i in common]));cv=float(np.mean([c[i][window][key] for i in common]))
            out['windows'][window][key]=dict(baseline=bv,candidate=cv,percent_change=100*(cv/bv-1) if bv else None)
    return out
def main():
    assert read(D/'current_policy_independent_audit_10.json')['passed'];OUT.mkdir(exist_ok=False)
    names=['damped_original_01','clearance_damped_01','load_feedback_original_02','load_feedback_clearance_02',
        'normal_descent_03','hover_rotate_normal_03','normal_ankle_balance_04','normal_load_balance_04',
        'normal_task_mild_05','normal_task_firm_05','normal_landing2s_06','normal_landing3s_06','normal_slow_orientation_08','normal_counterbalance_13']
    audits=[read(D/n/'independent_audit_v2.json') for n in names]
    labels=['Damping','Deeper','Load','Load/deeper','Normal5s','Large lean','Ankle balance','Load balance','Pose mild','Pose firm','Normal2s','Normal3s','Slow/modest lean','Arm counterbalance']
    fig,ax=plt.subplots(figsize=(15,6),layout='constrained');x=np.arange(len(names));w=.25
    for j,(key,color,label) in enumerate([('clean','#278461','Clean support'),('slipping','#d18c13','Any slipping'),('unwanted_contacts','#bd3843','Unwanted contact')]):
        ax.bar(x+(j-1)*w,[a['candidate'][key] for a in audits],w,color=color,label=label)
    ax.axhline(8,color='#278461',ls='--',label='Matched resting parent:8clean')
    ax.set(xticks=x,xticklabels=labels,ylim=(0,25),ylabel='Episodes out of24',title='Physical development probes: slipping and housing contacts remain explicit')
    plt.setp(ax.get_xticklabels(),rotation=25,ha='right');ax.legend(ncols=4,loc='upper center',fontsize=9)
    fig.savefig(OUT/'comparison.png',dpi=150);plt.close(fig)
    retention={r:read(D/f'current_retention_{r}/retention.json') for r in ['07','08','09']}
    fig,axes=plt.subplots(1,3,figsize=(16,4),layout='constrained')
    for ax,(r,data) in zip(axes,retention.items()):
        keys=list(data);x=np.arange(len(keys));ax.bar(x-.15,[data[k]['retained'] for k in keys],.3,color='#278461',label='Existing successes retained')
        ax.bar(x+.15,[data[k]['summary']['clean'] for k in keys],.3,color='#3c6ca3',label='All clean cases')
        ax.axhline(20,color='#333',ls='--',label='Union retention target20/bank')
        ax.set(xticks=x,xticklabels=['Old development','Prior transfer','Recent development'],ylim=(0,25),ylabel='Cases out of24',title={'07':'Original routing','08':'Slope guard','09':'Slope and height guards'}[r])
        ax.legend(fontsize=8,loc='lower left')
    fig.savefig(OUT/'retention.png',dpi=150);plt.close(fig)
    comparisons={};OLD=D.parent/'low_positive_transfer_01/ppo_01'
    for bank,oldbank in [('known_old','development_old'),('known_transfer','development_transfer'),('known_recent','holdout')]:
        c=read(D/'current_retention_09'/bank/'candidate/results.json')['episodes'];comparisons[bank]={}
        for seed in [62111,62117]:
            for controller in ['reference','parent','current_61111','current_61117','final','selected']:
                path=OLD/f'seed_{seed}'/f'{oldbank}_{controller}'
                b=read(path/'results.json')['episodes']
                comparisons[bank][f'{seed}:{controller}']=dict(baseline=summarize(b),candidate=summarize(c),
                    gained=[i for i in range(24) if clean(c[i]) and not clean(b[i])],
                    lost=[i for i in range(24) if clean(b[i]) and not clean(c[i])],
                    common_clean_effort=pairs(b,c),
                    strict_common_effort=pairs([dict(z,success=z['success'] and not z['traction_bad_ticks']) for z in b],
                                              [dict(z,success=z['success'] and not z['traction_bad_ticks']) for z in c]),
                    source=str(path),same_physical_gpu=(seed==62111))
    (OUT/'comparisons.json').write_text(json.dumps(comparisons,indent=2))
    for n,a in zip(names,audits):(OUT/(n+'_audit.json')).write_text(json.dumps(a,indent=2))
    for n in ['retention_targets.json','reservation.json','selector_revision_08.json','selector_revision_09.json','timing_diagnosis_v2_06.json',
              'slow_orientation_screen_08.json','normal_orientation_screen_07.json','current_policy_independent_audit_10.json',
              'prior_recheck_after_retention_10.json','rate_clearance_screen_10.json','counterbalance_screen_13.json','counterbalance_diagnosis_13.json']:
        shutil.copy2(D/n,OUT/n)
    (OUT/'retention.json').write_text(json.dumps(retention,indent=2))
    data=retention['09'];retained=sum(r['retained'] for r in data.values());totalclean=sum(r['summary']['clean'] for r in data.values())
    contact=sum(r['summary']['unwanted_contacts'] for r in data.values());slips=sum(r['summary']['slipping'] for r in data.values())
    rows=''.join(f'<tr><td>{html.escape(n)}</td><td>{a["candidate"]["stable"]}</td><td>{a["candidate"]["clean"]}</td><td>{a["candidate"]["slipping"]}</td><td>{a["candidate"]["unwanted_contacts"]}</td><td>{a["lost_stable_cases"]}</td></tr>' for n,a in zip(names,audits))
    (OUT/'findings.html').write_text('''<!doctype html><html><meta charset="utf-8"><title>Low-seat physical repair progress</title><style>body{font:17px system-ui;max-width:1250px;margin:30px auto;padding:20px;color:#172231}img{width:100%}table{border-collapse:collapse;width:100%}td,th{padding:8px;border-bottom:1px solid #ccc;text-align:left}.notice{background:#fff1d7;padding:18px;border-radius:8px}</style><h1>Physical repair and retention — interim results</h1><div class="notice">No new PPO or fresh-holdout test has started. These are known development surfaces and frozen existing policies. No policy is promoted.</div>'''+
        f'<p>The guarded three-second landing with current early 61111 and resting 59117 retains <b>{retained}/60</b> previously successful conditions and is clean in <b>{totalclean}/72</b> development trials. It still records <b>{slips} slipping cases and {contact} unwanted-contact cases</b>. A no-slip/contact repair remains the prerequisite for PPO.</p>'+
        '''<p>The original routing lost two nominal 10° cases and two nominal 39 cm cases when measured values crossed its boundaries. The failed trials are preserved. Measured slope above 10.5° and height below 38 cm now select the low-seat repair, with the existing 40 cm/+20° branch retained; physical slip/contact thresholds and materials are unchanged. All first-ten-second motions, frozen early inputs/actions and starts were independently checked.</p><img src="retention.png"><p>The 20-per-bank retention target is the union of successes across existing policies, not the score of one policy.</p><h2>Physical probes</h2><img src="comparison.png"><table><tr><th>Candidate</th><th>Stable</th><th>Clean</th><th>Slip</th><th>Contact union</th><th>Lost stable cases</th></tr>'''+rows+
        '''</table><p>Every probe has 24 full 20-second trials. Contact counts include forbidden body OR illegal support-patch contact. Slipping/contact counts can overlap. Conservative margin warnings are separate and remain in the audit files.</p><p>The slower/modest-lean candidate is a separate physical experiment, not the controller used in the 72-case retention chart. Inspect its row above for actual outcomes. Its narrow model housing clearance is not a certificate. Large-rotation tracking lag is consistent with slew limits but does not prove a single cause. The new arm-counterbalance probe retained14 clean cases but fixed none of the eight remaining targeted trials:10 slips and2 housing contacts remained in its24-case isolating bank. Its conditional full-retention gate stopped correctly; no additional retention, PPO or holdout was launched.</p><p><a href="comparisons.json">Full effort comparisons</a> use only paired full-duration clean successes, with strict-margin subsets, whole20/last10/last5 torque-squared, absolute torque and RMS, including leg/upper metrics where recorded. Primary matched comparisons use the same GPU0 baselines; GPU1 repetitions are labeled separately. No failed-fragment savings or new PPO gains are claimed.</p><p>Torque is an effort proxy, not electrical energy. This is reference-assisted simulation on nearby rigid seats, not real-robot validation or sitting/leaning anywhere.</p></html>''')
    (OUT/'SHA256.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file()},indent=2))
    print(OUT)
if __name__=='__main__':main()
