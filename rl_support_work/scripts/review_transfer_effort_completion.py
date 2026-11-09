"""Supplement the frozen export with an independently checked, readable conclusion.

Reads exported results only. Does not change study definitions or original evidence.
"""
import argparse
import csv
import hashlib
import html
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def table(headers, rows):
    return '<div class="table"><table><thead><tr>' + ''.join('<th>'+html.escape(str(x))+'</th>' for x in headers) + '</tr></thead><tbody>' + ''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows) + '</tbody></table></div>'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--export', type=Path, required=True)
    a = ap.parse_args()
    b = a.export
    original = read(b/'SHA256.json')
    assert isinstance(original, dict)
    for path, h in original.items():
        assert sha(b/path) == h, path
    audit = dict(original_files_verified=len(original), tasks={}, total_transitions=0, active=0, warmup=0)
    headline, selected, budgets, detailed, success_rows, geometry_rows = [], [], [], [], [], []
    all_pairs = []
    for task in ('sitting', 'leaning'):
        v = read(b/task/'verification.json')
        assert v == read(b/'independent_review'/task/'verification.json') and v['passed']
        ref = read(b/task/'holdout_reference/results.json')['episodes']
        ref_success = {e['case'] for e in ref if e['success']}
        ref_failure = {e['case'] for e in ref if any(e['failures'].values())}
        t = dict(independent_verification_identical=True, frozen_hashes=v['frozen_hashes'], prior_hashes=v['prior_hashes'], holdout_cases={}, development_regressions={})
        for label, e in v['evaluations'].items():
            success_rows.append([task, label, f"{e['successes']}/{e['count']}", e['physical_failures'], ', '.join(f'{k}: {n}' for k,n in e['failures_by_flag'].items() if n) or 'none'])
            if label.startswith('holdout'):
                eps=read(b/task/label/'results.json')['episodes']
                successes={x['case'] for x in eps if x['success']}
                failures={x['case'] for x in eps if any(x['failures'].values())}
                assert successes==ref_success and failures==ref_failure
                assert all(x['duration_s']==20 and x['steps']==1000 and x['start_stage']==0 for x in eps)
                t['holdout_cases'][label]=dict(successes=sorted(successes),physical_failures=sorted(failures))
        gm=[g for e in v['evaluations'].values() for g in e['geometry']]
        t['geometry']=dict(missing=sum(g['moving_missing'] for g in gm), max_offset_mm=max(g['max_offset_mm'] for g in gm),max_normal_deg=max(g['max_normal_deg'] for g in gm))
        for seed, budget in v['training'].items():
            for k in ('active','warmup'):audit[k]+=budget[k]
            audit['total_transitions']+=budget['transitions']
            budgets.append([task,seed,budget['updates'],budget['active'],budget['warmup'],budget['transitions']])
            assert v['selection'][seed]['selected_block']==0 and not v['selection'][seed]['selection_used_holdout']
            for typ in ('final','selected'):
                label=f'holdout_{typ}_{seed}'
                e=v['evaluations'][label]
                c=v['comparisons'][label]
                pcts=[c['corrected_reference']['grouped'][w]['total_squared_torque_Nm2']['change_percent'] for w in ('metrics_whole_trial','metrics_last10s','metrics_last5s')]
                pp=c['parent']['grouped']['metrics_whole_trial']['total_squared_torque_Nm2']['change_percent']
                row=[task,seed,f"{e['successes']}/{e['count']}",e['physical_failures']]+[f'{x:+.2f}%' for x in pcts]+[f'{pp:+.2f}%']
                (headline if typ=='final' else selected).append(row)
            refdev=read(b/task/'assessment_reference/results.json')['episodes']
            enddev=read(b/task/f'dev_{seed}_block3/results.json')['episodes']
            geoms=read(b/task/f'dev_{seed}_block3/geometry.json')
            lost=[]
            for r,p in zip(refdev,enddev):
                assert r['case']==p['case']
                if r['success'] and not p['success']:
                    g=geoms[p['geometry_index']]
                    lost.append(dict(case=p['case'],height_m=g['height'],slope_deg=g['pitch_deg'],failures=p['failures']))
            t['development_regressions'][seed]=lost
        for label, comparisons in v['comparisons'].items():
            for baseline,c in comparisons.items():
                for w,metrics in c['grouped'].items():
                    for metric,values in metrics.items():
                        detailed.append(dict(task=task,evaluation=label,baseline_name=baseline,window=w,metric=metric,**values))
                grouped=defaultdict(list)
                for p in c['pairs']:
                    all_pairs.append(dict(task=task,evaluation=label,baseline_name=baseline,**p))
                    grouped[(p['label'],p['height_m'],p['slope_deg'],p['window'],p['metric'])].append(p)
                for (name,height,slope,window,metric),pairs in grouped.items():
                    pv=mean(p['policy'] for p in pairs);bv=mean(p['baseline'] for p in pairs)
                    geometry_rows.append(dict(task=task,evaluation=label,baseline_name=baseline,label=name,height_cm=height*100,slope_deg=slope,window=window,metric=metric,common_successes=len(pairs),policy=pv,baseline=bv,change_percent=100*(pv/bv-1)))
        audit['tasks'][task]=t
    assert audit['total_transitions']==589824 and audit['active']+audit['warmup']==589824
    for name,rows in [('paired_metrics.csv',detailed),('effort_by_surface.csv',geometry_rows),('individual_pairs.csv',all_pairs)]:
        with (b/name).open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (b/'completion_review.json').write_text(json.dumps(audit,indent=2,allow_nan=False)+'\n')
    headers=['Task','Seed','Stable success','Physical failures','Whole 20s vs reference','Last 10s vs reference','Last 5s vs reference','Whole 20s vs parent']
    surface_table=[]
    for task in ('sitting','leaning'):
        v=read(b/task/'verification.json')
        for g in v['by_geometry']['holdout_reference']:
            row=[task,f"{g['height_m']*100:.2f}",g['slope_deg'],f"{g['success']}/{g['count']}",g['physical_failures']]
            for seed in v['training']:
                vals=[x for x in geometry_rows if x['task']==task and x['evaluation']==f'holdout_final_{seed}' and x['baseline_name']=='corrected_reference' and x['label']==g['label'] and x['window']=='metrics_whole_trial' and x['metric']=='total_squared_torque_Nm2']
                row.append(f"{vals[0]['change_percent']:+.2f}%" if vals else 'No common success')
            surface_table.append(row)
    document='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Unseen surfaces: final effort PPO findings</title>
<style>body{margin:0;background:#f4f6fa;color:#152739;font:17px/1.6 system-ui,sans-serif}main{max-width:1180px;margin:40px auto;padding:32px;background:white;border-radius:18px}h1{font-size:36px;line-height:1.2}h2{margin-top:42px;font-size:25px}p{max-width:1000px}.verdict{background:#fff3d9;border-left:5px solid #a96510;padding:18px 24px;font-size:20px}.table{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:14px;margin:18px 0}td,th{padding:10px;text-align:left;border-bottom:1px solid #dce3eb;vertical-align:top}th{background:#edf2f8}img{width:100%;height:auto}a{color:#135ea7}.muted{color:#536375}@media(max-width:650px){main{margin:0;padding:20px}h1{font-size:29px}}</style><main>
<p class="muted">Humanoid sitting and leaning · verified October 14, 2026 · no policy promoted</p>
<h1>The new PPO training did not reliably reduce effort.</h1>
<p class="verdict"><strong>Leaning retained stability, but effort increased. Sitting remained unreliable, and its two seeds disagreed on effort.</strong> There is no repeatable improvement over the corrected reference motion on the reserved heights and slopes.</p>
<p>Both tasks finished all 128 new updates per seed. Every held-out test starts standing and runs for 20 seconds. Success requires no physical failure during the trial and continuous valid support for the final 5 seconds. Each task has eight withheld surface geometries, each repeated at two placements: 16 trials, not 16 independent surfaces.</p>
<h2>Final full-budget policies</h2><p>The corrected reference succeeds on <strong>4/16 sitting trials</strong> with 12 physical failures, and <strong>16/16 leaning trials</strong> with no physical failures. Every final policy retains exactly the same held-out success and failure cases. Negative effort change is better. All percentages compare matched full-duration successes only.</p>'''+table(headers,headline)+'''
<p>Effort here is the time-averaged sum of squared joint torques in physical Nm over all 37 joints. Sitting comparisons cover only four successful trials: two geometries, each repeated twice. All four final policies increase the sum of absolute torques, so the small squared-torque decrease in sitting seed 55611 does not mean every measure of joint effort decreased.</p>
<p>Sitting seed 55611 also regresses on the development set: 8/18 reference successes fall to 6/18. The lost trials are the two placements of the 40 cm seat at −5°. Seed 55617 retains 8/18. Leaning retains 18/18 development successes in both seeds. Full-budget sitting seed 55611 fails the development stability gate.</p>
<h2>Earlier checkpoints selected before seeing the holdout</h2><p>All four selections come from the first 32 new updates; none uses parent fallback. The selection rule filters for stability, then ranks eligible trained checkpoints by effort. The parent is a fallback only, so selection alone does not guarantee an improvement over the parent.</p>'''+table(headers,selected)+'''
<p>The best-looking leaning selection (55711) uses 0.96% less whole-trial squared torque than the reference, but 0.47% more than its own parent. Its advantage over the reference was already present before this training. Its final-10-second advantage is only 0.11%. These earlier selections do not establish a repeatable learning benefit.</p>
<h2>Where the unseen surfaces succeed and fail</h2><p>All parents, final policies and selected policies share the reference's held-out success/failure cases. Sitting fails because of forbidden contacts and/or impact threshold violations; there are no latched falls, torque-limit or joint-limit failures in these held-out trials. The counts overlap: a trial may trigger both contact and impact. A low-torque ending after a physical failure is excluded from savings claims.</p>'''+table(['Task','Top-center height (cm)','Slope (°)','Success (all controllers)','Physical failures','First final seed effort','Second final seed effort'],surface_table)+'''
<p>Leaning heights describe the inclined slab's top-center, not a horizontal seat. The reserved set includes sitting heights 37–43 cm and slopes −7.5° to +7.5°, and leaning top-center heights 62.14–68.14 cm with slopes 37.5°–52.5°. These are a finite set of nearby rigid slabs, not arbitrary furniture.</p>
<h2>What the checks tell us</h2><p>Depth/kinematic tracking remained available in audited valid motion before failures: no missing observations. Across development and holdout evaluation, maximum measured geometry offset error was 2.212 mm for sitting and 0.959 mm for leaning; maximum normal error was 0.359° and 0.091°. The contact failures occurred despite available tracking.</p>
<p>On common held-out successes, final leaning leg squared torque rises 1.23% / 8.14% while upper-body effort falls slightly. The increased leg demand outweighs the small upper-body savings. Sitting seed 55611 reduces leg squared torque by 1.24% but raises upper-body effort by 0.28%; the other seed increases both. More training did not consistently improve these tradeoffs.</p>
<h2>Verification and budgets</h2>'''+table(['Task','Seed','New updates','Active transitions','Reset warmup','Total'],budgets)+'''
<p>Total: <strong>589,824 transitions = 573,359 active + 16,465 reset warmup</strong>. The separate 4,608-transition development smoke is excluded. Independent verifier reruns exactly reproduce both original verification reports, checking finite changed actor/critic weights, optimizer continuity, actual final/selected checkpoint loading, exact physical and sensor starts, 20-second horizons, fixed/finger commands, bounded learned commands, and active/warmup/unfinished accounting. Each task passes 82 frozen hashes and the same 436 protected prior hashes. All 1,290 original exported file hashes also pass on the Mac.</p>
<p>Holdout surfaces were reserved before assessment and excluded from reward calibration, training and checkpoint selection. Two seeds reuse the same cases. Torque is an effort proxy, not electrical energy or proof of a global optimum. This is reference-assisted residual PPO, not unassisted discovery of sitting or leaning.</p>
<h2>What follows from this result</h2><p>Keep the corrected reference as the comparison baseline. Sitting still needs contact-safe adaptation to surface height and slope before effort gains are meaningful; leaning needs a new effort optimization experiment that must beat its own parent as well as the reference. These are proposed directions, not new training launched by this check.</p>
<h2>Plot and full evidence</h2><img src="comparison.png" alt="Development and holdout success and physical failure counts, and whole-trial squared torque changes for both tasks.">
<p><a href="report.html">Original generated report</a> · <a href="completion_review.json">Independent completion review</a> · <a href="paired_metrics.csv">All paired metrics</a> · <a href="effort_by_surface.csv">Metrics by surface</a> · <a href="individual_pairs.csv">Individual trial pairs</a> · <a href="SHA256.json">Original file hashes</a> · <a href="REVIEW_SHA256.json">Supplemental review hashes</a></p>
<details><summary>All evaluation success and failure counts</summary>'''+table(['Task','Evaluation','Stable success','Physical failures','Failure flags'],success_rows)+'''</details></main></html>'''
    (b/'findings.html').write_text(document)
    # Preserve the exact review implementation with its derived artifacts.
    (b/'completion_review_source.py').write_bytes(Path(__file__).read_bytes())
    extra=['findings.html','completion_review.json','completion_review_source.py','paired_metrics.csv','effort_by_surface.csv','individual_pairs.csv']
    extra += [str(p.relative_to(b)) for p in (b/'independent_review').rglob('*') if p.is_file()]
    (b/'REVIEW_SHA256.json').write_text(json.dumps({p:sha(b/p) for p in sorted(extra)},indent=2)+'\n')
    print(json.dumps(audit,indent=2))
    print(table(headers,headline))
    print('Surface effort changes:')
    for row in surface_table: print(row)


if __name__=='__main__':
    main()
