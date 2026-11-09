"""Offline comparison of immutable-anchor and directly anchored view tracking."""
import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
import numpy as np
from analyze_shared_frame_six import assess, longest_rejection, measurement_quality

p = argparse.ArgumentParser()
p.add_argument('--version', default='03')
p.add_argument('--validation', action='store_true')
a = p.parse_args()
root = Path(__file__).resolve().parents[1]
out = Path('/data/humanoid-sitting-perception')/f'shared_frame_reference_results_{a.version}'
out.mkdir(exist_ok=False)
load = lambda path: json.loads(path.read_text())
tags = [f'{tag}_{mode}' for tag in ('rear','original','fresh','unseen','holdout')
        for mode in ('clean','noisy','occlusion')] + ['stress','motion']
tags += [tag+'_'+mode for tag in ('extension','shape') for mode in ('clean','noisy','occlusion')]
if a.validation:
    tags += ['validation_'+mode for mode in ('clean','noisy','occlusion')]
reports = {}; hashes = set(); allrows = {}
for tag in tags:
    folder = root/f'runs/shared_frame_reference_{tag}_{a.version}'
    rows = load(folder/'estimates.json'); errors = load(folder/'errors.json')
    manifest = load(folder/'manifest.json'); hashes.add(manifest['source_sha256']['shared_frame.py'])
    result = assess(rows, errors)
    result['no_seat_samples'] = sum(r.get('status')=='no_seat' for r in rows)
    allrows[tag] = rows
    result['compute_seconds'] = manifest['compute_seconds']
    bank = [r for r in rows if r['registration']['reference'] == 'direct_anchor_view_consensus']
    result['bank_samples'] = len(bank)
    result['bank_violations'] = assess(bank, errors)['violations'] if bank else []
    result['bank_status'] = dict(Counter(r['registration'].get('bank_status','not_needed') for r in rows))
    result['admission_status'] = dict(Counter(r['registration']['reference_admission'] for r in rows))
    result['cases'] = []
    for case in sorted({r['case'] for r in rows}):
        rs = [r for r in rows if r['case']==case]
        cr = assess(rs, errors); cr['case'] = case
        cr['longest_rejected_run_frames'] = {q:longest_rejection(rs,q) for q in ('clearance','forward','lateral','yaw')}
        cr['final_bank_size'] = rs[-1]['registration']['reference_bank_size']
        cr['first_alignment_after_10s'] = next((r['stamp'] for r in rs if r['stamp']>=10 and r['valid']),None)
        result['cases'].append(cr)
    if not tag.startswith(('validation_','extension_','shape_')):
        base = root/f'runs/shared_frame_integrity_{tag}_03'
        old = load(base/'estimates.json')
        result['baseline'] = assess(old, load(base/'errors.json'))
        assert len(old)==len(rows)
        result['gained_alignment_frames'] = sum(r['valid'] and not b['valid'] for r,b in zip(rows,old))
        result['lost_alignment_frames'] = sum(b['valid'] and not r['valid'] for r,b in zip(rows,old))
    if tag.endswith('occlusion'):
        black = [r for r in rows if 6<=r['stamp']<6.5 or 8<=r['stamp']<10]
        result['blackout_accepted'] = sum(any(v['valid'] for v in measurement_quality(r)['quantities'].values()) for r in black)
        assert result['blackout_accepted']==0
    if not tag.startswith('validation_'):
        previous_tag=tag.replace('shape_','validation_')
        previous=load(root/f'runs/shared_frame_reference_{previous_tag}_02/estimates.json')
        assert len(previous)==len(rows)
        result['pose_outputs_unchanged_from_conditioned_candidate']=all(
            r.get(key)==b.get(key) for r,b in zip(rows,previous)
            for key in ('T_O_P','T_S_P','direct_clearance_m'))
        assert result['pose_outputs_unchanged_from_conditioned_candidate']
        result['before_axis_fix_availability'] = {q:sum(measurement_quality(r)['quantities'][q]['valid']
            for r in previous) for q in ('clearance','forward','lateral','yaw')}
    reports[tag] = result
assert len(hashes)==1
summary = dict(runs=reports, source_sha256=hashes.pop(),
    samples=sum(r['samples'] for r in reports.values()),
    violations=sum(len(r['violations']) for r in reports.values()),
    reference_matches=sum(r['bank_samples'] for r in reports.values()))
summary['axis_contract_tradeoff']={q:dict(
    before=sum(r['before_axis_fix_availability'][q] for r in reports.values() if 'before_axis_fix_availability' in r),
    after=sum(r['quantities'][q]['available'] for r in reports.values() if 'before_axis_fix_availability' in r))
    for q in ('clearance','forward','lateral','yaw')}
fixture=Path('/data/humanoid-sitting-perception/shared_frame_integrity_sensor_fixture_02')
isolated=load(fixture.parent/'shared_frame_reference_sensor_only_03/estimates.json')
assert sorted(path.name for path in fixture.iterdir())==['sensors.npz']
assert isolated==allrows['fresh_clean'][:len(isolated)]
summary['sensor_only']=dict(samples=len(isolated), exact_match=True,
    bank_matches=sum(r['registration']['reference']=='direct_anchor_view_consensus' for r in isolated),
    fixture_files=['sensors.npz'])
if a.validation:
    capture=root/'runs/shared_frame_reference_validation_capture_04'
    online=load(capture/'online_estimates.json'); offline=allrows['validation_clean']
    assert len(online)==len(offline)
    summary['online_replay'] = dict(samples=len(online))
    for key in ('T_O_P','T_S_P','direct_clearance_m'):
        assert all((b.get(key) is None)==(c.get(key) is None) for b,c in zip(online,offline))
        summary['online_replay'][key] = max((float(np.max(np.abs(np.asarray(b.get(key))-c.get(key))))
            for b,c in zip(online,offline) if b.get(key) is not None), default=0.)
    for key in ('quality','registration'):
        summary['online_replay'][key+'_exact'] = all(b.get(key)==c.get(key) for b,c in zip(online,offline))
    assert all(summary['online_replay'][k]==0 for k in ('T_O_P','T_S_P','direct_clearance_m'))
    assert summary['online_replay']['quality_exact'] and summary['online_replay']['registration_exact']
(out/'results.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
source=out/'source';source.mkdir()
for path in (Path(__file__),folder/'source/shared_frame.py',folder/'source/test_shared_frame.py', root/'scripts/evaluate_shared_frame.py',
             root/'scripts/analyze_shared_frame_six.py', root/'scripts/audit_shared_frame_observability.py',
             root/'scripts/test_shared_frame_evaluation.py'):
    shutil.copy2(path,source/path.name)
lines=['# Directly anchored reference views: results','',
 f"{summary['samples']:,} case/frame samples; {summary['violations']} accepted quantity/frame violations; {summary['reference_matches']} frames used reference consensus. Samples across sensor modes are correlated.",'',
 '## Method','',
 'The original scan remains immutable. At most eight separately captured views can be admitted, only from a full-rank pose registered directly to the original scan. Reverse registration must also be full rank, exceed 60% overlap, have under 8 mm plane RMS and close within 1 cm / 0.015 rad. Admissions are at least one second apart and require 3 cm translation or 0.08 rad rotation from the last admitted view. Each admitted array is copied and read-only. Views obtained from the bank cannot admit new views; graph depth is bounded. The display map is never a registration target.','',
 'The original reference is tried first. Every fallback match must also have its smallest scaled Hessian eigenvalue above 10 and above 0.5% of its largest eigenvalue (condition number below 200). This engineering margin rejects modes near the original rank cutoff; matching two references cannot repair a shared unobservable direction. A fallback requires at least two views whose independently fitted poses all agree within 1 cm / 0.015 rad. Large corrections and lost tracking use the existing bounded recovery checks. A successful fallback carries another 8 mm engineering registration allowance. These consistency checks are not statistically independent measurements or calibrated probabilities.','',
 'A rectangular patch now requires both opposing extent pairs before claiming the seat short axis. One visible pair can indicate a line direction but cannot distinguish the short axis of an incomplete rectangle or a cropped curve. Seat-frame offsets also require that axis to be confirmed. A sufficiently covered fitted ellipse remains an explicit shape hypothesis.','',
 '## Accuracy and availability','',
 '| Condition | Before pose | After pose | Gained / lost | Bank matches | Clearance | Max clearance cm | Max forward cm | Max lateral cm | Max yaw deg | Violations |',
 '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for tag,r in reports.items():
    before=str(r['baseline']['alignment_valid']) if 'baseline' in r else ('new capture' if tag.startswith('validation_') else 'additional regression')
    change=f"{r.get('gained_alignment_frames','—')} / {r.get('lost_alignment_frames','—')}"
    maxima=[r['quantities'][q]['accepted_error'] for q in ('clearance','forward','lateral','yaw')]
    lines.append(f"| {tag} | {before} | {r['alignment_valid']}/{r['samples']} | {change} | {r['bank_samples']} | {r['quantities']['clearance']['available']} | "+
                 ' | '.join(f"{v['max']:.3f}" if v else 'unavailable' for v in maxima)+f" | {len(r['violations'])} |")
lines += ['', 'Limits remain 2 cm clearance, 3 cm forward/lateral, 5 degrees axis yaw. Noisy replay adds 5 mm axial noise, 10% missing pixels and 60 ms delay. Occlusion removes depth at 6–6.5 s and 8–10 s; all quantities are withheld throughout each blackout. Raw rejected predictions remain in the artifacts and can drift.','',
 '## Verification and scope','',
 'Three additional offline evaluation tests pass. The evaluator now inventories metrics across all frames instead of assuming the first frame contains a seat. Explicit no_seat outputs count as unavailable for every quantity and remain in availability denominators. Saved runtime outputs were not altered. Evaluation-only replay reproduces all 483 previously recorded error dictionaries exactly; the original evaluator and failure logs are retained.','',
 'The first reference-view candidate (90be43b47e03ce0c61d05cc192f5ae4ba04a343a4fab2c834b1db5ef42447e34) accepted five lateral errors above 3 cm on the noisy rear-camera couch, reaching 4.274 cm. The smallest Hessian modes were only about 4.8–5.6 despite reference agreement. All outputs and its report remain in runs ending reference_*_01 and shared-frame-reference-results-01.md. It was not promoted. The added conditioning gate was fixed before this complete re-evaluation; error limits were not changed.','',
 'The conditioned candidate (8389184975985911fa77cea888ddca3b1c4ac6c673503c207d2c65427e3643c9) removed the five noisy lateral failures, but its next two-surface bank revealed a partial-ellipse startup yaw error of 11.234 degrees, accepted once in clean and once in occluded replay. Overall relative pose was already withheld; the individual yaw flag was incorrect. Those two failures are retained in candidate _02. The final shape fix leaves all pose and clearance numeric outputs exactly unchanged on the 23 regression conditions and corrects measurement eligibility. The final two-surface capture is a separate post-freeze test.','',
 'Observer SHA-256: `'+summary['source_sha256']+'`. All 41 focused tests pass, covering immutable reference storage, direct-anchor admission, rejection of a single view or disagreeing fits, preservation of the added allowance, and the prior geometry/identity/timestamp contracts. All prior datasets and failures are retained.','',
 'Sensor-only isolation replay: '+str(summary['sensor_only'])+'. The fixture directory contains no truth or geometry manifest.','',
 'The previous clean couch still loses tracking at 7.8 s (rank 3, 56% overlap with the initial scan), falling to 21% overlap at 8.0 s. Its rejected raw pose has 11.4 cm error at 8.0 s. The recovery search currently varies XY with a fixed height prior; reference transitions alone do not resolve this gap. Next investigate height recovery from an explicitly associated current floor/seat plane, preserving the identity guards. The original-camera and analytical fixtures include incompletely observed seat extents; reference transitions cannot invent hidden boundaries. Static surfaces, a visible flat floor, ideal encoder/IMU measurements and the virtual camera mount remain assumptions. This phase has no perception feedback to control, no PPO and no soft-landing tuning.','',
 'The extension and shape banks contain the previous four- and two-surface candidate captures and are now regression data. The final new capture contains two new surfaces and uses the frozen corrected observer. Compute times were measured under concurrent regression load, not as a real-time benchmark. Per-case rejection durations, recovery timing, admission reasons and accepted/raw errors are in results.json.']
if a.validation:
    lines += ['', 'New geometry was captured after source freeze. Online/replay comparison: '+str(summary['online_replay'])+'.']
lines += ['', '## Effect of requiring a confirmed coordinate axis','',
          'On the 23 identical-input regressions, the final axis correction changes eligibility only; all pose and direct-clearance values exactly match the conditioned candidate. Partially bounded seats now withhold additional coordinate-dependent quantities.','',
          '| Quantity | Available before axis correction | Available after |', '|---|---:|---:|']
for q,v in summary['axis_contract_tradeoff'].items():
    lines.append(f"| {q} | {v['before']} | {v['after']} |")
(root/f'docs/shared-frame-reference-results-{a.version}.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k!='runs'}),flush=True)
