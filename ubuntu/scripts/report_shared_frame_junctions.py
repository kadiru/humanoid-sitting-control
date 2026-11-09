"""Offline assessment of measured seat/wall junctions and fresh geometry."""
import json
from pathlib import Path
import numpy as np
from analyze_shared_frame_six import assess

root=Path(__file__).resolve().parents[1]
out=root/'runs/shared_frame_junction_results_02';out.mkdir(exist_ok=False)
groups={
 'existing_rear':('shared_frame_rear_down_six_capture_01','shared_frame_junction_rear'),
 'existing_original':('shared_frame_capture_01','shared_frame_junction_original'),
 'fresh_geometry':('shared_frame_junction_fresh_capture_01','shared_frame_junction_fresh')}
all_reports={};records={};configs={};sources={}
for group,(capture,prefix) in groups.items():
    configs[group]=json.loads((root/'runs'/capture/'manifest.json').read_text())
    for mode in ('clean','noisy','occlusion'):
        name=prefix+'_'+mode+'_01';folder=root/'runs'/name
        rows=json.loads((folder/'estimates.json').read_text())
        errors=json.loads((folder/'errors.json').read_text())
        report=assess(rows,errors);report['cases']=[]
        for i,g in enumerate(configs[group]['geometry_bank']):
            rs=[r for r in rows if r['case']==i]
            case=assess(rs,errors)
            case.update(case=i,geometry=g,junction_count_max=max(r['quality']['seat_boundary']['junction_count'] for r in rs),
                        junction_frames=sum(r['quality']['seat_boundary']['junction_count']>0 for r in rs))
            report['cases'].append(case)
        if mode=='occlusion':
            lost=[r for r in rows if 6<=r['stamp']<6.5 or 8<=r['stamp']<10]
            report['depth_blackout']=dict(samples=len(lost),alignment_accepted=sum(r['valid'] for r in lost),
                 clearance_accepted=sum(r['quality']['quantities']['clearance']['valid'] for r in lost))
        key=group+'/'+mode;all_reports[key]=report;records[key]=rows
        sources[name]=json.loads((folder/'manifest.json').read_text())['source_sha256']['shared_frame.py']
for name in ('shared_frame_junction_stress_01','shared_frame_junction_motion_01'):
    folder=root/'runs'/name
    all_reports[name]=assess(json.loads((folder/'estimates.json').read_text()),json.loads((folder/'errors.json').read_text()))
    sources[name]=json.loads((folder/'manifest.json').read_text())['source_sha256']['shared_frame.py']
assert len(set(sources.values()))==1
old=json.loads((root/'runs/shared_frame_rear_down_six_clean_02/estimates.json').read_text())
new=records['existing_rear/clean']
invariance={k:max(float(np.max(np.abs(np.asarray(x[k])-y[k]))) for x,y in zip(old,new)) for k in ('T_O_P','T_S_P','direct_clearance_m')}
assert max(invariance.values())<1e-7
online=json.loads((root/'runs/shared_frame_junction_fresh_capture_01/online_estimates.json').read_text())
offline=records['fresh_geometry/clean']
assert len(online)==len(offline)==966
assert all(a['case']==b['case'] and a['stamp']==b['stamp'] for a,b in zip(online,offline))
live={}
for k in ('T_O_P','T_S_P','direct_clearance_m'):
    assert all((a[k] is None)==(b[k] is None) for a,b in zip(online,offline))
    live[k]=max((float(np.max(np.abs(np.asarray(a[k])-b[k]))) for a,b in zip(online,offline) if a[k] is not None),default=0.)
live['quality_exact_match']=all(a['quality']==b['quality'] for a,b in zip(online,offline))
assert max(v for k,v in live.items() if k!='quality_exact_match')<1e-7 and live['quality_exact_match']
assert configs['fresh_geometry']['source_sha256']['shared_frame.py']==next(iter(sources.values()))
before={i:sum(r['valid'] for r in old if r['case']==i) for i in range(6)}
summary=dict(runs=all_reports,source_sha256=sources,existing_pose_invariance=invariance,
    fresh_live_replay=live,before_clean_alignment_by_case=before,
    total_samples=sum(r['samples'] for r in all_reports.values()),
    total_violating_quantity_frames=sum(len(r['violations']) for r in all_reports.values()),
    note='Frozen controller. Existing and analytical cases are regressions. Fresh geometry was fixed before its evaluation; the observer was not tuned on those results. Provisional thresholds are not probabilistic guarantees.')
(out/'results.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
qs=('clearance','forward','lateral','yaw')
lines=['# Measured seat/wall junctions: validation results','',
 'The observer now uses bounded intersections of measured vertical surfaces and the measured seat plane to confirm seat boundaries. It does not need furniture labels, model dimensions, simulator poses, or a new controller.','',
 '**Fresh-geometry validation failed:** accepted offsets exceed 3 cm, and the noisy run also has clearance and yaw outliers. Boundary recovery passes the existing-scene regressions, but this observer is not ready to authorize alignment control. The frozen fresh-bank failures are retained, not tuned away.','',
 '## Availability on existing scenes','',
 '| Surface | Before: clean alignment | After: clean | After: noisy | After: occlusion |', '|---|---:|---:|---:|---:|']
for i,g in enumerate(configs['existing_rear']['geometry_bank']):
    values=[all_reports['existing_rear/'+m]['cases'][i]['alignment_valid'] for m in ('clean','noisy','occlusion')]
    lines.append(f"| {g['family']} / {g['shape']} | {before[i]}/161 | "+' | '.join(f'{v}/161' for v in values)+' |')
lines += ['','## Accepted errors and availability','',
 '| Bank / condition | All alignment gates | Clearance available | Clearance P95 (mm) | Forward P95 (mm) | Lateral P95 (mm) | Yaw P95 (deg) | Violating quantity/frames |',
 '|---|---:|---:|---:|---:|---:|---:|---:|']
for name,r in all_reports.items():
    vals=[r['quantities'][q]['accepted_error'] for q in qs]
    errors=[f"{v['p95']*(1 if q=='yaw' else 10):.3f}" if v else '—' for q,v in zip(qs,vals)]
    lines.append(f"| {name} | {r['alignment_valid']}/{r['samples']} | {r['quantities']['clearance']['available']}/{r['samples']} | "+' | '.join(errors)+f" | {len(r['violations'])} |")
lines += ['', 'Fresh-bank accepted-error maxima (clearance / forward / lateral in cm; yaw in degrees):']
for mode in ('clean','noisy','occlusion'):
    r=all_reports['fresh_geometry/'+mode]
    lines.append('- '+mode+': '+' / '.join(f"{r['quantities'][q]['accepted_error']['max']:.3f}" for q in qs))
lines += ['', 'Errors are conditional on each quantity being accepted; all rejected-frame errors remain available. Provisional per-frame limits: 2 cm clearance, 3 cm forward/lateral, 5° axis yaw. Noisy inputs add 5 mm Gaussian axial noise, 10% missing pixels, and 60 ms latency. Occlusion removes all depth at 6–6.5 s and 8–10 s. Flags remain heuristic, not calibrated probabilities. Higher noisy availability reflects gate sensitivity, not evidence that noise improves sensing.','',
 '## Fresh geometry bank','',
 '| Surface | Depth × width × height (m) | Seat x/y (m) | Yaw (deg) | Clean / noisy / occluded alignment |',
 '|---|---|---|---:|---|']
for i,g in enumerate(configs['fresh_geometry']['geometry_bank']):
    values=[all_reports['fresh_geometry/'+m]['cases'][i]['alignment_valid'] for m in ('clean','noisy','occlusion')]
    lines.append(f"| {g['family']} / {g['shape']} | {g['depth']} × {g['width']} × {g['height']} | {g['x']} / {g['y']} | {np.rad2deg(g['yaw']):.1f} | "+' / '.join(f'{v}/161' for v in values)+' |')
lines += ['', 'The geometry JSON was recorded before capture; the same observer hash was used for every run. This is a small new simulation bank, not evidence of arbitrary geometry generalization. The analytical motion banks remain prior regressions.','',
 '## Geometry evidence and verification','',
 '- Vertical surfaces are fitted from depth-cloud points/normals with bounded deterministic plane proposals. Each junction requires at least 24 wall points, ≤8 mm fit RMS, ≥12 cm wall span, visible height above 8 cm, and samples within 5 cm of seat height. Only locally supported intersection samples within 3 cm of observed seat points are retained. No unseen span is filled. These are geometric tolerances, not proof of physical attachment.',
 '- Junction samples qualify the seat boundary; they never enter the odometry registration map. Existing pelvis/seat poses and direct clearance are unchanged: '+str(invariance)+'.',
 '- 27 tests passed, including measured wall support, rotated junctions, floating-wall rejection, distant-wall rejection, and preservation of gaps.',
 '- Fresh six-scene online/replay check (966 frames): '+str(live)+'.',
 '- No PPO, training, soft-landing tuning, or perception-driven actuation was introduced. Original source/checkpoint preservation is separately verified.',
 '', '## Practical limits and next step','',
 'The scene map assumes static surfaces. A near-base point within 5 cm is evidence for a bounded geometric intersection, not a certified contact seam. Clearance is to the observed seat plane; finite support holes and body geometry remain separate. The virtual camera mount and ideal encoder/IMU equivalents are still simulation assumptions.',
 'The fresh-bank failures include accumulated-map pose errors of 5.7–7.5 cm in inspected chair/bench frames. A shifted map can retain low local registration residuals and reset its engineering allowance, while its seat center shifts by a different amount. Accurate current-plane clearance therefore coexists with wrong map-relative offsets. Current-frame boundary checks reject the incomplete shape in the inspected failing dimensions; this is diagnostic evidence, not a deployed fix. See `runs/shared_frame_junction_drift_diagnostic_01/diagnostic.json`.',
 'Next fix map consistency and recovery during partial visibility or fast motion, and distinguish current relative measurements from map predictions. Revalidate the preserved failures without relaxing error limits. Then test extrinsic/IMU calibration errors and finite body-support margins before an alignment-only controller; keep descent disabled.', '',
 'Machine-readable measurements: `runs/shared_frame_junction_results_02/results.json`. Final visualizations show observed junctions in cyan. All original recordings and prior observer snapshots are retained.', '']
(root/'docs/shared-frame-junction-results.md').write_text('\n'.join(lines))
print(json.dumps(dict(samples=summary['total_samples'],violations=summary['total_violating_quantity_frames'],
                     live=live,invariance=invariance)),flush=True)
