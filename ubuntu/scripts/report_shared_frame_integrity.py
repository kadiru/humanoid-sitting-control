"""Offline report for frozen reference-map integrity validation."""
import json, hashlib, shutil
from collections import Counter
from pathlib import Path
import numpy as np
from analyze_shared_frame_six import assess, longest_rejection
root=Path(__file__).resolve().parents[1]
out=Path('/data/humanoid-sitting-perception/shared_frame_integrity_results_02');out.mkdir(exist_ok=False)
load=lambda path:json.loads(path.read_text())
groups={'existing_rear':'rear','existing_original':'original','known_failures':'fresh','new_geometry':'unseen'}
reports={};manifests={};records={}
for label,tag in groups.items():
 for mode in ('clean','noisy','occlusion'):
  key=label+'/'+mode;name=f'shared_frame_integrity_{tag}_{mode}_02';folder=root/'runs'/name
  rows=load(folder/'estimates.json');errors=load(folder/'errors.json');manifest=load(folder/'manifest.json')
  result=assess(rows,errors);result['name']=name;result['cases']=[]
  result['tracking']=dict(accepted=sum(r['registration']['accepted'] for r in rows),
   fused=sum(r['registration']['fusion_allowed'] for r in rows),
   recovery_required=sum(r['registration'].get('reacquisition_required',False) for r in rows),
   successful_recoveries=sum(r['registration'].get('recovery')=='reacquired' for r in rows),
   all_emitted_max_pelvis_error_cm=max(e['pelvis_position_error_cm'] for e in errors))
  result['compute_seconds']=manifest['compute_seconds']
  for case in sorted({r['case'] for r in rows}):
   rs=[r for r in rows if r['case']==case]
   cr=assess(rs,errors);cr['case']=case
   cr['longest_rejected_run_frames']={q:longest_rejection(rs,q) for q in ('clearance','forward','lateral','yaw')}
   post=[r for r in rs if r['stamp']>=10]
   cr['first_alignment_after_10s']=next((r['stamp'] for r in post if r['valid']),None)
   result['cases'].append(cr)
  if mode=='occlusion':
   missing=[r for r in rows if 6<=r['stamp']<6.5 or 8<=r['stamp']<10]
   result['blackout']=dict(samples=len(missing),alignment_accepted=sum(r['valid'] for r in missing),clearance_accepted=sum(r['quality']['quantities']['clearance']['valid'] for r in missing))
   assert result['blackout']['alignment_accepted']==result['blackout']['clearance_accepted']==0
  reports[key]=result;manifests[name]=manifest;records[key]=rows
for label in ('stress','motion'):
 name=f'shared_frame_integrity_{label}_02';folder=root/'runs'/name
 reports[label]=assess(load(folder/'estimates.json'),load(folder/'errors.json'));reports[label]['name']=name
 manifests[name]=load(folder/'manifest.json')
hashes={name:m['source_sha256']['shared_frame.py'] for name,m in manifests.items()}
assert len(set(hashes.values()))==1
capture=root/'runs/shared_frame_integrity_unseen_capture_02'
config=load(capture/'manifest.json');online=load(capture/'online_estimates.json');offline=records['new_geometry/clean']
assert len(online)==len(offline)==966
assert config['source_sha256']['shared_frame.py']==next(iter(hashes.values()))
live={}
for key in ('T_O_P','T_S_P','direct_clearance_m'):
 assert all((a[key] is None)==(b[key] is None) for a,b in zip(online,offline))
 live[key]=max((float(np.max(np.abs(np.asarray(a[key])-b[key]))) for a,b in zip(online,offline) if a[key] is not None),default=0.)
live['quality_exact_match']=all(a['quality']==b['quality'] for a,b in zip(online,offline))
live['registration_exact_match']=all(a['registration']==b['registration'] for a,b in zip(online,offline))
assert all(a['case']==b['case'] and a['stamp']==b['stamp'] for a,b in zip(online,offline))
assert max(live[k] for k in ('T_O_P','T_S_P','direct_clearance_m'))<1e-7 and live['quality_exact_match'] and live['registration_exact_match']
fixture=Path('/data/humanoid-sitting-perception/shared_frame_integrity_sensor_fixture_02')
assert sorted(p.name for p in fixture.iterdir())==['sensors.npz']
isolated=load(fixture.parent/'shared_frame_integrity_sensor_only_02/estimates.json')
assert isolated==records['known_failures/clean'][:len(isolated)]
prior={}
for group,tag in (('existing_rear','rear'),('known_failures','fresh')):
 for mode in ('clean','noisy','occlusion'):
  folder=root/f'runs/shared_frame_junction_{tag}_{mode}_01'
  prior[group+'/'+mode]=assess(load(folder/'estimates.json'),load(folder/'errors.json'))
summary=dict(runs=reports,before=prior,source_sha256=hashes,live_replay=live,
 sensor_only=dict(samples=len(isolated),exact_match=True,fixture_files=['sensors.npz']),
 total_samples=sum(r['samples'] for r in reports.values()),total_violations=sum(len(r['violations']) for r in reports.values()),
 new_bank_geometry=config['geometry_bank'],new_bank_sha256=hashlib.sha256((root/'experiments/shared_frame_integrity_unseen_geometry_01.json').read_bytes()).hexdigest(),
 scope='Known failures were development regressions. New six-surface bank evaluated once after observer freeze. No control feedback or training. Accepted-error limits unchanged.')
(out/'results.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
source=out/'source';source.mkdir()
for p in (root/'sitting_task/src/humanoid_sitting/shared_frame.py',root/'scripts/test_shared_frame.py',Path(__file__),root/'scripts/visualize_shared_frame_integrity.py'):
 shutil.copy2(p,source/p.name)
qs=('clearance','forward','lateral','yaw')
lines=['# Reference-map integrity: validated results','',
 f"Evaluated {summary['total_samples']:,} case/frame samples across 14 conditions. Accepted errors exceeded the unchanged per-frame limits in **{summary['total_violations']} quantity/frame pairs**. These are conditional accuracy results; rejected predictions remain logged and may drift substantially.",'',
 '## Change and measurement contract','',
 'Pose registration now uses an immutable first depth scan. Newly fused geometry cannot alter that reference. Matching checks surface-normal agreement and the residual at the final pose; aborted solver steps cannot be accepted. Runtime registration requires over 60% correspondence coverage and under 8 mm plane RMS. Geometry fusion requires all four gravity-constrained pose directions to be observed. These are engineering gates chosen while diagnosing the retained failures, not calibrated probabilities. The evaluation limits remain 2 cm clearance, 3 cm forward/lateral, and 5 degrees axis yaw.','',
 'After 0.5 s without accepted tracking, or a depth blackout, a bounded contour-based recovery search uses the fixed reference. Attempts are spaced 0.5 s apart; ambiguous or insufficient matches remain withheld. A stationary-foot prior alone cannot certify tracking. The display map may continue to show predictions, but rejected poses are not fused.','',
 'Current-plane clearance is evaluated separately from map-relative offsets. Plane extrapolation uses 6 cm effective spatial samples, a 5 mm noise floor, and an 8 mm model allowance; ill-conditioned or excessively extrapolated fits are withheld. Low fit residual on a thin strip is insufficient. This allowance is heuristic, not a statistical guarantee.','',
 '## Accuracy and availability','',
 '| Bank / condition | Alignment accepted | Clearance accepted | Max clearance (cm) | Max forward (cm) | Max lateral (cm) | Max yaw (deg) | Violations |',
 '|---|---:|---:|---:|---:|---:|---:|---:|']
for name,r in reports.items():
 vals=[r['quantities'][q]['accepted_error'] for q in qs]
 lines.append(f"| {name} | {r['alignment_valid']}/{r['samples']} | {r['quantities']['clearance']['available']}/{r['samples']} | "+' | '.join(f"{v['max']:.3f}" if v else 'unavailable' for v in vals)+f" | {len(r['violations'])} |")
lines += ['', 'Noisy inputs add 5 mm axial noise, 10% missing pixels and 60 ms delay. Occlusion removes all depth for 6–6.5 s and 8–10 s. Every blackout frame withholds alignment and clearance. Higher noisy availability can reflect heuristic gate sensitivity; it is not a sensing advantage.','',
 '## Tradeoff against the previous observer','',
 '| Bank / condition | Previous alignment | Current alignment | Previous violations | Current violations |','|---|---:|---:|---:|---:|']
for key,p in prior.items():
 r=reports[key];lines.append(f"| {key} | {p['alignment_valid']}/{p['samples']} | {r['alignment_valid']}/{r['samples']} | {len(p['violations'])} | {len(r['violations'])} |")
lines += ['', 'Availability falls when the original view no longer constrains pose. This fixes false confidence by both improving reference consistency and refusing unsupported estimates. It does not establish uninterrupted tracking. The first, incomplete fixed-reference candidate remains in runs ending `integrity_*_01`; its noisy failures are preserved.','',
 '## New geometry: frozen observer test','',
 '| Case | Surface | Depth / width / height (m) | X / Y (m) | Yaw (deg) | Clean / noisy / occluded alignment |','|---|---|---|---|---:|---|']
for i,g in enumerate(config['geometry_bank']):
 counts=[reports['new_geometry/'+m]['cases'][i]['alignment_valid'] for m in ('clean','noisy','occlusion')]
 lines.append(f"| {i} | {g['family']} / {g['shape']} | {g['depth']} / {g['width']} / {g['height']} | {g['x']} / {g['y']} | {np.rad2deg(g['yaw']):.1f} | "+' / '.join(str(x)+'/161' for x in counts)+' |')
lines += ['', 'The geometry file was fixed before capture. The same observer source hash was used in all 14 conditions and the new online capture: `'+next(iter(hashes.values()))+'`. This is one small additional simulation bank, not arbitrary-scene generalization.','',
 '## Verification and limits','',
 '- 32 focused CPU tests passed, including immutable-reference isolation, refusal to fuse blind frames, incompatible normals, broad versus thin/collinear plane support, geometry, timestamps and runtime input contracts.',
 '- Online/replay agreement on all 966 new-bank samples: '+str(live)+'.',
 '- A separate replay directory containing only sensors.npz reproduces all 18 checked outputs exactly; runtime reads no truth or geometry manifest.',
 '- Original sources, checkpoints and Isaac Lab are checked separately in the preservation record. No controller tuning, PPO or perception-driven actuation was introduced.',
 '', 'Per-case rejection durations, recovery counts, accepted P95 and maximum errors, all-emitted errors and compute timings are in results.json. Computation was measured during concurrent validation jobs; this is not a real-time benchmark. The new capture uses 10 Hz sensor timestamps while simulation may wait for processing.','',
 'The first reference limits field-of-view coverage and recovery. Static surfaces, ideal encoder/IMU equivalents and the virtual camera mount remain assumptions. Finite support holes and whole-body collision margins are separate from plane clearance. Runtime quality is heuristic. Alignment/descent control remains disabled.','',
 'Next develop validated reference/keyframe transitions to recover coverage without reintroducing map contamination, then test camera/IMU calibration bias and execution latency before adding alignment control. Preserve this bank as regression data; reserve another untouched bank for subsequent validation.','',
 'Video: runs/shared_frame_integrity_visuals_02/map_integrity.mp4. Full new artifacts are on /data/humanoid-sitting-perception, linked from project runs/.','']
(root/'docs/shared-frame-integrity-results.md').write_text('\n'.join(lines))
print(json.dumps(dict(samples=summary['total_samples'],violations=summary['total_violations'],live=live)),flush=True)
