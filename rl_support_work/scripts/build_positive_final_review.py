"""Add a read-only interpretation supplement without changing the frozen export."""
from pathlib import Path
import collections
import hashlib
import html
import json
import re
import subprocess

ROOT = Path('/Users/kadir/humanoid-sitting-control/validated_positive_slope_01')
P = ROOT / 'final_results'
v = json.loads((P / 'verification.json').read_text())
independent = json.loads((ROOT / 'independent_verified_03/verification.json').read_text())
assert independent == v and v['passed']
hashes = json.loads((P / 'SHA256.json').read_text())
assert all(hashlib.sha256((P / k).read_bytes()).hexdigest() == h for k, h in hashes.items())
audit = dict(independent_verification_identical=True, original_export_hashes=len(hashes),
             frozen_sources=v['source_hashes'], protected_prior_files=v['prior_hashes'],
             outcomes={}, effort={}, geometry={}, training=v['training'],
             contact_definition='Union of physical forbidden-body contacts and illegal support-patch flags. The inherited admitted/stress unwanted_contact field counts patch flags only; use this union.',
             selected_61111_regression='Passed the 18-environment selection gate, but the separate 24-environment development replay slipped for one 20ms tick at10.50s on44cm/+15deg, case21. Exact selected checkpoint verified. Outcome retained as failure; no causal attribution to batch size or numerical inference established.',
             policy_promotion=False)
for bank, controllers in v['paths'].items():
    for name in controllers:
        folder = P / 'results' / bank / name
        rows = json.loads((folder / 'results.json').read_text())['episodes']
        geom = json.loads((folder / 'geometry.json').read_text())
        def summarize(rr):
            return dict(trials=len(rr), stable=sum(r['success'] for r in rr),
                        clean=sum(r['success'] and not any(r['failures'].values()) and not r['slip_ticks'] and not r['illegal_ticks'] for r in rr),
                        slip=sum(bool(r['slip_ticks']) for r in rr),
                        unwanted_contacts=sum(bool(r['illegal_ticks'] or r['failures']['forbidden']) for r in rr),
                        physical_failures=sum(any(r['failures'].values()) for r in rr),
                        strict=sum(r['strict_transfer'] for r in rr))
        groups = collections.defaultdict(list)
        for r in rows:
            g = geom[r['geometry_index']]
            groups[f"{g['height']*100:g}cm/{g['pitch_deg']:+g}deg"].append(r)
        key = bank + '_' + name
        item = dict(all=summarize(rows), by_surface={k:summarize(rr) for k,rr in groups.items()})
        if bank == 'development':
            stress = [r for r in rows if (geom[r['geometry_index']]['height'] < .38 and geom[r['geometry_index']]['pitch_deg'] >= 15) or (abs(geom[r['geometry_index']]['height']-.4)<1e-6 and geom[r['geometry_index']]['pitch_deg']>=20)]
            item.update(admitted=summarize([r for r in rows if r not in stress]), stress=summarize(stress))
        audit['outcomes'][key] = item
        audit['geometry'][key] = v['evaluations'][key]['checks']['geometry_before_failure']
        if key in v['comparisons']:
            audit['effort'][key] = {b:dict(common_cases=c['common_success_cases'], grouped=c['grouped'], strict_policy_subset=v['strict_comparisons'][key][b]['grouped']) for b,c in v['comparisons'][key].items()}

video = json.loads((P / 'video/verification.json').read_text())
probe = json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration:stream=codec_name,nb_frames','-of','json',str(P/'video/summary.mp4')]))
assert video['passed'] and video['episodes']==480 and video['successes']==355 and video['failures']==125
assert 0<float(probe['format']['duration'])<60 and probe['streams'][0]['codec_name']=='h264'
assert hashlib.sha256((P/'video/summary.mp4').read_bytes()).hexdigest()==video['sha256']
audit.update(video=video,local_ffprobe=probe)

summary = '''<section style="background:#f0f7fa;border-left:5px solid #34728c;padding:18px 24px;margin:24px 0">
<h2>Verified conclusion — 16 October 2026</h2>
<p><strong>The repaired transfer improved reliability. New early PPO preserved that improvement but added no meaningful effort reduction.</strong> Both full-budget policies advanced from an8–9second pulse to a6–7second pulse. The learned adjustment lasts one second and is bounded to0.001rad; the reference then resumes, with the existing resting policy acting at11/15seconds. This is limited earlier intervention.</p>
<table><tr><th>Unseen positive slopes:24trials</th><th>Stable support</th><th>No slip/contact success</th><th>Slipping</th><th>Unwanted contact</th></tr>
<tr><td>Previous reference and current policies</td><td>18/24</td><td>18/24</td><td>6</td><td>4</td></tr>
<tr><td>Repaired reference / repaired resting policy</td><td>24/24</td><td>20/24</td><td>4</td><td>0</td></tr>
<tr><td>Both new full-budget PPO policies</td><td>24/24</td><td>20/24</td><td>4</td><td>0</td></tr></table>
<p>The remaining four unseen failures are37cm seats at+11.25° and+16.25°, each repeated twice. They slip during transfer despite retaining support. Across the admitted development conditions both finals retain18/18clean outcomes; all six excluded stress trials still slip, including two physical housing contacts at36cm/+20°. These failures remain visible.</p>
<p>On the20 common clean unseen trials, final61111/61117 reduce whole-trial torque-squared by1.033%/1.018% and resting final-five-second torque-squared by6.979%/6.929% versus the repaired reference. <strong>Almost all of this benefit already comes from the frozen resting policy.</strong> Against that policy on the identical repaired motion, new PPO changes whole-trial effort by−0.0024%/+0.0126% and resting effort by−0.0086%/+0.0450%. Those tiny, inconsistent changes do not establish an incremental learning benefit. On16 common clean trials versus current59117, whole-trial changes are−0.0016%/+0.0051%; versus current early60117, final-five-second effort slightly increases0.0155%/0.0471%.</p>
<p>Development-selected policies remain separate from the full-budget finals. Selected61111 (block1) slips once on44cm/+15° during the full development replay after passing its18-case selection evaluation; it achieves17/18admitted clean,17/24overall. Selected61117 (block2) retains18/18admitted clean. Both selected policies achieve20/24clean on unseen surfaces, with no meaningful extra effort savings. No policy is promoted or selected using holdout results.</p>
<p>Conservative margin warnings remain separate: the two finals have8/24 and4/24strict-margin-clean unseen trials. The inherited force ratio is not a measured static friction capacity. Geometry remained available at every audited observation through the evaluation failure boundary, with maximum plane-offset error2.24mm and normal error0.324° on the unseen finals. Inference: the recorded failures do not show a loss of geometry observations as their cause.</p>
<p>Independent verifier03 exactly matches verifier02. All188frozen and17,849protected hashes pass; budgets are439,200main physics transitions plus3,600separate initial settling ticks. Actual checkpoint loads, optimizer continuity/replay, sensor inputs, fixed fingers, bounded commands, full20second horizons, and pre-holdout final/selection timing pass. CPU replay tolerances: weights atol5e−5/rtol5e−4, inference atol/rtol1e−5; physical startup/prefix arrays matched exactly.</p>
<p><strong>Video:</strong>58.4seconds,480episodes across48conditions and10controllers;355SUCCESS and125FAILURE, with every first-event frame verified. It is an offline schematic reconstructed from recorded transforms, not a photorealistic simulator recording. Two seeds repeat a small set of rigid surfaces. Torque is an effort proxy, not electrical energy. A useful next development target is the remaining37cm transfer slip and sustained sensor-feedback control; no additional experiment was launched.</p>
<p><a href="review_audit.json">Detailed independent review</a> · <a href="findings.html">Original frozen-export report</a> · <a href="video/summary.mp4">Download the58.4-second video</a></p></section>'''
original = (P / 'findings.html').read_text()
review = original.replace('<video controls', summary + '<video controls', 1)
def readable_spacing(document):
    head, body = document.split('</style>', 1)
    parts = re.split(r'(<[^>]+>)', body)
    for i in range(0, len(parts), 2):
        parts[i] = re.sub(r'(?<=[A-Za-z])(?=[0-9+−])', ' ', parts[i])
        parts[i] = re.sub(r'(?<=[0-9])(?=[A-Za-df-zA-Z])', ' ', parts[i])
        parts[i] = re.sub(r'(?<=[0-9])(?=episodes|effort)', ' ', parts[i])
        parts[i] = re.sub(r'(?<=[,:;])(?=[A-Za-z0-9])', ' ', parts[i])
        parts[i] = re.sub(r'(?<=\d), (?=\d)', ',', parts[i])
        parts[i] = parts[i].replace('e −', 'e−').replace('e +', 'e+')
    return head + '</style>' + ''.join(parts)
review = readable_spacing(review)
for name, text in [('review.html',review),('review_audit.json',json.dumps(audit,indent=2))]:
    target=P/name
    assert not target.exists(), target
    target.write_text(text)
manifest={name:hashlib.sha256((P/name).read_bytes()).hexdigest() for name in ['review.html','review_audit.json']}
(P/'REVIEW_SHA256.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(dict(passed=True,original_files=len(hashes),supplement_files=len(manifest),video=video),indent=2))
