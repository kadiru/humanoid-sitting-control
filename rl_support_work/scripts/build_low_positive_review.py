"""Add a readable interpretation without changing the frozen study export."""
import argparse
import html
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verification', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    v = json.loads(args.verification.read_text())
    assert v['passed']
    args.output.mkdir(parents=True, exist_ok=True)
    windows = ['metrics_whole_trial', 'metrics_last10s', 'metrics_last5s']
    names = {'reference': 'Repaired reference', 'parent': 'Frozen resting PPO 59117',
             'current_61111': 'Previous PPO 61111', 'current_61117': 'Previous PPO 61117',
             'final': 'Final PPO, starts at 10 s', 'selected': 'Selected PPO, tested from 8 s'}
    banks = {'development_old': 'Original development surfaces',
             'development_transfer': 'Previous holdout, now development transfer',
             'holdout': 'Fresh unseen surfaces'}
    audit = {'passed_independent_verification': True, 'sources': v['sources'],
             'protected_files': v['protected'], 'outcomes': {}, 'paired_effort': {},
             'case_changes_vs_parent': {}, 'geometry': {}, 'scope': v['scope']}
    esc = html.escape
    def table(headers, rows):
        return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+esc(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+esc(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'
    def fmt(x):
        return 'n/a' if x is None else f'{x:+.3f}%'
    def outcome(seed, bank, label):
        key = f'{seed}_{bank}_{label}'
        e = v['evaluations'][key]
        audit['outcomes'][key] = {k: e[k] for k in ['trials', 'stable', 'clean', 'slipping', 'unwanted_contact', 'physical', 'strict_clean']}
        audit['geometry'][key] = e['checks']['geometry_before_failure']
        return e
    def effort(seed, bank, label, baseline):
        key = f'{seed}_{bank}_{label}'
        c = v['comparisons'][key][baseline]['grouped']['all']
        audit['paired_effort'][key+'__vs_'+baseline] = c
        return c
    parts = ['''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Continuous sitting PPO — reviewed results</title><style>
body{font:17px/1.55 system-ui,sans-serif;color:#1d2936;background:#f6f8fb;margin:0}main{max-width:1180px;margin:auto;padding:32px 26px 70px}h1{font-size:34px;line-height:1.2;margin-bottom:12px}h2{font-size:24px;margin-top:32px}h3{font-size:20px}.lead,.card,details{background:white;border:1px solid #d8e0e9;border-radius:12px;padding:20px;margin:18px 0}.lead{border-left:6px solid #bc5b13}.muted{color:#536276}table{border-collapse:collapse;width:100%;font-size:15px}td,th{border-bottom:1px solid #dce3ea;padding:10px 12px;text-align:left;vertical-align:top}th{background:#edf2f7}tr:last-child td{border:0}.scroll{overflow-x:auto}img{max-width:100%;height:auto;background:white}summary{cursor:pointer;font-weight:650}a{color:#155ea8}code{font-size:14px}li{margin:8px 0}.tag{font-size:13px;letter-spacing:.04em;color:#536276;text-transform:uppercase}</style><main>
<div class="tag">Simulation study · two training seeds · independent verification passed</div>
<h1>More PPO control, but the remaining low-seat slips persist</h1>
<div class="lead"><strong>This pilot does not establish a better overall sitting policy.</strong> PPO now adjusts the legs repeatedly during the last 10 seconds of the motion. An earlier 8-second start passed development screening, but introduced slips on other surfaces. The full-budget policies did not reduce effort below the existing policies. No policy was promoted.</div>
<p>All 24 fresh-holdout trials remained supported without unwanted contacts or other physical failures. Sliding still occurred. Here, <strong>clean</strong> means a complete 20-second supported trial with no slipping, forbidden-body contact, illegal support-patch contact or other physical failure.</p>
<h2>Fresh unseen heights and positive slopes</h2>
<p>12 rigid surfaces, each tested twice: heights 36.5, 38.5, 40.5 and 42.5 cm; slopes +7.5°, +12.5° and +17.5°, descending toward the robot’s feet. The same conditions were reused for the two seeds.</p>''']
    rows = []
    for label in ['reference', 'parent', 'current_61111', 'current_61117']:
        a = outcome(62111, 'holdout', label)
        b = outcome(62117, 'holdout', label)
        assert audit['outcomes'][f'62111_holdout_{label}'] == audit['outcomes'][f'62117_holdout_{label}']
        rows.append([names[label], 'Both GPUs', f"{a['stable']}/24", f"{a['clean']}/24", a['slipping'], a['unwanted_contact'], f"{a['strict_clean']}/24"])
    for seed in [62111, 62117]:
        for label in ['final', 'selected']:
            a = outcome(seed, 'holdout', label)
            rows.append([names[label], seed, f"{a['stable']}/24", f"{a['clean']}/24", a['slipping'], a['unwanted_contact'], f"{a['strict_clean']}/24"])
    parts.append(table(['Controller', 'Seed', 'Supported', 'Clean', 'Slipping', 'Unwanted contacts', 'Clean + no margin warnings'], rows))
    parts.append('''<p class="muted">Margin warnings use the inherited conservative force ratio. That ratio is not a measured static-friction capacity. Warnings remain distinct from observed sliding.</p>
<h2>Did total joint effort improve?</h2><p>Values below are changes in the sum of squared torque across all 37 joints, averaged over time. Negative means lower effort. Each comparison uses only common full-duration clean successes in both controllers; failed fragments are excluded. Different baseline rows can contain different paired trials.</p>''')
    for seed in [62111, 62117]:
        parts.append(f'<h3>Seed {seed}</h3>')
        rows=[]
        for label in ['final', 'selected']:
            for base in ['reference', 'parent', 'current_61111', 'current_61117']:
                c=effort(seed, 'holdout', label, base)
                rows.append([names[label], names[base], c[windows[0]]['total_squared_torque_Nm2']['common_successes']]+[fmt(c[w]['total_squared_torque_Nm2']['change_percent']) for w in windows])
        parts.append(table(['Policy', 'Compared with', 'Clean pairs', 'Whole 20 s', 'Final 10 s', 'Final 5 s'], rows))
    parts.append('''<p><strong>Interpretation:</strong> the new full-budget policies use about 0.70–0.80% less whole-trial squared torque than the repaired reference, but 0.01–0.13% more than the previous policies. Their final-five-second effort is also higher than the previous policies. The lower effort versus the reference largely comes from the existing frozen resting policy.</p>
<p>Selected seed 62117 saves only about 0.06–0.07% whole-trial effort versus the previous policies (0.61–0.65% over the final five seconds), while losing two previously clean trials on another comparison bank. This is not an improvement without regression.</p>
<h2>Where did the slips remain or move?</h2>
<ul><li>Both low-seat fresh-holdout failures persist at <strong>36.5 cm / +12.5° and +17.5°</strong>, in both placements.</li>
<li>Final seed 62111 resolves the reference’s two slips at 38.5 cm / +7.5°, matching the previous policies. Final seed 62117 retains those slips.</li>
<li>Starting selected seed 62111 at 8 seconds resolves both 40 cm / +20° development slips, but adds one fresh-holdout slip at 42.5 cm / +12.5°.</li>
<li>Selected seed 62117 adds two slips at 39 cm / +11.25° on development transfer. Its full-budget final also adds two original-development slips at 44 cm / +15°.</li>
<li>The two known housing contacts at 36 cm / +20° remain. They were kept in training and reporting.</li></ul>''')
    for bank in banks:
        parts.append('<details><summary>'+esc(banks[bank])+': all outcomes and lost/gained clean cases</summary>')
        rows=[]
        for seed in [62111,62117]:
            parent=outcome(seed,bank,'parent');pc={x['case'] for x in parent['by_condition'] if x['clean']}
            for label in names:
                e=outcome(seed,bank,label)
                rows.append([seed,names[label],e['stable'],e['clean'],e['slipping'],e['unwanted_contact'],e['physical'],e['strict_clean']])
                if label in ['final','selected']:
                    ec={x['case'] for x in e['by_condition'] if x['clean']}
                    audit['case_changes_vs_parent'][f'{seed}_{bank}_{label}']={'lost':sorted(pc-ec),'gained':sorted(ec-pc),'conditions':e['by_condition']}
                    for base in ['reference','parent','current_61111','current_61117']:effort(seed,bank,label,base)
        parts.append(table(['Seed','Controller','Supported /24','Clean /24','Slip','Contact','Physical failure','No warnings /24'],rows))
        parts.append('</details>')
    parts.append('<details><summary>Other effort proxies and body contributions on fresh holdout</summary><p>Compared with the frozen resting policy on the same repaired motion. Nm RMS means the time average of instantaneous all-joint RMS, not RMS over the whole trajectory.</p>')
    rows=[]
    for seed in [62111,62117]:
        for label in ['final','selected']:
            c=effort(seed,'holdout',label,'parent')
            for w,title in zip(windows,['Whole20s','Final10s','Final5s']):
                rows.append([seed,label,title]+[fmt(c[w][m]['change_percent']) for m in ['total_absolute_torque_Nm','all_joint_rms_Nm','leg_squared_torque_Nm2','upper_squared_torque_Nm2']])
    parts.append(table(['Seed','Policy','Window','Absolute torque','Mean joint RMS','Leg torque²','Upper-body torque²'],rows));parts.append('</details>')
    parts.append('''<h2>What was validated</h2><p>Both fresh runs completed 16 PPO updates: 448,000 active physics ticks plus 10,500 indexed-reset warmup ticks each, 917,000 total. Initial settling was counted separately. The independent audit checks frozen sources and protected prior files, initialization parity, finite changed weights, CPU optimizer replay and Adam continuity, complete rewards, actual 417-channel inputs, matched starts, full 20-second horizons, bounded/slew-limited commands, fixed fingers and holdout timing.</p>
<p>The actor was trained from 10 seconds onward. Selected checkpoints were then tested starting at 8 seconds; there was no 8-second training. The reference controller and frozen resting policy remain active throughout their prescribed windows. This is continuous bounded reference-assisted feedback.</p>
<p>On fresh holdout, all 2,448 audited geometry observations per final/selected policy were available. Maximum seat-plane offset error was 2.256 mm and normal error about 0.310°. The observed slips do not coincide with missing geometry. Across all comparison banks, detailed before-failure geometry checks are retained in the verification file.</p>
<p>Torque is an effort proxy, not electrical energy or proof of a global optimum. Two seeds and repeated nearby rigid surfaces do not establish broad generalization. No policy was promoted.</p>
<h2>Verified comparison plot</h2><img src="comparison.png" alt="Fresh holdout clean successes and final-policy effort changes">
<p><a href="findings.html">Original generated report</a> · <a href="verification.json">Full grouped audit, including height/slope and strict subsets</a> · <a href="review_audit.json">Review data and case changes</a></p>
</main></html>''')
    (args.output/'review.html').write_text(''.join(parts))
    (args.output/'review_audit.json').write_text(json.dumps(audit,indent=2)+'\n')


if __name__=='__main__':
    main()
