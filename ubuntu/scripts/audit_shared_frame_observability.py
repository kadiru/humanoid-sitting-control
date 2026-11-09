"""Offline observability audit. Error labels are never consumed by the tracker."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_shared_frame_six import measurement_quality


def stats(values):
    values = np.asarray(values, float); values = values[np.isfinite(values)]
    return None if not len(values) else dict(n=len(values), p50=float(np.median(values)),
        p95=float(np.quantile(values, .95)), max=float(values.max()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--inputs', nargs='+', required=True)
    a = parser.parse_args(); root = Path(__file__).resolve().parents[1]
    out = Path(a.output); out.mkdir(parents=True, exist_ok=False)
    reports = {}; all_data = {}
    for name in a.inputs:
        folder = root/'runs'/name
        rows = json.loads((folder/'estimates.json').read_text())
        errors = json.loads((folder/'errors.json').read_text())
        by_key = {(e['case'], e['frame']): e for e in errors}
        for r in rows:
            r['quality'] = measurement_quality(r)
            r.setdefault('legacy_validity_reasons', ['no_seat_reference'])
            r['_error'] = by_key[(r['case'], r['frame'])]
            r['_legacy_valid'] = r.get('legacy_valid', r['valid'])
        all_data[name] = rows
        causes = Counter('+'.join(r['legacy_validity_reasons']) or 'valid' for r in rows)
        report = dict(samples=len(rows), legacy_valid=sum(r['_legacy_valid'] for r in rows),
            legacy_invalid_causes=dict(causes), groups={}, quantities={}, cases=[])
        for group, subset in [('legacy_valid', [r for r in rows if r['_legacy_valid']]),
                              ('legacy_invalid', [r for r in rows if not r['_legacy_valid']]),
                              ('scalar_budget_only', [r for r in rows if r['legacy_validity_reasons']==['scalar_drift_budget']])]:
            report['groups'][group] = dict(n=len(subset), errors={k: stats([r['_error'].get(k, np.nan) for r in subset])
                for k in ('pelvis_position_error_cm', 'clearance_error_cm', 'forward_offset_error_cm',
                          'lateral_offset_error_cm', 'yaw_axis_error_deg')})
        metrics = dict(clearance=('direct_clearance_error_cm', 2.), forward=('forward_offset_error_cm', 3.),
                       lateral=('lateral_offset_error_cm', 3.), yaw=('yaw_axis_error_deg', 5.))
        for r in rows:
            if r.get('direct_clearance_m') is not None:
                r['_error']['direct_clearance_error_cm'] = 100*abs(r['direct_clearance_m']-r['_error']['truth_clearance_m'])
        for q, (metric, threshold) in metrics.items():
            selected = [r for r in rows if r['quality']['quantities'][q]['valid']]
            report['quantities'][q] = dict(valid=len(selected), valid_fraction=len(selected)/len(rows),
                error_when_valid=stats([r['_error'].get(metric, np.nan) for r in selected]),
                all_emitted_error=stats([r['_error'].get(metric, np.nan) for r in rows]),
                errors_above_threshold_when_valid=sum(r['_error'].get(metric, np.nan)>threshold for r in selected),
                provisional_frame_threshold=threshold,
                invalid_reasons=dict(Counter(reason for r in rows for reason in r['quality']['quantities'][q]['reasons'])))
        report['alignment_valid'] = sum(r['quality']['alignment_valid'] for r in rows)
        for case in sorted({r['case'] for r in rows}):
            subset = [r for r in rows if r['case']==case]
            report['cases'].append(dict(case=case, rank=dict(Counter(r['registration']['rank'] for r in subset)),
                kinematic_prior_frames=sum(r['kinematic_prior'] for r in subset),
                legacy_valid=sum(r['_legacy_valid'] for r in subset),
                quality_valid={q:sum(r['quality']['quantities'][q]['valid'] for r in subset) for q in metrics},
                median_null_fraction={q:float(np.median([r['quality']['quantities'][q]['instantaneous_null_fraction'] for r in subset]))
                                      for q in ('forward', 'lateral', 'yaw')},
                median_self_mask_fraction=float(np.median([1-r['sensor_diagnostics']['voxel_points_after_mask']/max(1,r['sensor_diagnostics']['voxel_points_before_mask']) for r in subset])),
                max_error={metric:stats([r['_error'].get(metric,np.nan) for r in subset]) for metric,_ in metrics.values()}))
        reference = {'clean': 'shared_frame_clean_07', 'noisy':'shared_frame_noisy_06',
                     'occlusion':'shared_frame_occlusion_06'}
        mode = next((key for key in reference if '_quality_'+key+'_' in name), None)
        if mode:
            baseline = json.loads((root/'runs'/reference[mode]/'estimates.json').read_text())
            report['baseline_pose_max_abs_difference'] = max(float(np.max(np.abs(np.asarray(r['T_O_P'])-b['T_O_P']))) for r,b in zip(rows,baseline))
            report['baseline_relative_pose_max_abs_difference'] = max(float(np.max(np.abs(np.asarray(r['T_S_P'])-b['T_S_P']))) for r,b in zip(rows,baseline))
        reports[name] = report
        plot_timelines(out/name, rows)
    (out/'audit.json').write_text(json.dumps(reports, indent=2)+'\n')
    lines = ['# Observability and measurement availability audit', '',
        'Offline errors use the separate truth archive. Runtime quality uses only sensor geometry and kinematics. Directional allowances are engineering heuristics, not calibrated probabilities.', '',
        '| Run | Old valid | Current clearance | Forward | Lateral | Yaw | All alignment gates |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name,r in reports.items():
        n = r['samples']; q=r['quantities']
        lines.append(f"| {name} | {r['legacy_valid']}/{n} | {q['clearance']['valid']}/{n} | {q['forward']['valid']}/{n} | {q['lateral']['valid']}/{n} | {q['yaw']['valid']}/{n} | {r['alignment_valid']}/{n} |")
        print(name, json.dumps(dict(causes=r['legacy_invalid_causes'], quantities=q, cases=r['cases'])), flush=True)
    lines += ['', 'Errors conditional on a quality flag are diagnostic, not an independent calibration set. Forward/lateral frame thresholds are 3 cm, clearance 2 cm, and axis yaw 5 degrees. The main `valid` flag now requires all alignment gates; the former rule is archived as `legacy_valid`. Inspect `quality.quantities` for each measurement and its reasons.', '']
    (out/'audit.md').write_text('\n'.join(lines))


def plot_timelines(path, rows):
    cases = sorted({r['case'] for r in rows})
    fig, axes = plt.subplots(len(cases), 4, figsize=(18, 2.65*len(cases)), constrained_layout=True)
    for i,case in enumerate(cases):
        rs = [r for r in rows if r['case']==case]; t = [r['stamp'] for r in rs]
        ax = axes[i,0]
        for q,c in [('forward','#bf5836'),('lateral','#147ca0'),('yaw','#8258a2')]:
            ax.plot(t,[r['quality']['quantities'][q]['instantaneous_null_fraction'] for r in rs],label=q,color=c)
        ax.set(ylabel=f'Case {case}: unobserved fraction',ylim=(-.03,1.03)); ax.legend(fontsize=7, loc='upper right')
        ax = axes[i,1]
        ax.plot(t,[100*r['position_sigma_m'] for r in rs],label='old scalar budget',color='#999')
        for q,c in [('forward','#bf5836'),('lateral','#147ca0')]:
            ax.plot(t,[100*r['quality']['quantities'][q]['allowance'] for r in rs],label=q,color=c)
        ax.axhline(3.5,color='#333',ls=':',lw=.8); ax.set_ylabel('Engineering allowance (cm)'); ax.legend(fontsize=7)
        ax = axes[i,2]
        for k,label,color in [('direct_clearance_error_cm','current clearance','#23805a'),
                              ('forward_offset_error_cm','forward','#bf5836'),('lateral_offset_error_cm','lateral','#147ca0')]:
            ax.plot(t,[r['_error'].get(k,np.nan) for r in rs],label=label,color=color)
        ax.set_ylabel('Measured error (cm)'); ax.legend(fontsize=7)
        ax=axes[i,3]
        available = [[r['quality']['quantities'][q]['valid'] for r in rs] for q in ('clearance','forward','lateral','yaw')]
        ax.imshow(available,origin='lower',aspect='auto',interpolation='nearest',cmap='RdYlGn',vmin=0,vmax=1,
                  extent=(t[0],t[-1],-.5,3.5))
        ax.set_yticks(range(4),['clearance','forward','lateral','yaw']); ax.set_title('Measurement usable: green / unknown: red')
        for ax in axes[i]:
            ax.set_xlabel('Sensor time (s)'); ax.grid(alpha=.15)
    fig.suptitle(path.name+' — measurement availability, not control authorization',fontsize=14)
    fig.savefig(path.with_suffix('.png'),dpi=125); plt.close(fig)


if __name__ == '__main__':
    main()
