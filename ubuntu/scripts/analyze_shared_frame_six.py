"""Offline six-surface camera assessment. No output is consumed by runtime."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import imageio.v2 as imageio
from evaluate_shared_frame import draw_axes


QUANTITIES = dict(clearance=('direct_clearance_error_cm', 2.),
                  forward=('forward_offset_error_cm', 3.),
                  lateral=('lateral_offset_error_cm', 3.), yaw=('yaw_axis_error_deg', 5.))


def measurement_quality(row):
    """Legacy no_seat outputs have no quantity records; all are unavailable.

    This accessor does not modify the saved runtime row or manufacture a pose.
    Missing startup measurements remain in every availability denominator.
    """
    if 'quality' in row:
        return row['quality']
    if row.get('status') != 'no_seat' or row.get('valid', True):
        raise ValueError('Missing quality outside an explicit invalid no_seat output')
    return dict(alignment_valid=False, quantities={q:dict(valid=False,
        reasons=['no_seat_reference'], instantaneous_null_fraction=1., allowance=np.nan)
        for q in QUANTITIES})


def summarize(values):
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    return dict(n=len(v), p95=float(np.quantile(v, .95)), max=float(v.max())) if len(v) else None


def longest_rejection(rows, key):
    longest = start = 0
    for r in rows:
        start = 0 if measurement_quality(r)['quantities'][key]['valid'] else start + 1
        longest = max(start, longest)
    return longest


def assess(rows, errors):
    by_key = {(e['case'], e['frame']): e for e in errors}
    quantities = {}; violations = []
    for key, (metric, limit) in QUANTITIES.items():
        accepted = []; emitted = []
        for row in rows:
            err = by_key[(row['case'], row['frame'])]
            value = (100*abs(row['direct_clearance_m']-err['truth_clearance_m'])
                     if key == 'clearance' and row.get('direct_clearance_m') is not None
                     else err.get(metric, np.nan))
            emitted.append(value)
            if measurement_quality(row)['quantities'][key]['valid']:
                accepted.append(value)
                if not np.isfinite(value) or value > limit:
                    violations.append(dict(case=row['case'], frame=row['frame'], stamp=row['stamp'],
                                           quantity=key, error=float(value), limit=limit))
        quantities[key] = dict(available=len(accepted), fraction=len(accepted)/len(rows),
            accepted_error=summarize(accepted), all_emitted_error=summarize(emitted),
            threshold=limit,
            invalid_reasons=dict(Counter(reason for r in rows for reason in measurement_quality(r)['quantities'][key]['reasons'])))
    return dict(samples=len(rows), alignment_valid=sum(measurement_quality(r)['alignment_valid'] for r in rows),
        quantities=quantities, violations=violations,
        finite_support=dict(unsupported_observed_cells_fraction=summarize([
            by_key[(r['case'],r['frame'])].get('unsupported_observed_cells_fraction',np.nan) for r in rows]),
            note='Observed cells checked against finite truth polygon with 1 cm tolerance; clearance is still to the seat plane, not a certified landing footprint.'))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--capture', required=True); p.add_argument('--prefix', required=True)
    p.add_argument('--output', required=True); p.add_argument('--video', action='store_true')
    p.add_argument('--version', default='01')
    p.add_argument('--original-run', default='shared_frame_quality_clean_04')
    a = p.parse_args(); root = Path(__file__).resolve().parents[1]
    out = Path(a.output); out.mkdir(parents=True, exist_ok=False)
    capture = root/'runs'/a.capture
    config = json.loads((capture/'manifest.json').read_text())
    geometry = config['geometry_bank']; runs = {}; records = {}; errors = {}
    names = {mode: a.prefix+'_'+mode+'_'+a.version for mode in ('clean','noisy','occlusion')}
    names['original'] = a.original_run
    for mode, name in names.items():
        folder = root/'runs'/name
        rows = json.loads((folder/'estimates.json').read_text())
        errs = json.loads((folder/'errors.json').read_text())
        records[mode] = rows; errors[mode] = errs
        result = assess(rows, errs)
        result['source_sha256'] = json.loads((folder/'manifest.json').read_text())['source_sha256']
        result['cases'] = []
        for i, g in enumerate(geometry):
            rs = [r for r in rows if r['case'] == i]
            case = assess(rs, errs)
            case.update(case=i, geometry=g, longest_rejected_run_frames={q:longest_rejection(rs,q) for q in QUANTITIES},
                        registration_ranks=dict(Counter(r['registration']['rank'] for r in rs)))
            result['cases'].append(case)
        runs[mode] = result
    with np.load(capture/'truth.npz') as ar: new_pose = ar['root_pose']
    with np.load(root/'runs/shared_frame_capture_01/truth.npz') as ar: old_pose = ar['root_pose']
    comparison = []
    for i,g in enumerate(geometry):
        delta = 100*np.linalg.norm(new_pose[:,i,:3]-old_pose[:,i,:3],axis=-1)
        angle = np.rad2deg((Rotation.from_quat(old_pose[:,i,3:]).inv()*Rotation.from_quat(new_pose[:,i,3:])).magnitude())
        comparison.append(dict(case=i, family=g['family'], root_position_difference_cm=summarize(delta),
                               root_rotation_difference_deg=summarize(angle)))
    online = json.loads((capture/'online_estimates.json').read_text())
    clean = records['clean']; assert len(online) == len(clean)
    replay = {k:max(float(np.max(np.abs(np.asarray(o[k])-r[k]))) for o,r in zip(online,clean))
              for k in ('T_O_P','T_S_P')}
    replay['quality_flags_identical'] = all(o['valid']==r['valid'] and all(
        o['quality']['quantities'][q]['valid']==r['quality']['quantities'][q]['valid'] for q in QUANTITIES)
        for o,r in zip(online,clean))
    same_observer = config['source_sha256']['shared_frame.py'] == runs['clean']['source_sha256']['shared_frame.py']
    replay['same_observer_version_as_online_capture'] = same_observer
    replay['scope'] = ('Identical observer: both poses and flags must agree.' if same_observer else
        'Revised shape association: odometry must agree; relative seat pose and quality can change. This is not live validation of the revised shape fit.')
    assert replay['T_O_P'] < 1e-6
    if same_observer:
        assert replay['T_S_P'] < 1e-6 and replay['quality_flags_identical']
    missing = [r for r in records['occlusion'] if 6 <= r['stamp'] < 6.5 or 8 <= r['stamp'] < 10]
    blackout = dict(samples=len(missing), alignment_accepted=sum(r['valid'] for r in missing),
                    clearance_accepted=sum(r['quality']['quantities']['clearance']['valid'] for r in missing))
    recovered = []
    for i in range(len(geometry)):
        rs = [r for r in records['occlusion'] if r['case']==i and r['stamp']>=10]
        accepted = next((r for r in rs if not r['registration'].get('reacquisition_required',False)),None)
        recovered.append(dict(case=i, first_relocalized_stamp=accepted['stamp'] if accepted else None,
                              final_recovery=rs[-1]['registration'].get('recovery'),
                              final_requires_reacquisition=rs[-1]['registration'].get('reacquisition_required')))
    report = dict(capture=a.capture, names=names, runs=runs, trajectory_comparison=comparison,
        live_replay=replay, depth_blackout=blackout, reacquisition=recovered,
        analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Frozen estimator and frozen controller, six previously seen simulated surface geometries. Runtime uses sensor archive only. Truth is offline. Quality budgets are heuristics; no control authorization, hardware calibration or untouched geometry generalization claim.')
    (out/'validation.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    lines = ['# Frozen observer: six-surface validation', '', report['scope'], '',
        '| Condition | Complete alignment | Clearance | Forward | Lateral | Yaw | Accepted threshold violations |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for mode,r in runs.items():
        n=r['samples']; counts=[f"{r['quantities'][q]['available']}/{n}" for q in QUANTITIES]
        lines.append(f"| {mode} | {r['alignment_valid']}/{n} | "+' | '.join(counts)+f" | {len(r['violations'])} |")
    lines += ['', 'Thresholds: 2 cm clearance, 3 cm forward/lateral, 5 degrees axis yaw. Errors are evaluated per accepted frame; all rejected-frame errors are retained.', '',
        '| Surface | Original alignment | New clean | New noisy | New occlusion |', '|---|---:|---:|---:|---:|']
    for i,g in enumerate(geometry):
        lines.append(f"| {g['family']} / {g['shape']} | "+' | '.join(f"{runs[m]['cases'][i]['alignment_valid']}/161" for m in ('original','clean','noisy','occlusion'))+' |')
    lines += ['', '## Accepted-error P95', '', '| Condition | Clearance (cm) | Forward (cm) | Lateral (cm) | Yaw (degrees) |', '|---|---:|---:|---:|---:|']
    for mode,r in runs.items():
        values = [r['quantities'][q]['accepted_error'] for q in QUANTITIES]
        lines.append('| '+mode+' | '+' | '.join(f"{v['p95']:.3f}" if v else 'unavailable' for v in values)+' |')
    lines += ['', f'Live/replay agreement: `{replay}`.', f'Depth blackout abstention: `{blackout}`.',
              '', '## Rejected measurements and finite support', '']
    for mode in ('clean','noisy','occlusion'):
        r=runs[mode]
        lines.append(f"- {mode}: "+'; '.join(q+': '+str(r['quantities'][q]['invalid_reasons']) for q in QUANTITIES)+'.')
        lines.append(f"  Unsupported observed cells: {r['finite_support']['unsupported_observed_cells_fraction']}.")
    lines += ['', '## Experimental isolation', '', 'Offline differences between the original and added-camera root trajectories:', '']
    for c in comparison:
        lines.append(f"- Case {c['case']} ({c['family']}): max position {c['root_position_difference_cm']['max']:.6g} cm; max rotation {c['root_rotation_difference_deg']['max']:.6g} degrees.")
    lines += ['', '## Limitations', '', 'The six geometries are existing regression scenes. The camera mount is simulated and its RGB channel remains unavailable; depth is valid and is shown. Clearance measures the current seat plane; finite support occupancy is measured separately. Observed support admission does not certify the whole body or a sitting trajectory. Partial view, occlusion, and unobserved motion can still make alignment unavailable. No PPO, controller tuning, or perception-driven actuation was introduced.', '']
    (out/'validation.md').write_text('\n'.join(lines))
    render_summary(out, capture, root, geometry, records)
    if a.video: render_video(out,capture,root,names['clean'],geometry,records['clean'],errors['clean'])
    print(json.dumps({m:dict(alignment=r['alignment_valid'],violations=len(r['violations'])) for m,r in runs.items()}),flush=True)


def render_summary(out,capture,root,geometry,records):
    with np.load(capture/'sensors.npz') as ar: depth=ar['depth'][80]
    with np.load(root/'runs/shared_frame_capture_01/sensors.npz') as ar: old=ar['depth'][80]
    fig,axs=plt.subplots(len(geometry),3,figsize=(14,17),constrained_layout=True)
    for i,g in enumerate(geometry):
        for col,d,title in [(0,old[i],'Original view'),(1,depth[i],'Rear vertical view')]:
            axs[i,col].imshow(np.ma.masked_invalid(d),cmap='viridis',vmin=.1,vmax=1.6)
            axs[i,col].axis('off');axs[i,col].set_title(f"{g['family']} / {g['shape']} — {title}",fontsize=10)
        modes=('original','clean','noisy','occlusion')
        flags=[[r['valid'] for r in records[m] if r['case']==i] for m in modes]
        ax=axs[i,2];ax.imshow(flags,aspect='auto',interpolation='nearest',origin='lower',cmap='RdYlGn',vmin=0,vmax=1,extent=(0,16,-.5,3.5))
        ax.set_yticks(range(4),['Original','New clean','New noisy','New occlusion']);ax.set_xlabel('Sensor time (s)')
        ax.set_title('All alignment gates: green / unconfirmed: red',fontsize=10)
    fig.suptitle('Six surfaces: actual depth at 8 s and complete-alignment availability\nSame 0.1–1.6 m depth color scale; frozen observer/controller',fontsize=15)
    fig.savefig(out/'six_surface_comparison.png',dpi=135);plt.close(fig)


def render_video(out,capture,root,name,geometry,rows,errors):
    with np.load(root/'runs'/name/'tracks.npz') as ar: tr={k:ar[k] for k in ar.files}
    with np.load(capture/'sensors.npz') as ar: depth=ar['depth']
    fig=plt.figure(figsize=(13.6,8.4),dpi=100)
    writer=imageio.get_writer(out/'six_surface_shared_frame.mp4',fps=10,codec='libx264',quality=7,macro_block_size=2)
    for i,g in enumerate(geometry):
        rs=[r for r in rows if r['case']==i];es=[e for e in errors if e['case']==i]
        selected=np.linspace(0,len(rs)-1,40).astype(int)
        for k in selected:
            fig.clear();ax=fig.add_axes([.01,.30,.54,.61],projection='3d');ax.set_facecolor('#f6f8fb')
            for key,color,size in [('cloud',None,1.5),('cells','#ed9d26',7),('proxy','#d63c79',3)]:
                pts=tr[key][k,i];pts=pts[np.isfinite(pts).all(1)]
                if len(pts):
                    if color: ax.scatter(*pts.T,color=color,s=size,alpha=.6)
                    else: ax.scatter(*pts.T,c=pts[:,2],cmap='viridis',s=size,alpha=.35,vmin=0,vmax=.7)
            ax.add_collection3d(Line3DCollection(tr['skeleton'][k,i],colors='#1a2a40',linewidths=2))
            draw_axes(ax,tr['T_O_P'][k,i]);draw_axes(ax,tr['T_O_S'][k,i])
            ax.set(xlim=(-1.,.4),ylim=(-.8,.8),zlim=(0,1.45),xlabel='O x (m)',ylabel='O y (m)',zlabel='O z (m)')
            ax.set_box_aspect((1.4,1.6,1.45));ax.view_init(22,-57);ax.set_title('Common frame: depth, body, observed support',fontsize=11)
            cam=fig.add_axes([.59,.60,.31,.28]);cam.imshow(np.ma.masked_invalid(depth[k,i]),cmap='viridis',vmin=.1,vmax=1.6)
            cam.axis('off');cam.set_title('Observer axial depth (RGB unavailable)',fontsize=10)
            row=rs[k]
            for j,q in enumerate(QUANTITIES):
                ok=row['quality']['quantities'][q]['valid']
                label='measured' if ok else 'unconfirmed'
                if q=='clearance' and ok: label=f"{100*row['direct_clearance_m']:.1f} cm measured"
                fig.text(.57,.54-j*.049,q.capitalize()+': '+label,fontsize=12,color='#157249' if ok else '#a34727')
            fig.text(.57,.315,'Plane clearance and finite support are separate measurements.',fontsize=9)
            fig.text(.57,.29,'Orange: observed support. Magenta: body geometry.',fontsize=9)
            ax=fig.add_axes([.08,.10,.83,.145]);t=[r['stamp'] for r in rs]
            for q,col in [('clearance','#24805c'),('forward','#c56534'),('lateral','#167a9e')]:
                v=[100*abs(r['direct_clearance_m']-e['truth_clearance_m']) if q=='clearance' and r['direct_clearance_m'] is not None else e.get(QUANTITIES[q][0],np.nan) for r,e in zip(rs,es)]
                mask=np.array([r['quality']['quantities'][q]['valid'] for r in rs])
                ax.plot(t,v,color=col,alpha=.35,lw=1)
                ax.scatter(np.asarray(t)[mask],np.asarray(v)[mask],s=5,color=col,label=q+' accepted')
            ax.axvline(row['stamp'],color='#111',lw=1);ax.set(xlabel='Sensor time (s)',ylabel='Offline error (cm)')
            ax.grid(alpha=.2);ax.legend(fontsize=8,ncol=3,loc='upper right')
            fig.suptitle(f"{g['family']} / {g['shape']} — sensor t = {row['stamp']:.1f} s",fontsize=16,y=.98)
            fig.text(.08,.015,'Truth only labels errors. Faint curves include rejected estimates; dots indicate accepted measurements. 16 s shown in 4 s.',fontsize=9)
            fig.canvas.draw();writer.append_data(np.asarray(fig.canvas.buffer_rgba())[...,:3])
            if k==selected[20]:fig.savefig(out/f'case_{i:02d}_mid.png',dpi=135)
    writer.close();plt.close(fig)


if __name__=='__main__': main()
