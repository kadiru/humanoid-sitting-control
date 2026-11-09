"""Privileged OFFLINE metrics and visualization. Never imported by the runtime."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull, cKDTree
from scipy.spatial.transform import Rotation
from humanoid_sitting.shared_frame import BodyModel, transform, apply
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import imageio.v2 as imageio


def pose(value):
    return transform(Rotation.from_quat(value[3:]).as_matrix(), value[:3])


def summary(values):
    v = np.asarray(values, float); v = v[np.isfinite(v)]
    if not len(v):
        return None
    return dict(n=len(v), mean=float(np.mean(v)), median=float(np.median(v)),
                p95=float(np.quantile(v, .95)), max=float(np.max(v)))


def metric_keys(metrics):
    # The first scan can have no seat. Later observed quantities must still be
    # evaluated, and initial unavailable samples must not be dropped.
    return sorted({k for m in metrics for k in m
        if k.endswith(('_cm', '_deg', '_ms', '_fraction')) and
        not k.startswith(('truth', 'estimated', 'sigma'))})


def polygon(g):
    if g.get('shape') == 'ellipse':
        a = np.arange(16)*2*np.pi/16
        p = np.c_[.5*np.cos(a), .5*np.sin(a)]
    elif g.get('shape') == 'chamfered':
        p = np.array([[-.35,-.5],[.35,-.5],[.5,-.35],[.5,.35],[.35,.5],[-.35,.5],[-.5,.35],[-.5,-.35]])
    else:
        p = np.array([[-.5,-.5],[.5,-.5],[.5,.5],[-.5,.5]])
    p *= [g['depth'], g['width']]
    if g['family'] in ('chair', 'rigid_couch'):
        p[:, 0] = np.maximum(p[:, 0], -g['depth']/2+.06)
    return p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--capture', required=True)
    parser.add_argument('--tracking', required=True)
    parser.add_argument('--video', action='store_true')
    a = parser.parse_args()
    capture = Path(a.capture); out = Path(a.tracking)
    cfg = json.loads((capture/'manifest.json').read_text())
    with np.load(capture/'truth.npz') as archive:
        truth = {k: archive[k] for k in archive.files}
    with np.load(capture/'sensors.npz') as archive:
        sensor = {k: archive[k] for k in ('q', 'rgb')}
    with np.load(out/'tracks.npz') as archive:
        tracks = {k: archive[k] for k in archive.files}
    estimates = json.loads((out/'estimates.json').read_text())
    model = BodyModel(Path(__file__).resolve().parents[1]/'runs/g1_kinematics_geometry_04/model.json')
    count = len(cfg['geometry_bank']); metrics = []; alignments = []
    for i in range(count):
        init = pose(truth['root_pose'][0, i])
        # Gauge alignment only: world XY origin and yaw are not observable.
        R = init[:3, :3] @ tracks['T_O_P'][0, i, :3, :3].T
        yaw = np.arctan2(R[1, 0], R[0, 0])
        alignments.append(transform(Rotation.from_euler('z', yaw).as_matrix(), [*init[:2, 3], 0.]))
    for row in estimates:
        k, i = row['frame'], row['case']; g = cfg['geometry_bank'][i]
        T_W_O = alignments[i]; T_O_W = np.linalg.inv(T_W_O)
        true_root = pose(truth['root_pose'][k, i]); true_seat = pose(truth['seat_pose'][k, i])
        true_seat = true_seat @ transform(p=[0., 0., .05])
        p = polygon(g); center = (p.min(0)+p.max(0))/2
        true_seat = true_seat @ transform(p=[*center, 0.])
        expected = np.linalg.inv(true_seat) @ true_root
        T = tracks['T_O_P'][k, i]; S = tracks['T_O_S'][k, i]
        est_world = T_W_O @ T
        body_error = []
        fk = model.fk(sensor['q'][k, i])
        for j, name in enumerate(cfg['body_names']):
            body_error.append(np.linalg.norm((est_world @ fk[name])[:3, 3]-truth['body_pose'][k, i, j, :3]))
        m = dict(case=i, frame=k, stamp=row['stamp'], valid=row['valid'], status=row['status'],
            pelvis_position_error_cm=100*np.linalg.norm(est_world[:3, 3]-true_root[:3, 3]),
            pelvis_orientation_error_deg=np.rad2deg(Rotation.from_matrix(est_world[:3, :3].T@true_root[:3, :3]).magnitude()),
            body_position_error_cm=100*max(body_error),
            floor_height_error_cm=100*abs(tracks['floor'][k, i, 3]/tracks['floor'][k, i, 2]),
            latency_ms=row['latency_s']*1000,
            rank=row['registration']['rank'], sigma_cm=100*row['position_sigma_m'])
        true_pelvis_vertices = apply(true_root, model.vertices['pelvis'])
        true_clearance = float(apply(np.linalg.inv(true_seat), true_pelvis_vertices)[:, 2].min())
        m['truth_clearance_m'] = true_clearance
        if np.isfinite(S).all():
            n = S[:3, 2]; d = -n @ S[:3, 3]
            seat_center_O = apply(T_O_W, true_seat[:3, 3][None])[0]
            est_rel = np.linalg.inv(S) @ T
            yaw_est = np.arctan2(est_rel[1, 0], est_rel[0, 0])
            yaw_true = np.arctan2(expected[1, 0], expected[0, 0])
            yaw_error = (yaw_est-yaw_true+np.pi/2) % np.pi-np.pi/2
            cells = tracks['cells'][k, i]; cells = cells[np.isfinite(cells).all(1)]
            local = apply(np.linalg.inv(true_seat) @ T_W_O, cells)[:, :2]
            eq = ConvexHull(p-center).equations
            unsupported = (local @ eq[:, :2].T+eq[:, 2]).max(1) > .01
            observed_fraction = row['observed_area_m2']/ConvexHull(p).volume
            m.update(clearance_error_cm=100*abs(row['clearance_m']-true_clearance),
                clearance_signed_error_cm=100*(row['clearance_m']-true_clearance),
                truth_clearance_m=true_clearance, estimated_clearance_m=row['clearance_m'],
                seat_plane_height_error_cm=100*abs(n @ seat_center_O+d)/n[2],
                seat_normal_error_deg=np.rad2deg(np.arccos(np.clip(n[2], -1, 1))),
                seat_visible_center_error_cm=100*np.linalg.norm(S[:2, 3]-seat_center_O[:2]),
                lateral_offset_error_cm=100*abs(est_rel[1, 3]-expected[1, 3]),
                forward_offset_error_cm=100*abs(est_rel[0, 3]-expected[0, 3]),
                yaw_axis_error_deg=abs(np.rad2deg(yaw_error)),
                unsupported_observed_cells_fraction=float(unsupported.mean()),
                observed_fraction_of_true_support=float(observed_fraction))
        metrics.append(m)
    keys = metric_keys(metrics)
    report = dict(capture=str(capture), tracking=str(out), frames=len(metrics), cases=count,
        valid_frames=sum(x['valid'] for x in metrics), valid_fraction=float(np.mean([x['valid'] for x in metrics])),
        metric_scope='All emitted estimates, including stale/invalid. XY+yaw gauge aligned at first frame only.',
        aggregate={key: summary([m.get(key, np.nan) for m in metrics]) for key in keys},
        valid_only={key: summary([m.get(key, np.nan) for m in metrics if m['valid']]) for key in keys},
        cases_summary=[], thresholds=dict(clearance_p95_cm=2., normal_p95_deg=2., lateral_p95_cm=3., yaw_p95_deg=5.),
        tolerance_basis='5 mm depth noise plus up to 1 cm extrinsic/contact-geometry systematic budget; 2 cm clearance, 3 cm lateral and 5 deg heading are provisional simulation gates, not calibrated hardware guarantees.',
        evaluator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    for i, g in enumerate(cfg['geometry_bank']):
        subset = [m for m in metrics if m['case'] == i]
        root = truth['root_pose'][:, i]
        report['cases_summary'].append(dict(case=i, geometry=g, frames=len(subset),
            valid_fraction=float(np.mean([m['valid'] for m in subset])),
            pelvis_travel_cm=(100*np.ptp(root[:, :3], axis=0)).tolist(),
            metrics={key: summary([m.get(key, np.nan) for m in subset]) for key in keys}))
    report['gates'] = {key: bool(report['aggregate'].get(metric) and report['aggregate'][metric]['p95'] <= value) for key, metric, value in [
        ('clearance', 'clearance_error_cm', 2.), ('normal', 'seat_normal_error_deg', 2.),
        ('lateral', 'lateral_offset_error_cm', 3.), ('yaw_axis', 'yaw_axis_error_deg', 5.)]}
    (out/'evaluation.json').write_text(json.dumps(report, indent=2)+'\n')
    (out/'errors.json').write_text(json.dumps(metrics, indent=2)+'\n')
    print(json.dumps(dict(valid_fraction=report['valid_fraction'], gates=report['gates'],
        metrics={k: v for k, v in report['aggregate'].items() if k in
            ('clearance_error_cm', 'lateral_offset_error_cm', 'yaw_axis_error_deg', 'pelvis_position_error_cm', 'seat_plane_height_error_cm')}), indent=2), flush=True)
    colors = plt.cm.tab10(np.arange(count))
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), constrained_layout=True)
    panels = [('clearance_error_cm', 'Pelvis collider / seat plane error (cm)'),
              ('pelvis_position_error_cm', 'Pelvis position in odometry frame (cm)'),
              ('seat_plane_height_error_cm', 'Seat plane height error (cm)'),
              ('lateral_offset_error_cm', 'Lateral offset error (cm)'),
              ('yaw_axis_error_deg', 'Seat-axis yaw error modulo 180 degrees'),
              ('sigma_cm', 'Position uncertainty budget (cm; heuristic)')]
    for ax, (key, title) in zip(axes.ravel(), panels):
        for i in range(count):
            values = [m for m in metrics if m['case'] == i]
            ax.plot([m['stamp'] for m in values], [m.get(key, np.nan) for m in values],
                    color=colors[i], label=cfg['geometry_bank'][i]['family']+' '+cfg['geometry_bank'][i]['shape'])
        ax.set_title(title); ax.set_xlabel('Sensor time (s)'); ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=7); fig.savefig(out/'errors.png', dpi=140); plt.close(fig)
    if a.video:
        render_video(out, cfg, sensor, tracks, estimates, metrics, alignments, truth)


def draw_axes(ax, T, length=.10):
    for j, color in enumerate(('tomato', 'limegreen', 'royalblue')):
        p = np.stack((T[:3, 3], T[:3, 3]+length*T[:3, j]))
        ax.plot(*p.T, color=color, lw=2.5)


def render_video(out, cfg, sensor, tracks, estimates, metrics, alignments, truth):
    frames, count = tracks['T_O_P'].shape[:2]
    fig = plt.figure(figsize=(13.6, 7.6), dpi=100)
    writer = imageio.get_writer(out/'shared_frame.mp4', fps=10, codec='libx264', quality=7,
                                macro_block_size=2)
    # Four seconds per case, sampled over the full captured motion; labels show time.
    for i in range(count):
        for k in np.linspace(0, frames-1, min(48, frames)).astype(int):
            fig.clear(); ax = fig.add_subplot(1, 2, 1, projection='3d')
            ax.set_facecolor('#f7f9fc')
            points = tracks['cloud'][k, i]; points = points[np.isfinite(points).all(1)]
            ax.scatter(*points.T, c=points[:, 2], cmap='viridis', s=1.4, alpha=.40, vmin=0., vmax=.7)
            cells = tracks['cells'][k, i]; cells = cells[np.isfinite(cells).all(1)]
            ax.scatter(*cells.T, color='#ed9d26', marker='s', s=4, alpha=.5)
            ax.add_collection3d(Line3DCollection(tracks['skeleton'][k, i], colors='#192940', linewidths=2.))
            proxy = tracks['proxy'][k, i]; proxy = proxy[np.isfinite(proxy).all(1)]
            ax.scatter(*proxy.T, color='#d63c79', s=2.5, alpha=.65)
            T = tracks['T_O_P'][k, i]; S = tracks['T_O_S'][k, i]
            draw_axes(ax, T); draw_axes(ax, S)
            # Floor grid in O; the actual estimated coefficients are logged per frame.
            g = np.linspace(-.9, .4, 8); h = np.linspace(-.7, .7, 8)
            floor = tracks['floor'][k, i]
            for x in g:
                z = -(floor[0]*x+floor[1]*h+floor[3])/floor[2]
                ax.plot(np.full_like(h, x), h, z, color='#aaa', alpha=.35, lw=.5)
            ax.set(xlim=(-1., .4), ylim=(-.8, .8), zlim=(0., 1.45), xlabel='O x (m)', ylabel='O y (m)', zlabel='O z (m)')
            ax.set_box_aspect((1.4, 1.6, 1.45)); ax.view_init(22, -57)
            ax.set_title('Estimated shared 3D frame\ncloud + observed support + encoder skeleton')
            image_ax = fig.add_axes([.58, .49, .35, .36])
            image_ax.imshow(sensor['rgb'][k, i]); image_ax.axis('off'); image_ax.set_title('Onboard rear/down RGB-D view')
            row = estimates[k*count+i]; err = metrics[k*count+i]
            yaw = np.rad2deg(row.get('yaw_axis_error_rad', np.nan))
            details = '\n'.join([
                f"{cfg['geometry_bank'][i]['family']} / {cfg['geometry_bank'][i]['shape']}     sensor t = {row['stamp']:.1f} s",
                f"Track: {row['status']}   valid: {row['valid']}   ICP rank: {row['registration']['rank']}/4",
                f"Pelvis collider clearance: {100*row.get('clearance_m', np.nan):.1f} cm",
                f"Lateral / forward offset: {100*row.get('pelvis_offset_m', [np.nan]*3)[1]:.1f} / {100*row.get('pelvis_offset_m', [np.nan]*3)[0]:.1f} cm",
                f"Pelvis-to-seat yaw axis: {yaw:.1f} deg (180 deg ambiguity)",
                f"Offline clearance error: {err.get('clearance_error_cm', np.nan):.2f} cm",
                f"Offline pelvis position error: {err['pelvis_position_error_cm']:.2f} cm",
                'Colored points: depth; orange: observed seat; magenta: body proxy',
                'Simulator truth is used only for the error labels.'])
            fig.text(.55, .40, details, va='top', fontsize=10, linespacing=1.6, family='monospace')
            fig.suptitle('G1 shared-frame perception — frozen controller, no training', fontsize=16)
            fig.text(.5, .925, 'Selected frames; each 16 s trajectory plays in approximately 4.8 s', ha='center', fontsize=10)
            fig.canvas.draw()
            rgb = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
            writer.append_data(rgb)
            if k == 0:
                fig.savefig(out/f'case_{i:02d}.png', dpi=140)
            if k == np.linspace(0, frames-1, min(48, frames)).astype(int)[min(48, frames)//2]:
                fig.savefig(out/f'case_{i:02d}_mid.png', dpi=140)
            if k == frames-1:
                fig.savefig(out/f'case_{i:02d}_end.png', dpi=140)
    writer.close(); plt.close(fig)


if __name__ == '__main__':
    main()
