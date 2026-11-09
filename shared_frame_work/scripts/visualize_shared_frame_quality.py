"""Render measurement-specific confidence with offline errors labelled separately."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import imageio.v2 as imageio
from evaluate_shared_frame import draw_axes


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output', required=True)
    parser.add_argument('--include-camera-probe', action='store_true')
    a = parser.parse_args(); out = Path(a.output); out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    examples = [('shared_frame_capture_01', 'shared_frame_quality_clean_04', 1,
                 'Physics recording: bench descent'),
                ('shared_frame_stress_capture_01', 'shared_frame_stress_04', 2,
                 'Analytical fixture: 12 cm slide during depth blackout')]
    if a.include_camera_probe:
        examples.append(('shared_frame_rear_down_capture_02','shared_frame_rear_down_02',0,
                         'Physics recording: rear vertical camera, bench descent'))
    fig = plt.figure(figsize=(13.6, 8.2), dpi=100)
    writer = imageio.get_writer(out/'measurement_quality.mp4', fps=10, codec='libx264', quality=7, macro_block_size=2)
    for scene,run,case,title in examples:
        with np.load(root/'runs'/run/'tracks.npz') as ar:
            tr = {k:ar[k] for k in ar.files}
        with np.load(root/'runs'/scene/'sensors.npz') as ar:
            rgb = ar['rgb'][:,case]
            fallback_depth = ar['depth'][:,case] if not np.any(rgb) else None
        rows = [r for r in json.loads((root/'runs'/run/'estimates.json').read_text()) if r['case']==case]
        errors = [r for r in json.loads((root/'runs'/run/'errors.json').read_text()) if r['case']==case]
        t = [r['stamp'] for r in rows]
        direct = [100*abs(r['direct_clearance_m']-e['truth_clearance_m']) if r['direct_clearance_m'] is not None else np.nan
                  for r,e in zip(rows,errors)]
        for k in np.linspace(0,len(rows)-1,80).astype(int):
            fig.clear(); ax=fig.add_axes([.02,.30,.52,.61],projection='3d')
            ax.set_facecolor('#f6f8fb')
            cloud=tr['cloud'][k,case]; cloud=cloud[np.isfinite(cloud).all(1)]
            if len(cloud): ax.scatter(*cloud.T,c=cloud[:,2],cmap='viridis',s=1.7,alpha=.35,vmin=0,vmax=.7)
            cells=tr['cells'][k,case]; cells=cells[np.isfinite(cells).all(1)]
            ax.scatter(*cells.T,color='#ed9d26',s=7,marker='s',alpha=.55)
            ax.add_collection3d(Line3DCollection(tr['skeleton'][k,case],colors='#1a2a40',linewidths=2.))
            proxy=tr['proxy'][k,case]; proxy=proxy[np.isfinite(proxy).all(1)]
            ax.scatter(*proxy.T,color='#d63c79',s=3,alpha=.7)
            draw_axes(ax,tr['T_O_P'][k,case]); draw_axes(ax,tr['T_O_S'][k,case])
            ax.set(xlim=(-1.,.4),ylim=(-.8,.8),zlim=(0,1.45),xlabel='O x (m)',ylabel='O y (m)',zlabel='O z (m)')
            ax.set_box_aspect((1.4,1.6,1.45)); ax.view_init(22,-57)
            row=rows[k]; quality=row['quality']
            ax.set_title('Estimated cloud, body, and observed support',fontsize=11)
            cam=fig.add_axes([.59,.58,.30,.31])
            if fallback_depth is not None:
                cam.imshow(np.ma.masked_invalid(fallback_depth[k]),cmap='viridis',vmin=.1,vmax=1.6)
                cam.set_title('Axial depth (RGB unavailable in this capture)',fontsize=10)
            else:
                cam.imshow(rgb[k])
                cam.set_title('Analytical axial-depth image' if scene=='shared_frame_stress_capture_01' else 'Camera RGB',fontsize=10)
            cam.axis('off')
            for j,(key,label) in enumerate([('clearance','Clearance'),('forward','Forward center'),('lateral','Lateral center'),('yaw','Yaw axis')]):
                q=quality['quantities'][key]; okay=q['valid']
                value='measured' if okay else 'unconfirmed'
                if key=='clearance' and okay: value=f"{100*row['direct_clearance_m']:.1f} cm measured"
                fig.text(.57,.53-j*.046,f'{label}: {value}',fontsize=12,color='#157249' if okay else '#a34727')
            recovery=row['registration'].get('recovery','')
            fig.text(.57,.315,'Map recovery: '+(recovery or ('required' if row['registration'].get('reacquisition_required') else 'tracking')),fontsize=10)
            fig.text(.57,.285,'Flags require visible constraints; foot agreement alone is insufficient.',fontsize=9,color='#444')
            plot=fig.add_axes([.08,.075,.83,.16])
            plot.plot(t,[e['lateral_offset_error_cm'] for e in errors],label='Lateral error',color='#167a9e')
            plot.plot(t,direct,label='Current-plane clearance error',color='#24805c')
            plot.axvline(row['stamp'],color='#111',lw=1)
            if scene=='shared_frame_stress_capture_01': plot.axvspan(2.,4.5,color='#aaa',alpha=.2,label='Depth unavailable')
            plot.set(xlabel='Sensor time (s)',ylabel='Offline error (cm)',ylim=(-.2,13 if case==2 else 1.))
            plot.grid(alpha=.2); plot.legend(fontsize=8,loc='upper left',ncol=3)
            fig.suptitle(f'{title}     t = {row["stamp"]:.1f} s',fontsize=15,y=.975)
            fig.text(.08,.012,'Truth is used only for the error plot. Each full sequence is shown in 8 s; sensor time is explicit.',fontsize=9,color='#555')
            fig.canvas.draw(); writer.append_data(np.asarray(fig.canvas.buffer_rgba())[...,:3])
            if (scene=='shared_frame_capture_01' and k==80) or (scene!='shared_frame_capture_01' and k in (35,46)):
                fig.savefig(out/f'{run}_frame_{k:03d}.png',dpi=135)
    writer.close(); plt.close(fig)
    print('QUALITY_VIDEO_COMPLETE',out,flush=True)


if __name__=='__main__': main()
