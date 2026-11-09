"""Show sensor-estimated body/support/wall junctions with offline errors."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import imageio.v2 as imageio
from evaluate_shared_frame import draw_axes

root=Path(__file__).resolve().parents[1]
out=root/'runs/shared_frame_junction_visuals_02';out.mkdir(exist_ok=False)
fig=plt.figure(figsize=(13.6,8.4),dpi=100)
writer=imageio.get_writer(out/'seat_wall_junctions.mp4',fps=10,codec='libx264',quality=7,macro_block_size=2)
examples=[('Existing scene','shared_frame_rear_down_six_capture_01','shared_frame_junction_rear_clean_01'),
          ('Fresh geometry','shared_frame_junction_fresh_capture_01','shared_frame_junction_fresh_clean_01')]
for label,capture_name,run_name in examples:
    capture=root/'runs'/capture_name;run=root/'runs'/run_name
    cfg=json.loads((capture/'manifest.json').read_text())
    with np.load(run/'tracks.npz') as ar:tr={k:ar[k] for k in ar.files}
    with np.load(capture/'sensors.npz') as ar:depth=ar['depth']
    rows=json.loads((run/'estimates.json').read_text());errors=json.loads((run/'errors.json').read_text())
    for case in (0,3):
        rs=[r for r in rows if r['case']==case];es=[e for e in errors if e['case']==case]
        selected=np.linspace(0,len(rs)-1,60).astype(int);geometry=cfg['geometry_bank'][case]
        for k in selected:
            fig.clear();ax=fig.add_axes([.01,.30,.54,.61],projection='3d');ax.set_facecolor('#f6f8fb')
            for key,color,size in [('cloud',None,1.5),('cells','#ed9d26',6),('proxy','#d63c79',3),('junctions','#00a7b7',15)]:
                points=tr[key][k,case];points=points[np.isfinite(points).all(1)]
                points=points[(points[:,0]>=-1)&(points[:,0]<=.4)&(abs(points[:,1])<=.8)&(points[:,2]>=-.03)&(points[:,2]<=1.45)]
                if len(points):
                    if color:ax.scatter(*points.T,color=color,s=size,alpha=.95 if key=='junctions' else .5)
                    else:ax.scatter(*points.T,c=points[:,2],cmap='viridis',s=size,alpha=.35,vmin=0,vmax=.7)
            ax.add_collection3d(Line3DCollection(tr['skeleton'][k,case],colors='#192a40',linewidths=2))
            draw_axes(ax,tr['T_O_P'][k,case]);draw_axes(ax,tr['T_O_S'][k,case])
            floor=tr['floor'][k,case];yy=np.linspace(-.7,.7,8)
            for x in np.linspace(-.9,.3,8):
                zz=-(floor[0]*x+floor[1]*yy+floor[3])/floor[2]
                ax.plot(np.full_like(yy,x),yy,zz,color='#aaa',alpha=.4,lw=.5)
            ax.set(xlim=(-1,.4),ylim=(-.8,.8),zlim=(0,1.45),xlabel='O x (m)',ylabel='O y (m)',zlabel='O z (m)')
            ax.set_box_aspect((1.4,1.6,1.45));ax.view_init(22,-57)
            ax.set_title('Estimated common 3D frame — cyan: measured junctions',fontsize=10)
            cam=fig.add_axes([.60,.60,.30,.29]);cam.imshow(np.ma.masked_invalid(depth[k,case]),cmap='viridis',vmin=.1,vmax=1.6)
            cam.axis('off');cam.set_title('Observer axial depth',fontsize=10)
            row=rs[k];q=row['quality']['quantities']
            values=dict(clearance=None if row['direct_clearance_m'] is None else 100*row['direct_clearance_m'],
                forward=100*row['pelvis_offset_m'][0],lateral=100*row['pelvis_offset_m'][1],yaw=np.rad2deg(row['yaw_axis_error_rad']))
            for j,key in enumerate(('clearance','forward','lateral','yaw')):
                okay=q[key]['valid'];unit='deg' if key=='yaw' else 'cm'
                value=f'{values[key]:+.1f} {unit}' if okay else 'unconfirmed'
                fig.text(.57,.54-j*.047,key.capitalize()+': '+value,fontsize=12,color='#157249' if okay else '#a34727')
            fig.text(.57,.33,f"Observed wall junctions: {row['quality']['seat_boundary']['junction_count']}",fontsize=10)
            fig.text(.57,.30,f"Complete alignment: {sum(r['valid'] for r in rs)}/{len(rs)} frames in this clip",fontsize=10)
            fig.text(.57,.271,'Orange: observed support. Magenta: body collision geometry.',fontsize=9)
            plot=fig.add_axes([.08,.105,.83,.13]);t=np.array([r['stamp'] for r in rs])
            for key,color in [('clearance','#22805b'),('forward','#bd6436'),('lateral','#197a9d')]:
                value=np.array([100*abs(r['direct_clearance_m']-e['truth_clearance_m']) if key=='clearance' and r['direct_clearance_m'] is not None else e.get(key+'_offset_error_cm',np.nan) for r,e in zip(rs,es)])
                valid=np.array([r['quality']['quantities'][key]['valid'] for r in rs])
                plot.plot(t,value,color=color,alpha=.35,lw=1)
                plot.scatter(t[valid],value[valid],s=5,color=color,label=key+' accepted')
            plot.axvline(row['stamp'],color='#222',lw=1);plot.set_yscale('symlog',linthresh=.5)
            plot.axhline(3.,color='#bb4545',lw=.8,ls='--',label='3 cm offset limit')
            plot.set(xlabel='Sensor time (s)',ylabel='Offline error (cm)');plot.grid(alpha=.2);plot.legend(ncol=3,fontsize=8,loc='upper right')
            validation=' — bank validation FAILED' if label=='Fresh geometry' else ''
            fig.suptitle(f"{label}: {geometry['family']} — t = {row['stamp']:.1f} s{validation}",fontsize=14,y=.98)
            fig.text(.08,.033,'Text colors show recorded quality flags; the offline error plot audits their correctness. Faint curves include rejected estimates.',fontsize=8)
            fig.text(.08,.014,'Truth is used only for error labels. Scale is linear below 0.5 cm, logarithmic above; each 16 s sequence is shown in 6 s.',fontsize=8)
            fig.canvas.draw();writer.append_data(np.asarray(fig.canvas.buffer_rgba())[...,:3])
            if k==selected[30]:fig.savefig(out/f'{run_name}_case_{case}_mid.png',dpi=135)
writer.close();plt.close(fig)
print('JUNCTION_VIDEO_COMPLETE',out,flush=True)
