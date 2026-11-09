"""Offline comparison: estimated common frames and audited validity, no runtime labels."""
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
out=Path('/data/humanoid-sitting-perception/shared_frame_integrity_visuals_03');out.mkdir(exist_ok=False)
fig=plt.figure(figsize=(14,9),dpi=100)
writer=imageio.get_writer(out/'map_integrity.mp4',fps=10,codec='libx264',quality=7,macro_block_size=2)
clips=[
 ('Plane identity after loss: ellipse / clean',5,'shared_frame_integrity_unseen_clean_02','shared_frame_integrity_unseen_clean_03','Previous candidate: plane identity failure','Corrected plane association'),
 ('Large pose correction: box / noisy',2,'shared_frame_integrity_unseen_noisy_02','shared_frame_integrity_unseen_noisy_03','Previous candidate: false alignment','Corrected recovery before map fusion'),
 ('Original failure: couch / clean',3,'shared_frame_junction_fresh_clean_01','shared_frame_integrity_fresh_clean_03','Original observer: map drift','Fixed reference and identity checks'),
 ('Untouched geometry: chair',0,'shared_frame_integrity_holdout_clean_03','shared_frame_integrity_holdout_noisy_03','Final observer / clean','Final observer / noisy')]
for clip,(title,case,left,right,left_title,right_title) in enumerate(clips):
 data=[]
 for name,label in ((left,left_title),(right,right_title)):
  folder=root/'runs'/name
  rows=[r for r in json.loads((folder/'estimates.json').read_text()) if r['case']==case]
  errors=[r for r in json.loads((folder/'errors.json').read_text()) if r['case']==case]
  with np.load(folder/'tracks.npz') as f:tracks={k:f[k] for k in f.files}
  data.append((label,rows,errors,tracks))
 selected=np.linspace(0,len(data[0][1])-1,60).astype(int)
 for k in selected:
  fig.clear()
  for column,(label,rows,errors,tr) in enumerate(data):
   row=rows[k];x0=.01+column*.50
   ax=fig.add_axes([x0,.35,.48,.54],projection='3d');ax.set_facecolor('#f5f7fa')
   for key,color,size in [('cloud',None,1.1),('cells','#e39b29',5),('proxy','#d83b73',3),('junctions','#009eaa',12)]:
    p=tr[key][k,case];p=p[np.isfinite(p).all(1)]
    p=p[(p[:,0]>-1)&(p[:,0]<.4)&(abs(p[:,1])<.8)&(p[:,2]>-.03)&(p[:,2]<1.4)]
    if len(p):
     if color:ax.scatter(*p.T,color=color,s=size,alpha=.6)
     else:ax.scatter(*p.T,c=p[:,2],cmap='viridis',vmin=0,vmax=.7,s=size,alpha=.25)
   ax.add_collection3d(Line3DCollection(tr['skeleton'][k,case],colors='#21334d',linewidths=2))
   draw_axes(ax,tr['T_O_P'][k,case]);draw_axes(ax,tr['T_O_S'][k,case])
   plane=tr['floor'][k,case];yy=np.linspace(-.7,.7,8)
   for xx in np.linspace(-.9,.3,8):
    zz=-(plane[0]*xx+plane[1]*yy+plane[3])/plane[2]
    ax.plot(np.full_like(yy,xx),yy,zz,color='#aab',lw=.6,alpha=.4)
   ax.set(xlim=(-1,.4),ylim=(-.8,.8),zlim=(0,1.4),xlabel='O x (m)',ylabel='O y (m)',zlabel='O z (m)')
   ax.set_box_aspect((1.4,1.6,1.4));ax.view_init(23,-57);ax.set_title(label,fontsize=13,pad=1)
   status='Alignment available' if row['valid'] else 'Alignment withheld / prediction only'
   fig.text(x0+.05,.335,status,color='#16734b' if row['valid'] else '#b24632',fontsize=12)
   q=row['quality']['quantities'];parts=[]
   for key in ('clearance','forward','lateral','yaw'):
    if not q[key]['valid']:value='unknown'
    elif key=='clearance':value=f"{100*row['direct_clearance_m']:+.1f} cm"
    elif key=='yaw':value=f"{np.rad2deg(row['yaw_axis_error_rad']):+.1f} deg"
    else:value=f"{100*row['pelvis_offset_m'][0 if key=='forward' else 1]:+.1f} cm"
    parts.append(key+': '+value)
   fig.text(x0+.05,.307,'   '.join(parts[:2]),fontsize=10)
   fig.text(x0+.05,.284,'   '.join(parts[2:]),fontsize=10)
   plot=fig.add_axes([x0+.055,.10,.39,.14]);t=np.array([r['stamp'] for r in rows])
   for key,color in [('clearance','#23875a'),('forward','#b66d2c'),('lateral','#247ca4')]:
    values=np.array([100*abs(r['direct_clearance_m']-e['truth_clearance_m']) if key=='clearance' and r.get('direct_clearance_m') is not None else e.get(key+'_offset_error_cm',np.nan) for r,e in zip(rows,errors)]);valid=np.array([r['quality']['quantities'][key]['valid'] for r in rows])
    plot.plot(t,values,color=color,alpha=.22,lw=1)
    plot.scatter(t[valid],values[valid],color=color,s=7,label=key+' accepted')
   plot.axhline(2,color='#23875a',ls=':',lw=.8);plot.axhline(3,color='#c34845',ls='--',lw=1);plot.axvline(row['stamp'],color='#333',lw=1)
   plot.set_yscale('symlog',linthresh=.5);plot.set(xlim=(0,16),ylim=(0,100),xlabel='Sensor time (s)',ylabel='Offline error (cm)')
   plot.legend(loc='upper left',fontsize=7,ncol=3);plot.grid(alpha=.15)
  fig.suptitle(f"{title}  |  t = {data[0][1][k]['stamp']:.1f} s",fontsize=17,y=.975)
  fig.text(.5,.93,'Depth cloud + floor + observed seat (orange) + body geometry (pink) + junctions (cyan)',ha='center',fontsize=10)
  fig.text(.06,.040,'Dots: accepted estimates. Faint curves include rejected estimates; unavailable clearance is omitted. Limits: dotted green 2 cm clearance; dashed red 3 cm offsets.',fontsize=9)
  fig.text(.06,.018,'Simulator truth appears only in the offline error plots. Each 16 s trajectory plays in 6 s. Axis is linear below 0.5 cm and logarithmic above.',fontsize=9)
  fig.canvas.draw();writer.append_data(np.asarray(fig.canvas.buffer_rgba())[...,:3])
  if k==selected[40]:fig.savefig(out/f'comparison_{clip}.png',dpi=125)
writer.close();plt.close(fig)
print('MAP_INTEGRITY_VIDEO_COMPLETE',out,flush=True)
