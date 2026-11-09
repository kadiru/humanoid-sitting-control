"""Visualize sensor-only body/scene reconstruction; truth only on accuracy plot."""
import argparse,json,shutil,subprocess
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from humanoid_sitting.support_landing_geometry import extract_scene
from humanoid_sitting.shared_frame import BodyModel,apply,level_rotation
p=argparse.ArgumentParser();p.add_argument('--tracking',required=True);p.add_argument('--output',required=True);p.add_argument('--case',type=int,default=9);p.add_argument('--start',type=float,default=9.);p.add_argument('--end',type=float,default=15.);a=p.parse_args()
out=Path(a.output);out.mkdir(exist_ok=False);track=Path(a.tracking);m=json.loads((track/'manifest.json').read_text());cap=Path(m['capture'])
def load(path):
 with np.load(path) as f:return {k:f[k] for k in f.files}
s=load(cap/'proprioception.npz');d=load(cap/'depth.npz');z=load(track/'tracking.npz');truth=load(cap/'evaluation_labels.npz')
cal=json.loads((cap/'calibration.json').read_text());c=cal['cases'][str(a.case)];model=BodyModel(cap/'model.json');geometry=json.loads((cap/'model.json').read_text())
records_path=cap/'shadow_records.jsonl' if m.get('online') else track/'records.jsonl'
records={round(r['stamp']/.02):r for line in records_path.read_text().splitlines() if (r:=json.loads(line))['case']==a.case}
frames=np.flatnonzero((d['case']==a.case)&(d['t']>=a.start-1e-6)&(d['t']<=a.end+1e-6));camera_t=d['t'][d['case']==a.case];legacy=d['legacy'][d['case']==a.case]
plot_values=truth['true_clearance'][(s['t']>=a.start)&(s['t']<=a.end),a.case]*100
ylim=(min(-2,float(plot_values.min())-1),max(12,float(plot_values.max())+1))
row=json.loads((cap/'results.json').read_text())['episodes'][a.case]
outcome='successful trial' if row['success'] else 'failed trial: '+', '.join(k for k,v in row['failures'].items() if v)
fig=plt.figure(figsize=(13,7));writer=FFMpegWriter(fps=10,codec='libx264',bitrate=2400)
saved_poster=False;saved_mapped=False
with writer.saving(fig,str(out/'landing_tracking.mp4'),100):
 for frame in frames:
    t=float(d['t'][frame]);k=round(t/.02)-1;r=records[k+1];R=level_rotation(s['gravity'][k,a.case]);poses=model.fk(s['q'][k,a.case])
    scene=extract_scene(model,np.array(c['K']),np.array(c['T_P_C']),d['depth'][frame],s['q'][k,a.case],s['gravity'][k,a.case],poses,R)
    fig.clear();grid=fig.add_gridspec(2,2,width_ratios=[1.1,1],height_ratios=[1,.34],left=.02,right=.96,top=.86,bottom=.07,wspace=.2)
    ax=fig.add_subplot(grid[:,0],projection='3d');points=scene['points'][::4]
    ax.scatter(*points.T,c=points[:,2],cmap='viridis',s=1,alpha=.5,vmin=-.75,vmax=-.08)
    lookup={name:T[:3,3]@R.T for name,T in poses.items()}
    for j in geometry['joints']:
        if j['body0'] and j['body0'][0] in lookup and j['body1'][0] in lookup:
            v=np.array([lookup[j['body0'][0]],lookup[j['body1'][0]]]);ax.plot(*v.T,color='#1f2937',lw=2.2)
    pelvis=model.vertices['pelvis']@R.T;ax.scatter(*pelvis.T,c='#d97706',s=8);ax.scatter(0,0,0,c='#dc2626',s=35)
    if r['valid']:
        n=np.array(r['normal_P']);v=model.vertices['pelvis'][np.argmin(model.vertices['pelvis']@n)];offset=r['clearance_m']-v@n
        target=v-r['clearance_m']*n;line=np.array([v,target])@R.T;ax.plot(*line.T,c='#e11d48',lw=3)
        nL=R@n;x,y=np.meshgrid([-.7,.15],[-.45,.45]);h=-(offset+nL[0]*x+nL[1]*y)/nL[2]
        ax.plot_surface(x,y,h,color='#10b981',alpha=.18)
    ax.set(xlim=(-1.,.5),ylim=(-.65,.65),zlim=(-.8,.75),xlabel='Forward (m)',ylabel='Sideways (m)',zlabel='Height relative to pelvis (m)')
    ax.set_box_aspect([1.5,1.3,1.55]);ax.view_init(elev=15,azim=-60);ax.set_title('Depth cloud + body from joint kinematics',fontsize=11)
    ax=fig.add_subplot(grid[0,1]);valid=z['valid'][:,a.case]
    ax.plot(s['t'],truth['true_clearance'][:,a.case]*100,'--',color='#64748b',lw=1.3,label='Simulator truth (scoring only)')
    ax.plot(s['t'],np.where(valid,z['clearance'][:,a.case]*100,np.nan),color='#047857',lw=1.5,marker='.',markersize=2,label='New tracker at 50 Hz')
    ax.scatter(camera_t,np.where(legacy[:,8]>0,legacy[:,0]*100,np.nan),color='#c2410c',s=11,label='Original depth-only estimator')
    ax.axvline(t,color='black',lw=1);ax.axhline(0,color='#aaa',lw=.5);ax.set(xlim=(a.start,a.end),ylim=ylim,xlabel='Time (s)',ylabel='Pelvis collision clearance (cm)');ax.grid(alpha=.2);ax.legend(fontsize=8)
    ax=fig.add_subplot(grid[1,1]);ax.axis('off');value=f"{r['clearance_m']*100:.2f} cm" if r['valid'] else 'Uncertain — withheld'
    source={'current_floor_and_seat':'Fresh floor and seat','depth_with_kinematic_floor':'Seat + floor reference from joints','depth_with_temporal_identity':'Fresh seat + recent surface identity','kinematic_prediction':'Short bridge from joint motion','seat_from_observed_floor':'Visible floor + recently mapped seat height','unavailable':'No trustworthy current estimate'}[r['source']]
    ax.text(0,.95,f'Distance: {value}\nSource: {source}\nOrange dots disappear where the original estimator rejects depth.\nThe 3D view uses depth, encoders, calibration and IMU only.',va='top',fontsize=10,linespacing=1.6)
    fig.suptitle(f'Seat tracking | case {a.case} | {t:.1f} s | {outcome}\nGreen: local seating plane (display extent only); red: pelvis clearance',fontsize=14)
    if not saved_poster and ((not d['legacy'][frame,8] and r['valid']) or frame==frames[-1]):
        fig.savefig(out/'landing_tracking.png',dpi=130);saved_poster=True
    if not saved_mapped and r['source']=='seat_from_observed_floor':
        fig.savefig(out/'landing_tracking.png',dpi=130);saved_mapped=True
    writer.grab_frame()
plt.close(fig);shutil.copy2(__file__,out/'render_source.py')
(out/'selection.json').write_text(json.dumps(dict(capture=str(cap),tracking=str(track),case=a.case,interval_s=[a.start,a.end],truth_only_on_plot=True),indent=2)+'\n')
subprocess.run(['ffmpeg','-v','error','-i',str(out/'landing_tracking.mp4'),'-f','null','-'],check=True)
