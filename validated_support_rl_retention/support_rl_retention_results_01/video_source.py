"""Full-body pose reconstruction and support traces for the retention study."""
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
root=Path(__file__).resolve().parents[1];store=Path('/data/humanoid-sitting-perception')
out=store/'support_rl_retention_results_01';assert not (out/'sitting_comparison.mp4').exists()
read=lambda p:json.loads(p.read_text())
report=read(out/'results.json');runs={}
for arm in ['parent','control','anchored']:
    p=store/report['runs']['easy_'+arm]
    runs[arm]=dict(trace=dict(np.load(p/'rollout.npz')),rows=read(p/'results.json')['episodes'],
        viz=read(p/'visualization_metadata.json'),bank=read(p/'geometry_bank.json'))
loss=report['comparisons']['easy_anchored']['lost_cases'] or report['comparisons']['easy_control']['lost_cases']
first=(loss or [0])[0]
common=[i for i in range(16) if i!=first and all(r['rows'][i]['success'] for r in runs.values())]
clips=[first,(common or [i for i in range(16) if i!=first])[0]]
selection=dict(bank='known easier',cases=clips,selection='First anchored loss, else first control loss, else case 0; then first success shared by all three, else first other case.')
(out/'video_selection.json').write_text(json.dumps(selection,indent=2)+'\n')
shutil.copy2(__file__,out/'video_source.py')
model=read(root/'runs/g1_kinematics_geometry_04/model.json')
labels={'parent':'Preserved policy','control':'Continued PPO','anchored':'PPO + parent penalty'}
colors={'parent':'#64748b','control':'#c2410c','anchored':'#0f766e'}
bounds={}
for case in clips:
    points=np.concatenate([(r['trace']['body'][:,case,:,:3]-r['viz']['origins'][case]).reshape(-1,3) for r in runs.values()])
    lo=points.min(0)-.12;hi=points.max(0)+.12
    bounds[case]=[(min(-1.05,lo[0]),max(.5,hi[0])),(min(-.7,lo[1]),max(.7,hi[1])),(-.03,max(1.5,hi[2]))]
def robot(ax,r,frame,case):
    positions=r['trace']['body'][frame,case,:,:3]-r['viz']['origins'][case]
    lookup=dict(zip(r['viz']['body_names'],positions))
    for j in model['joints']:
        if j['body0'] and j['body0'][0] in lookup and j['body1'][0] in lookup:
            a=np.array([lookup[j['body0'][0]],lookup[j['body1'][0]]]);ax.plot(*a.T,color='#263b50',lw=2)
    ax.scatter(*lookup['pelvis'],color='#f59e0b',s=50)
    g=r['bank'][r['viz']['geometry_indices'][case]];c,s=np.cos(g['yaw']),np.sin(g['yaw'])
    xy=np.array([[-.5,-.5],[.5,-.5],[.5,.5],[-.5,.5]])*[g['depth'],g['width']]
    xy=xy@np.array([[c,s],[-s,c]])+[g['x'],g['y']]
    ax.add_collection3d(Poly3DCollection([np.c_[xy,np.full(4,g['height'])]],facecolor='#10b981',alpha=.3,edgecolor='#047857'))
    b=bounds[case];ax.set(xlim=b[0],ylim=b[1],zlim=b[2]);ax.set_box_aspect([hi-lo for lo,hi in b])
    ax.set_xticks([-.8,0,.5]);ax.set_yticks([-.5,0,.5]);ax.set_zticks([0,.75,1.5]);ax.view_init(elev=18,azim=-65)
fig=plt.figure(figsize=(15,8));writer=FFMpegWriter(fps=10,codec='libx264',bitrate=2300)
with writer.saving(fig,str(out/'sitting_comparison.mp4'),100):
    for clip,case in enumerate(clips):
        for frame in range(0,200,2):
            fig.clear();grid=fig.add_gridspec(2,3,height_ratios=[1.6,1],hspace=.4,wspace=.32,top=.84,bottom=.08,left=.06,right=.96)
            t=float(runs['parent']['trace']['t'][frame])
            for col,(arm,r) in enumerate(runs.items()):
                row=r['rows'][case];z=r['trace'];ax=fig.add_subplot(grid[0,col],projection='3d');robot(ax,r,frame,case)
                status='PASS' if row['success'] else 'FAIL: '+', '.join(k for k,v in row['failures'].items() if v)
                live=z['live'][frame,case];clearance=f'{live[0]*100:.1f} cm' if live[8] else 'unavailable'
                ax.set_title(f"{labels[arm]}\nFull trial: {status}\nDepth clearance: {clearance}",fontsize=10)
            for col,(field,title,unit) in enumerate([('force','Seat contact force','N'),('support','Seat-supported body weight','fraction'),('torque','Leg torque during movement','Nm RMS')]):
                ax=fig.add_subplot(grid[1,col])
                for arm,r in runs.items():
                    z=r['trace'];ax.plot(z['t'],z[field][:,case],color=colors[arm],label=labels[arm])
                    if field=='force':ax.axhline(r['rows'][case]['peak_force_N'],color=colors[arm],ls=':',lw=.8)
                if field=='force':ax.axhline(632.527832,color='#b91c1c',ls='--',lw=.8)
                ax.axvline(t,color='black',lw=1);ax.set(xlim=(0,20),xlabel='Time (s)',ylabel=unit,title=title);ax.grid(alpha=.2)
                if col==1:ax.legend(fontsize=8)
            fig.suptitle(f'Can further learning preserve a successful sit?  |  case {case}  |  {t:.1f} / 20 seconds\nKnown easier bank · simulator reconstruction for evaluation only · 2x playback\nCurves: 10 Hz samples; dotted force lines: full-trial peaks (50 Hz)',fontsize=13)
            if frame==130:fig.savefig(out/f'video_case_{clip}.png',dpi=110)
            writer.grab_frame()
plt.close(fig)
subprocess.run(['ffmpeg','-v','error','-i',str(out/'sitting_comparison.mp4'),'-f','null','-'],check=True)
