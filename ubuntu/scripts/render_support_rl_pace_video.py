"""Re-render the same selected cases with bounds covering the entire falling body."""
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
root=Path(__file__).resolve().parents[1];store=Path('/data/humanoid-sitting-perception');out=store/'support_rl_pace_results_01'
assert not (out/'sitting_comparison_full_view.mp4').exists()
shutil.copy2(__file__,out/'full_view_video_source.py')
def read(p):return json.loads(p.read_text())
report=read(out/'results.json');comparisons=report['comparisons'];summaries=report['summaries'];runs={}
for key,name in report['runs'].items():
 p=store/name;runs[key]=dict(result=read(p/'results.json'),trace=dict(np.load(p/'rollout.npz')),
     viz=read(p/'visualization_metadata.json'),bank=read(p/'geometry_bank.json'))
labels={'prior':'Previous PPO','candidate':'PPO + pace'};colors={'prior':'#c2410c','candidate':'#047857'}
bounds={}
for case in range(16):
 points=np.concatenate([(runs['fresh_'+k]['trace']['body'][:,case,:,:3]-np.asarray(runs['fresh_'+k]['viz']['origins'][case])).reshape(-1,3) for k in labels])
 lo=points.min(0)-.15;hi=points.max(0)+.15
 bounds[case]=[[float(min(-1.05,lo[0])),float(max(.5,hi[0]))],
               [float(min(-.7,lo[1])),float(max(.7,hi[1]))],[-.03,float(max(1.5,hi[2]))]]
(out/'full_view_bounds.json').write_text(json.dumps(bounds,indent=2)+'\n')

model=read(root/'runs/g1_kinematics_geometry_04/model.json')
def draw_robot(ax,run,frame,case):
    positions=run['trace']['body'][frame,case,:,:3]-np.asarray(run['viz']['origins'][case])
    lookup={n:positions[i] for i,n in enumerate(run['viz']['body_names'])}
    for j in model['joints']:
        if j['body0'] and j['body0'][0] in lookup and j['body1'][0] in lookup:
            points=np.array([lookup[j['body0'][0]],lookup[j['body1'][0]]]);ax.plot(*points.T,color='#263b50',lw=2)
    ax.scatter(*lookup['pelvis'],s=55,color='#f59e0b')
    g=run['bank'][run['viz']['geometry_indices'][case]];yaw=g['yaw']
    R=np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])
    xy=np.array([[-.5,-.5],[.5,-.5],[.5,.5],[-.5,.5]])*[g['depth'],g['width']]
    xy=xy@R.T+[g['x'],g['y']]
    ax.add_collection3d(Poly3DCollection([np.c_[xy,np.full(4,g['height'])]],facecolor='#10b981',alpha=.35,edgecolor='#047857'))
    if g['family'] in ('chair','rigid_couch'):
        back=np.array([[-g['depth']/2,-g['width']/2],[-g['depth']/2,g['width']/2]])@R.T+[g['x'],g['y']]
        wall=np.array([[*back[0],g['height']],[*back[1],g['height']],[*back[1],g['height']+.35],[*back[0],g['height']+.35]])
        ax.add_collection3d(Poly3DCollection([wall],facecolor='#10b981',alpha=.15))
    b=bounds[case]
    ax.set(xlim=b[0],ylim=b[1],zlim=b[2])
    ax.set_xticks(np.round(np.linspace(*b[0],3),1));ax.set_yticks(np.round(np.linspace(*b[1],3),1));ax.set_zticks([0.,.75,1.5])
    ax.set_box_aspect([hi-lo for lo,hi in b]);ax.view_init(elev=18,azim=-65)

comp=comparisons['fresh']
first=(comp['lost_cases'] or summaries['fresh_candidate']['failed_cases'] or [0])[0]
other=[i for i,r in enumerate(runs['fresh_candidate']['result']['episodes']) if r['success'] and i!=first]
second=(comp['gained_cases'] or other or [i for i in range(16) if i!=first])[0]
clips=[first,second];(out/'full_view_clip_selection.json').write_text(json.dumps(dict(bank='fresh',cases=clips,
    selection='first lost/failed case, then first gained/successful other case'),indent=2)+'\n')
fig=plt.figure(figsize=(13,8));writer=FFMpegWriter(fps=10,codec='libx264',bitrate=2000)
with writer.saving(fig,str(out/'sitting_comparison_full_view.mp4'),100):
    for clip,case in enumerate(clips):
        for frame in range(0,200,2):
            fig.clear();grid=fig.add_gridspec(2,2,height_ratios=[1.4,1],hspace=.35,wspace=.3,top=.82,bottom=.08,left=.07,right=.95)
            for col,which in enumerate(['prior','candidate']):
                run=runs['fresh_'+which];z=run['trace'];row=run['result']['episodes'][case];t=float(z['t'][frame])
                ax=fig.add_subplot(grid[0,col],projection='3d');draw_robot(ax,run,frame,case)
                reasons=[k for k,v in row['failures'].items() if v]
                if not row['standing']:reasons.append('standing check')
                if row['hold_s']<3.:reasons.append('short supported hold')
                status='PASS' if row['success'] else 'FAIL: '+', '.join(reasons)
                ax.set_title(f'{labels[which]} | case {case} | {t:.1f}s\n{status}',fontsize=10)
                ax=fig.add_subplot(grid[1,col]);ax.plot(z['t'],z['force'][:,case],color=colors[which],label='Sampled force')
                ax.axhline(632.527832,color='#b91c1c',ls='--',label='Impact threshold');ax.axvline(t,color='black',lw=1)
                ax.axhline(row['peak_force_N'],color=colors[which],ls=':',label='Full-trial peak (50 Hz)')
                ymax=max(750,max(runs['fresh_'+w]['result']['episodes'][case]['peak_force_N'] for w in labels)*1.08)
                ax.set(xlim=(0,20),ylim=(0,ymax),xlabel='Time (s)',ylabel='Seat force (N)');ax.grid(alpha=.2)
                live=z['live'][frame,case];clearance=f'{live[0]*100:.1f} cm' if live[8] else 'unavailable'
                ax.set_title(f'Measured clearance: {clearance} | motion pace: {z["command_pace"][frame,case]:.2f}\nLeg torque: {z["torque"][frame,case]:.1f} Nm | full peak: {row["peak_force_N"]:.0f} N',fontsize=10)
                if col==0:ax.legend(fontsize=8)
            fig.suptitle('New wider holdout: previous PPO versus learned motion pace\nSimulator reconstruction for evaluation only · 2x playback\nForce curve sampled at 10 Hz; dotted line shows full-trial peak',fontsize=12)
            if frame==130:fig.savefig(out/f'case_{clip}_full_view.png',dpi=120)
            writer.grab_frame()
plt.close(fig)
subprocess.run(['ffmpeg','-v','error','-i',str(out/'sitting_comparison_full_view.mp4'),'-f','null','-'],check=True)
