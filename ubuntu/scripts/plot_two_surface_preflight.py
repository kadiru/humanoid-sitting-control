from pathlib import Path
import json
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
r=Path('/data/humanoid-sitting-perception/two_surface_curriculum_01');out=r/'preflight_visuals_02';out.mkdir(exist_ok=False)
d=json.loads((r/'selected_pose_validation_01/results.json').read_text());t=np.load(r/'selected_pose_validation_01/trace.npz')
chains=[['pelvis','torso_link'],*[['pelvis']+[s+x for x in ['_hip_pitch_link','_hip_roll_link','_hip_yaw_link','_knee_link','_ankle_pitch_link','_ankle_roll_link']] for s in ['left','right']],*[[ 'torso_link']+[s+x for x in ['_shoulder_pitch_link','_shoulder_roll_link','_shoulder_yaw_link','_elbow_pitch_link','_elbow_roll_link']] for s in ['left','right']]]
fig,axes=plt.subplots(2,7,figsize=(16,6),sharex=True,sharey=True)
for i,row in enumerate(d['rows']):
 ax=axes[i//7,i%7];g=row['geometry'];xyz=t['body'][-1,i,:,:3]-np.array(d['origins'][i])
 for chain in chains:
  ids=[d['body_names'].index(n) for n in chain if n in d['body_names']]
  ax.plot(xyz[ids,0],xyz[ids,2],'-o',color='#267c92',markersize=2,lw=2)
 center=np.array(g['top_center']);u=Rotation.from_quat(g['rotation_xyzw']).as_matrix()[:,0];edge=center[None]+np.array([-.5,.5])[:,None]*g['depth']*u
 ax.plot(edge[:,0],edge[:,2],lw=6,color='#cb8c37');ax.axhline(0,color='#747b7e',lw=1)
 ax.set_xlim(-.6,.35);ax.set_ylim(-.04,1.22);ax.set_aspect('equal');ax.set_title(f'Stage {i%7}\n'+('supported' if row['supported'] else 'feet-supported'),fontsize=10)
 ax.spines[['top','right']].set_visible(False)
 if i//7==1:ax.set_xlabel('Forward position (m)',fontsize=8)
 if i%7==0:ax.set_ylabel(('40 cm seat' if i<7 else '45° leaning surface')+'\nHeight (m)')
fig.suptitle('14/14 restored poses stable for 20 seconds — movement between poses is a separate test',fontsize=14)
fig.tight_layout();fig.savefig(out/'validated_waypoints.png',dpi=180)
b=json.loads((r/'reference_motion_gpu0_01/results.json').read_text());grid=np.array([x['success'] for x in b['episodes']]).reshape(2,7)
fig,ax=plt.subplots(figsize=(10,2.7));ax.imshow(grid,cmap=matplotlib.colors.ListedColormap(['#dfb09a','#91c7b1']),vmin=0,vmax=1,aspect='auto')
for i in range(2):
 for j in range(7):ax.text(j,i,'reached\nsupport' if grid[i,j] else 'failed\ngoal',ha='center',va='center',fontsize=10)
ax.set_xticks(range(7),['0: standing','1','2','3','4','5','6: supported']);ax.set_yticks([0,1],['40 cm seat','45° lean']);ax.set_title('Reference-only motion before PPO: 9/14 reach stable support\nEach rollout runs 20 s; final 5 s must continuously qualify')
fig.tight_layout();fig.savefig(out/'reference_before_ppo.png',dpi=180)
(out/'scope.txt').write_text('Pose stability and transition success are different. All14 poses passed independent stationary retention. The reference motion initially passes9/14 starting stages; no PPO result is shown here.')
print(str(out))
