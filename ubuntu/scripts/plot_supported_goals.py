import json
from pathlib import Path
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from humanoid_sitting.shared_frame import BodyModel,apply
root=Path(__file__).resolve().parents[1];r=Path('/data/humanoid-sitting-perception/supported_surfaces_01/transition_goal_validation_01')
m=BodyModel(root/'runs/g1_kinematics_geometry_04/model.json');b=np.load(r/'candidates.npz');d=json.loads((r/'results.json').read_text())
fig,axs=plt.subplots(2,4,figsize=(15,10),sharey=True)
for ax,(directory,i) in zip(axs.flat,[(r,0),(r,2),(r,3),(r,6),(r,8),(r,12),(r.parent/'steep_validation_06',4),(r,17)]):
 db=json.loads((directory/'results.json').read_text());bb=np.load(directory/'candidates.npz');row=db['rows'][i];g=row['geometry'];root=bb['root_pose'][i];R=Rotation.from_quat(root[3:]).as_matrix();fk=m.fk(bb['q'][i])
 for name,T in fk.items():
  if name not in m.vertices:continue
  pts=(apply(T,m.vertices[name])@R.T+root[:3])[:,[0,2]]
  try:pts=pts[ConvexHull(pts).vertices]
  except Exception:continue
  ax.add_patch(Polygon(pts,color='#429eab' if name in ['pelvis','torso_link','logo_link'] else '#414858',alpha=.7))
 S=Rotation.from_quat(g['rotation_xyzw']).as_matrix();corners=np.array([[-g['depth']/2,0,0],[g['depth']/2,0,0],[g['depth']/2,0,-.1],[-g['depth']/2,0,-.1]])@S.T+g['top_center']
 ax.add_patch(Polygon(corners[:,[0,2]],color='#d4a44b',alpha=.9));ax.axhline(0,color='#85858e',lw=2)
 ax.set_aspect('equal');ax.set_xlim(-.95,.55);ax.set_ylim(-.03,1.45);ax.grid(alpha=.15);ax.spines[['top','right']].set_visible(False)
 ax.set_title(f"{g['kind'].capitalize()} • {g['pitch_deg']:g}°\nSurface center {100*g['height']:.1f} cm",fontsize=11)
 ax.set_xlabel('Forward / back (m)')
axs[0,0].set_ylabel('Height above floor (m)');axs[1,0].set_ylabel('Height above floor (m)')
fig.suptitle('Physically validated supported goals — each held for 20 seconds',fontsize=17)
fig.text(.5,.015,'Side-view collision geometry • rigid surfaces • these are endpoint tests, not learned transitions from standing',ha='center',fontsize=10)
fig.tight_layout(h_pad=3.,rect=[0,.04,1,.94]);fig.savefig(r/'supported_goals.png',dpi=180)
