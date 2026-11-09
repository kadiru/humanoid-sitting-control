"""Visualize the recorded shape-association failure using estimated geometry only."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from humanoid_sitting.shared_frame import support_patch

root=Path(__file__).resolve().parents[1]
out=root/'runs/shared_frame_component_diagnostic_01';out.mkdir(exist_ok=False)
with np.load(root/'runs/shared_frame_ellipse_diagnostic_01.npz') as ar:
    d={k:ar[k] for k in ar.files}
near=cKDTree(d['points'][:,:2]).query(d['edges'][:,:2])[0]<.03
plane=dict(points=d['points'],normal=np.array([0.,0.,1.]),offset=-float(d['T'][2,3]))
patch=support_patch(plane,contour=d['edges'])
old_yaw=float(np.rad2deg(np.arctan2(d['T'][1,0],d['T'][0,0])))
new_yaw=float(np.rad2deg(np.arctan2(patch['T'][1,0],patch['T'][0,0])))
report=dict(frame=20,case=5,source_capture='shared_frame_rear_down_six_capture_01',
    contour_points=len(near),associated_contours=int(near.sum()),unrelated_contours=int((~near).sum()),
    before_axis_deg=old_yaw,after_axis_deg=new_yaw,after_method=patch['axis_method'],
    after_ellipse_rms_mm=1000*patch['curved_fit']['rms_m'],
    scope='Actual stored estimated points/edges at t=2.0 s. No truth or semantic furniture labels used for association.')
(out/'diagnostic.json').write_text(json.dumps(report,indent=2)+'\n')
fig,axs=plt.subplots(1,2,figsize=(12,5.6),constrained_layout=True)
for ax,title,T in zip(axs,['Before: all contours at seat height','After: contours associated with selected support'],[d['T'],patch['T']]):
    ax.scatter(*d['points'][:,:2].T,s=4,color='#a6b5c1',label='Observed support')
    ax.scatter(*d['edges'][near,:2].T,s=7,color='#d99025',label='Associated contours')
    ax.scatter(*d['edges'][~near,:2].T,s=25,color='#bf4545',marker='x',label='Unrelated contours')
    for j,col in [(0,'#d54650'),(1,'#158577')]:
        origin=T[:2,3];target=origin+.16*T[:2,j]
        ax.annotate('',xy=target,xytext=origin,arrowprops=dict(arrowstyle='->',lw=2,color=col))
    ax.set(xlim=(-.75,.16),ylim=(-.51,.51),xlabel='O x (m)',ylabel='O y (m)',title=title)
    ax.set_aspect('equal');ax.grid(alpha=.2)
axs[0].text(.03,.97,f'Fallback rectangle axis: {old_yaw:.2f}°',transform=axs[0].transAxes,va='top',fontsize=10)
axs[1].text(.03,.97,f'Ellipse axis: {new_yaw:.2f}°; fit RMS {report["after_ellipse_rms_mm"]:.2f} mm',transform=axs[1].transAxes,va='top',fontsize=10)
axs[1].legend(loc='lower right',fontsize=8)
fig.suptitle(f'{report["unrelated_contours"]} unrelated depth edges caused a valid ellipse fit to be rejected',fontsize=14)
fig.savefig(out/'contour_association.png',dpi=150);plt.close(fig)
print(json.dumps(report,indent=2))
