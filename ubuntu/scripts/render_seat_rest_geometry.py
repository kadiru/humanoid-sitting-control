"""Offline model-mesh side view of validated static posture and old reference."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from humanoid_sitting.shared_frame import BodyModel

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
    root=Path('/data/humanoid-sitting-perception/seat_load_transfer_01');old=Path('/data/humanoid-sitting-perception/contact_effort_repair_01/sitting_preflight_01/reference')
    model=BodyModel('runs/g1_kinematics_geometry_04/model.json')
    paths=[old,root/'rest_replay_02'];cases=[8,4];titles=['Current standing-to-seat reference','Validated low-effort resting target']
    fig,axs=plt.subplots(1,2,figsize=(11,7.4))
    for ax,path,case,title in zip(axs,paths,cases,titles):
        result=json.loads((path/'results.json').read_text());t=np.load(path/'trace.npz');names=result['body_names'];origin=np.array(result['origins'][case]);body=t['body'][-1,case].copy();body[:,:3]-=origin
        for name in model.vertices:
            if name not in names:continue
            j=names.index(name);pts=model.vertices[name]@Rotation.from_quat(body[j,3:]).as_matrix().T+body[j,:3];xz=pts[:,[0,2]]
            if len(xz)<4:continue
            poly=xz[ConvexHull(xz).vertices];color='#dfa55a' if 'hip_roll_link' in name or 'hip_pitch_link' in name else '#409b78' if name=='pelvis' else '#6b9aad'
            ax.add_patch(Polygon(poly,facecolor=color,edgecolor='#31515c',alpha=.8,linewidth=.6))
        ax.add_patch(Polygon([[-.65,.3],[-.15,.3],[-.15,.4],[-.65,.4]],facecolor='#bd925b',edgecolor='#805d33'))
        ax.plot([-.65,.4],[0,0],color='#5f6a6b',lw=2);ax.set_xlim(-.48,.36);ax.set_ylim(-.02,1.25);ax.set_aspect('equal');ax.set_title(title,fontsize=13);ax.set_xlabel('Forward position (m)');ax.set_ylabel('Height above floor (m)');ax.grid(alpha=.15)
        effort=float((t['torque'][-50:,case]**2).sum(-1).mean())
        load=float(t['vertical_support'][-50:,case].mean())
        ax.text(.02,.98,f'Final5s torque²: {effort:.1f} Nm²\nSeat carries {100*load:.1f}% of weight',transform=ax.transAxes,va='top',fontsize=11,bbox=dict(facecolor='white',edgecolor='#cbd7d4',alpha=.95))
    fig.suptitle('40cm level seat: upright support removes most of the leg effort',fontsize=16)
    fig.text(.5,.03,'Green: permitted pelvis contact. Orange: hip housings, which must stay clear.\nRight image starts already seated; reaching it from standing is evaluated separately.',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.075,1,.95));a.output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.output,dpi=160);print(a.output)

if __name__=='__main__':main()
