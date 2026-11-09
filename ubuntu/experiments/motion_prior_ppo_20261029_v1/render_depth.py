"""Actual sensor panels; no generated images or synthetic camera evidence."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from common import ROOT,read,write

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output)
    cases=read(ROOT/'experiments/furniture_test_20261028_v1/cases.json');panels=[];rows=[]
    for c in cases:
        folder=out/('depth_'+c['id']);s=read(folder/'depth_summary.json');r=s['frames'][0]
        with np.load(folder/'frame_0.npz') as z:d={k:z[k] for k in z.files}
        fig,ax=plt.subplots(1,3,figsize=(12,3.5),dpi=100)
        ax[0].imshow(d['rgb'][...,:3]);ax[0].set_title('Rear camera RGB');ax[0].axis('off')
        depth=d['depth'].copy();depth[~np.isfinite(depth)]=np.nan
        ax[1].imshow(depth,cmap='viridis',vmin=0,vmax=2);ax[1].set_title('Measured optical depth: 0–2 m');ax[1].axis('off')
        selected=d['matching']&d['target'];ax[1].scatter(d['sample_u'][selected],d['sample_v'][selected],s=2,c='red')
        cloud=d['points'];keep=np.isfinite(cloud).all(1)&(np.linalg.norm(cloud,axis=1)<3)
        ax[2].scatter(cloud[keep,0],cloud[keep,2],s=1,c='#888888');q=d['target_points']
        if len(q):ax[2].scatter(q[:,0],q[:,2],s=3,c='red')
        ax[2].set(xlabel='World x (m)',ylabel='Height (m)',title='Back-projected points; target in red',xlim=(-2,.3),ylim=(0,1.6));ax[2].set_aspect('equal')
        title=f"{c['title']} / {c['approach']} — depth {'OK' if r['camera_geometry_consistent'] else 'CHECK'}; target pixels {r['visible_target_pixels']}; recognizer {'TARGET' if r['recognizer_matches_target'] else 'OTHER / INVALID'}"
        fig.suptitle(title,fontsize=10);fig.tight_layout();name=folder/'camera_depth_points.png';fig.savefig(name);plt.close(fig)
        panels.append(Image.open(name).convert('RGB'));rows.append(dict(case=c['id'],summary=str(folder/'depth_summary.json'),panel=str(name),**r))
    w,h=panels[0].size;canvas=Image.new('RGB',(2*w,7*h),'white')
    for i,panel in enumerate(panels):canvas.paste(panel,((i%2)*w,(i//2)*h))
    canvas.save(out/'rear_depth_collage.jpg',quality=90)
    write(out/'depth_overview.json',dict(rows=rows,collage=str(out/'rear_depth_collage.jpg'),all14captured=True,
        surface_geometry_truth_used_only_for_offline_scoring=True))

if __name__=='__main__':main()
