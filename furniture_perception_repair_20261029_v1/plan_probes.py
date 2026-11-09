"""Interior targets chosen only from the observed patch, never from mesh truth."""
import numpy as np
from scipy.ndimage import distance_transform_edt
from paths import ASSETS,OUT,read,write

def main():
    plans={};missing=[]
    for c in read(ASSETS/'cases.json'):
        z=np.load(OUT/('robot_'+c['id'])/'frame_0.npz');pts=z['selected_points'];pix=z['selected_pixels']
        if len(pts)<20 or not z['geometry'][11]:missing.append(c['id']);continue
        mask=np.zeros(z['depth'].shape,bool);mask[tuple(pix.T)]=True
        # Distance to observed boundary, in image pixels converted at local depth.
        edge=distance_transform_edt(np.pad(mask,1))[1:-1,1:-1]
        clearance=edge[tuple(pix.T)]*z['depth'][tuple(pix.T)]/max(z['K'][0,0],z['K'][1,1])
        good=np.flatnonzero(clearance>=max(.01,.5*clearance.max()))
        if len(good)<3:missing.append(c['id']);continue
        first=int(good[np.argmax(clearance[good])]);chosen=[first]
        for _ in range(2):
            separation=np.linalg.norm(pts[good,None]-pts[chosen][None],axis=-1).min(1)
            chosen.append(int(good[np.argmax(separation)]))
        world=pts@z['root_rotation'].T+z['root_position'];normal=z['root_rotation']@z['geometry'][2:5];normal/=np.linalg.norm(normal)
        plans[c['id']]=[dict(point=world[j].tolist(),normal=normal.tolist(),observed_edge_clearance_m=float(clearance[j]),pixel=pix[j].tolist()) for j in chosen]
    write(OUT/'observed_probe_points.json',plans)
    write(OUT/'probe_plan_summary.json',dict(planned=list(plans),not_observable_or_too_narrow=missing,source='fresh depth patch only',truth_geometry_used=False,body_footprint_fit_not_guaranteed=True))

if __name__=='__main__':main()
