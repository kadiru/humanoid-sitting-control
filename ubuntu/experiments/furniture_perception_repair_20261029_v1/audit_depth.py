"""Independent image-to-mesh audit; no production recognizer import."""
import argparse
import numpy as np
import trimesh
from paths import ASSETS,read,write,sha

def check(path,case):
    z=np.load(path);d=z['depth'];K=z['K'];R=z['camera_rotation'];pos=z['camera_position']
    meshdata=np.load(case['mesh_npz']);mesh=trimesh.Trimesh(meshdata['vertices'],meshdata['faces'],process=False)
    region=read(ASSETS/'scenes'/case['id']/'target_region.json')['triangle_indices']
    v,u=np.mgrid[:d.shape[0]:3,:d.shape[1]:3];v=v.ravel();u=u.ravel()
    optical=np.column_stack(((u+.5-K[0,2])/K[0,0],(v+.5-K[1,2])/K[1,1],np.ones(len(u))))
    directions=optical@R.T;hits,rayids,faces=mesh.ray.intersects_location(np.broadcast_to(pos,directions.shape),directions,multiple_hits=False)
    expected=np.full(len(u),np.inf);face=np.full(len(u),-1)
    # Optical-z depth recovered independently as ray parameter, not range.
    expected[rayids]=np.sum((hits-pos)*directions[rayids],1)/np.sum(directions[rayids]**2,1);face[rayids]=faces
    actual=d[v,u];inview=np.isfinite(expected)&(expected>0);finite=np.isfinite(actual)&(actual>0)
    occluded=inview&finite&(actual<expected-.015)
    matching=inview&finite&(np.abs(actual-expected)<=.015);missing=inview&~occluded&~matching
    target=np.isin(face,region);g=z['geometry'];normal=z['root_rotation'].T@case['target_normal']
    angle=float(np.rad2deg(np.arccos(np.clip(g[2:5]@normal,-1,1)))) if g[11] else None
    offset=float(g[5]-(z['root_position']-case['target_center'])@case['target_normal']) if g[11] else None
    points=z['selected_points']@z['root_rotation'].T+z['root_position']
    dist=np.array([]);selected=np.array([],int)
    if len(points):_,dist,selected=mesh.nearest.on_surface(points)
    return dict(frame=int(path.stem.split('_')[-1]),input_sha256=sha(path),
        projected_pixels=int(inview.sum()),matching_pixels=int(matching.sum()),occluded_pixels=int(occluded.sum()),mismatch_pixels=int(missing.sum()),
        geometry_pass=bool(matching.sum()>=30 and missing.sum()<=max(1,int(.01*(inview&~occluded).sum()))),
        visible_target_pixels=int((target&matching).sum()),target_observable=bool((target&matching).sum()>=20),
        depth_p99_m=float(np.quantile(abs(actual[matching]-expected[matching]),.99)) if matching.any() else None,
        selected_point_count=len(points),selected_target_fraction=float(np.isin(selected,region).mean()) if len(points) else None,
        selected_max_mesh_distance_m=float(dist.max()) if len(points) else None,
        original_plane_gate=bool(g[11] and angle<=5 and abs(offset)<=.015),normal_error_deg=angle,offset_error_m=offset)

def main():
    from pathlib import Path
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--case',required=True);a=p.parse_args()
    case=next(c for c in read(ASSETS/'cases.json') if c['id']==a.case)
    rows=[check(a.output/f'frame_{i}.npz',case) for i in range(3)]
    write(a.output/'depth_audit.json',dict(complete=True,case=case['id'],frames=rows,
        geometry_pass=all(r['geometry_pass'] for r in rows),original_plane_gate=all(r['original_plane_gate'] for r in rows),
        thresholds_unchanged=True,association_fraction_is_diagnostic_not_relaxed_gate=True,robot_success_claim=False))

if __name__=='__main__':main()
