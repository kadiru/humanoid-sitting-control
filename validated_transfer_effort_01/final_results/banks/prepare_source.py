"""Freeze unseen surface grids without altering the validated motion or standing reset."""
import argparse,copy,hashlib,json,shutil
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output);out.mkdir(exist_ok=False)
    src=Path('/data/humanoid-sitting-perception/two_surface_curriculum_01/selected_path_01')
    geoms=json.loads((src/'geometry.json').read_text())
    goals=dict(np.load(src/'candidates.npz'));resets=dict(np.load(src/'reset_candidates.npz'))
    grids={'development':[(h,s) for h in [-.02,0.,.02] for s in [-5.,0.,5.]],
           'holdout':[(h,s) for h in [-.01,.01] for s in [-2.5,2.5]]+[(-.03,0.),(.03,0.),(0.,-7.5),(0.,7.5)],
           'identity':[(0.,0.)]}
    for split,grid in grids.items():
        folder=out/split;folder.mkdir();items=[];indices=[]
        for i,g in enumerate(geoms):
            for h,slope in grid:
                item=copy.deepcopy(g);R=Rotation.from_euler('y',g['pitch_deg']+slope,degrees=True)
                item.update(case=len(items),surface_id=len(items),source_motion_id=i,
                            label=f"{g['kind']}_dh{h:+.3f}_ds{slope:+.1f}",height=g['height']+h,
                            pitch_deg=g['pitch_deg']+slope,rotation_xyzw=R.as_quat().tolist(),
                            normal=R.as_matrix()[:,2].tolist(),top_center=[g['top_center'][0],g['top_center'][1],g['top_center'][2]+h],
                            height_delta_m=h,slope_delta_deg=slope,split=split,
                            known_anchor=h==0 and slope==0,reset_scope='Standing only; original intermediate/end states are commands, not validated resets on changed surfaces')
                for k in ['initial_forbidden_penetrations','nearest_bodies','source_validation','source_case_index']:item.pop(k,None)
                items.append(item);indices.append(i)
        (folder/'geometry.json').write_text(json.dumps(items,indent=2))
        np.savez_compressed(folder/'candidates.npz',**{k:v[indices] for k,v in goals.items()})
        np.savez_compressed(folder/'reset_candidates.npz',**{k:v[indices] for k,v in resets.items()})
        (folder/'selection.json').write_text(json.dumps({'source':str(src),'indices':indices,'scope':'Same motion and standing state, independently varied height/slope; no geometry-specific reference retuning'},indent=2))
    shutil.copy2(__file__,out/'prepare_source.py')
    (out/'bank_protocol.json').write_text(json.dumps({'grid':grids,'sitting_base_height_m':geoms[0]['height'],'leaning_base_top_center_height_m':geoms[1]['height'],'source_hashes':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in src.iterdir() if f.is_file()},'holdout':'Reserved before any assessment; never used for calibration, training, checkpoint selection or controller tuning','standing_only':True},indent=2))
    print(out)
if __name__=='__main__':main()
