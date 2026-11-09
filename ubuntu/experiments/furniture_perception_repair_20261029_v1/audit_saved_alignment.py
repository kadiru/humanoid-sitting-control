"""CPU diagnosis against saved captures; no fitted camera corrections."""
import json,hashlib
from pathlib import Path
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

ROOT=Path('/home/kadir/humanoid-sitting')
def main():
    old=ROOT/'outputs/motion_prior_ppo_20261029_v1';assets=ROOT/'experiments/furniture_test_20261028_v1'
    out=ROOT/'outputs/furniture_perception_repair_20261029_v1';out.mkdir(exist_ok=True)
    mount=np.array([[0.,2**-.5,-2**-.5],[1.,0.,0.],[0.,-2**-.5,-2**-.5]])
    rows=[]
    for case in json.loads((assets/'cases.json').read_text()):
        bank=np.load(ROOT/'outputs/full_body_contact_imitation_v4/dataset'/case['task']/'bank.npz')
        root=bank['reset_root_pose'][0];rootR=Rotation.from_quat(root[3:]).as_matrix();R=rootR@mount;pos=root[:3]+rootR@np.array([-.07,0,.38])
        m=np.load(case['mesh_npz']);mesh=trimesh.Trimesh(m['vertices'],m['faces'],process=False)
        for frame in range(3):
            path=old/('depth_'+case['id'])/f'frame_{frame}.npz';z=np.load(path);depth=z['depth'];K=z['K'];h,w=depth.shape
            v,u=np.mgrid[0:h:3,0:w:3];v=v.ravel();u=u.ravel();summary={}
            for name,po,ro,offset in [('old',z['camera_position'],Rotation.from_quat(z['camera_quaternion_ros']).as_matrix(),0.),('corrected_reset_reference',pos,R,.5)]:
                dirs=np.stack(((u+offset-K[0,2])/K[0,0],(v+offset-K[1,2])/K[1,1],np.ones(len(u))),-1)@ro.T
                hit,rays,faces=mesh.ray.intersects_location(np.broadcast_to(po,dirs.shape),dirs,multiple_hits=False)
                expected=(hit-po)@ro[:,2];measured=depth[v[rays],u[rays]];residual=measured-expected
                finite=np.isfinite(measured)&(measured>0);occluded=finite&(residual<-.015);error=abs(residual[finite&~occluded])
                summary[name]=dict(projected_pixels=len(rays),matching_pixels=int((finite&(abs(residual)<=.015)).sum()),mismatch_pixels=int((~finite|(residual>.015)).sum()),occluded_pixels=int(occluded.sum()),median_abs_m=float(np.median(error)),p99_abs_m=float(np.quantile(error,.99)),max_abs_m=float(error.max()))
            implied_root=z['camera_position']-Rotation.from_quat(z['camera_quaternion_ros']).as_matrix()@mount.T@np.array([-.07,0,.38])
            rows.append(dict(case=case['id'],frame=frame,metadata_implied_root=implied_root.tolist(),actual_reset_root=root.tolist(),results=summary,input_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    result=dict(scope='Saved-frame diagnosis using known reset pose only for offline geometry checks; fresh tests must bind each capture to contemporaneous robot pose. No mesh-fitted camera parameter and no actor truth input.',rows=rows)
    (out/'saved_alignment_diagnosis.json').write_text(json.dumps(result,indent=2)+'\n')
    for x in rows[::3]:print(x['case'],x['results'])

if __name__=='__main__':main()
