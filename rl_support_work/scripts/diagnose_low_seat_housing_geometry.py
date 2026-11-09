"""Offline mesh-clearance versus recorded contact/pose labels; never control."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from humanoid_sitting.shared_frame import BodyModel,apply,transform
from plan_seat_load_targets import clipped_vertices
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')
def read(p):return json.loads(p.read_text())
def main():
    m=BodyModel('runs/g1_kinematics_geometry_04/model.json');out=[]
    for variant in ['normal_landing3s_06','normal_slow_orientation_08']:
        p=D/variant/'candidate';r=read(p/'results.json');geom=read(p/'geometry.json');t=np.load(p/'trace.npz');e=np.load(p/'every_tick.npz');origin=np.asarray(r['origins']);names=r['body_names']
        for row in r['episodes']:
            i=row['case']
            if i%2 or not row['failures']['forbidden']:continue
            g=geom[row['geometry_index']];Rseat=Rotation.from_quat(g['rotation_xyzw']).as_matrix();center=np.array(g['top_center'])
            bad=np.flatnonzero(e['wrong'][:,i]>10.);first=float(e['t'][bad[0]])
            before=np.flatnonzero(t['t']<=first)[-1];samples=[]
            for k in sorted(set([max(before-1,0),before,min(before+1,len(t['t'])-1)])):
                body=t['body'][k,i];root=body[names.index('pelvis')];RT=transform(Rotation.from_quat(root[3:]).as_matrix(),root[:3]-origin[i]);fk=m.fk(t['q'][k,i]);items=[]
                for side in ['left','right']:
                    for joint in ['hip_pitch','hip_roll','knee','ankle_pitch']:
                        name=side+'_'+joint+'_link';j=names.index(name)
                        if name not in m.vertices:continue
                        pose=body[j];T=transform(Rotation.from_quat(pose[3:]).as_matrix(),pose[:3]-origin[i]);points=(apply(T,m.vertices[name])-center)@Rseat;points[:,0]-=.4;points=clipped_vertices(points)
                        kinematic=RT@fk[name]
                        items.append(dict(body=name,min_mesh_plane_clearance_m=float(points[:,2].min()) if len(points) else None,
                            measured_body_vs_fk_position_error_m=float(np.linalg.norm(kinematic[:3,3]-T[:3,3]))))
                samples.append(dict(t=float(t['t'][k]),bodies=items))
            result=dict(variant=variant,case=i,height=g['height'],slope=g['pitch_deg'],first_forbidden_s=first,
                first_wrong_force_N=float(e['wrong'][bad[0],i]),samples=samples)
            out.append(result);print(json.dumps(result),flush=True)
    p=D/'housing_geometry_diagnosis_08.json';assert not p.exists();p.write_text(json.dumps(dict(scope='Offline recorded transforms and model vertices. 0.1s pose sampling brackets0.02s contact labels; not exact event geometry and not a runtime input.',rows=out),indent=2))
if __name__=='__main__':main()
