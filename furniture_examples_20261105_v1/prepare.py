"""CPU-only scene/reset construction. No simulator or learner is imported."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='1'
import json,hashlib,shutil
from pathlib import Path
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from pxr import Usd,UsdGeom,UsdPhysics,UsdShade,Gf,Vt

ROOT=Path('/home/kadir/humanoid-sitting')
HERE=Path(__file__).resolve().parent
OLD=ROOT/'experiments/furniture_test_20261028_v1'
DATA=Path('/data/humanoid-sitting-perception/low_seat_contact_01')
read=lambda p:json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def mesh_prim(stage,path,mesh,color):
    g=UsdGeom.Mesh.Define(stage,path)
    g.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(mesh.vertices.astype(np.float32)))
    g.CreateFaceVertexCountsAttr([3]*len(mesh.faces));g.CreateFaceVertexIndicesAttr(mesh.faces.flatten().tolist())
    g.CreateSubdivisionSchemeAttr('none');g.CreateDoubleSidedAttr(True)
    g.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdPhysics.CollisionAPI.Apply(g.GetPrim()).CreateCollisionEnabledAttr(True)
    UsdPhysics.MeshCollisionAPI.Apply(g.GetPrim()).CreateApproximationAttr('none')
    return g

def build_l_couch():
    path=HERE/'source_assets/1kwsjhpY84.glb'
    m=trimesh.load(path,force='scene',skip_materials=True).to_geometry()
    T=np.eye(4);T[:3,:3]=[[1,0,0],[0,0,-1],[0,1,0]];m.apply_transform(T)
    scale=2.2/m.extents[0];m.apply_scale(scale)
    m.apply_translation([-m.bounds.mean(0)[0],-m.bounds.mean(0)[1],-m.bounds[0,2]])
    cases=[]
    for approach,yaw in [('seat',90),('rear_back',-90)]:
        mesh=m.copy();R=Rotation.from_euler('z',yaw,degrees=True).as_matrix()
        T=np.eye(4);T[:3,:3]=R;mesh.apply_transform(T)
        shift=-.18-mesh.bounds[1,0];mesh.apply_translation([shift,0,0])
        ident='1kwsjhpY84_'+approach;folder=HERE/'scenes'/ident;folder.mkdir(parents=True,exist_ok=False)
        stage=Usd.Stage.CreateNew(str(folder/'furniture.usda'));UsdGeom.SetStageUpAxis(stage,'Z');UsdGeom.SetStageMetersPerUnit(stage,1.)
        root=UsdGeom.Xform.Define(stage,'/Furniture');stage.SetDefaultPrim(root.GetPrim())
        rigid=UsdPhysics.RigidBodyAPI.Apply(root.GetPrim());rigid.CreateRigidBodyEnabledAttr(True);rigid.CreateKinematicEnabledAttr(True)
        UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(50.)
        mat=UsdShade.Material.Define(stage,'/Furniture/Material');phys=UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
        phys.CreateStaticFrictionAttr(.8);phys.CreateDynamicFrictionAttr(.6);phys.CreateRestitutionAttr(0.)
        g=mesh_prim(stage,'/Furniture/Mesh',mesh,(.36,.41,.57))
        UsdShade.MaterialBindingAPI.Apply(g.GetPrim()).Bind(mat,UsdShade.Tokens.weakerThanDescendants,'physics');stage.GetRootLayer().Save()
        np.savez_compressed(folder/'mesh.npz',vertices=mesh.vertices,faces=mesh.faces)
        c=mesh.triangles_center;n=mesh.face_normals
        if approach=='seat':
            eligible=(n[:,2]>.8)&(c[:,2]>.25)&(c[:,2]<.65)
            heights=np.round(c[:,2]/.01).astype(int)
            groups=np.unique(heights[eligible]);assert len(groups)
            h=max(groups,key=lambda k:mesh.area_faces[eligible&(heights==k)].sum())*.01
            mask=eligible&(abs(c[:,2]-h)<.015)
        else:mask=(n[:,0]>.65)&(c[:,2]>.55*mesh.bounds[1,2])
        assert mask.any()
        center=np.average(c[mask],axis=0,weights=mesh.area_faces[mask]);normal=np.average(n[mask],axis=0,weights=mesh.area_faces[mask]);normal/=np.linalg.norm(normal)
        write(folder/'target_region.json',dict(triangle_indices=np.flatnonzero(mask).tolist(),center=center.tolist(),mean_normal=normal.tolist(),status='PROPOSED_TARGET_UNVERIFIED',actor_truth_input=False))
        cases.append(dict(id=ident,asset='1kwsjhpY84',title='L-shaped couch',approach=approach,yaw_deg=yaw,
            task='sitting' if approach=='seat' else 'leaning',mesh_usd=str(folder/'furniture.usda'),mesh_npz=str(folder/'mesh.npz'),
            bounds=mesh.bounds.tolist(),target_center=center.tolist(),target_normal=normal.tolist(),uniform_scale=scale,
            color=[.36,.41,.57],collision='Exact triangles, anchored rigid furniture',contact_scoring_validated=False))
    return cases

def main():
    p=read(HERE/'protocol.json');needed={v['case'] for v in p['variants']};cases=[];bindings={}
    for c in read(OLD/'cases.json'):
        if c['id'] not in needed:continue
        folder=HERE/'scenes'/c['id'];shutil.copytree(OLD/'scenes'/c['id'],folder)
        for f in ['furniture.usda','mesh.npz','target_region.json']:
            bindings[str(OLD/'scenes'/c['id']/f)]=sha(OLD/'scenes'/c['id']/f)
            assert sha(folder/f)==bindings[str(OLD/'scenes'/c['id']/f)]
        c.update(mesh_usd=str(folder/'furniture.usda'),mesh_npz=str(folder/'mesh.npz'));cases.append(c)
    cases+=build_l_couch();write(HERE/'cases.json',cases)
    assert {c['id'] for c in cases}==needed
    original=read(DATA/'current_retention_09/known_old/candidate.command.json')
    profiles=read('/data/humanoid-sitting-perception/contact_effort_repair_01/sitting_repaired_profile.json')
    assert len(profiles)==1
    for v in p['variants']:
        if v['task']!='sitting':continue
        folder=HERE/'banks'/v['id'];folder.mkdir(parents=True,exist_ok=False)
        case=next(c for c in cases if c['id']==v['case'])
        src=DATA/'banks/known_old';g=read(src/'geometry.json')[0]
        n=np.array(case['target_normal']);axis=np.cross([0,1,0],n);axis/=np.linalg.norm(axis)
        q=Rotation.from_matrix(np.stack([axis,np.cross(n,axis),n],axis=1)).as_quat()
        g.update(case=0,label=v['id'],top_center=case['target_center'],normal=n.tolist(),rotation_xyzw=q.tolist(),
            height=case['target_center'][2],x=case['target_center'][0],y=case['target_center'][1],pitch_deg=float(np.rad2deg(np.arccos(n[2]))),source_scope='Furniture target truth is evaluation/reset only; runtime plane comes from camera')
        write(folder/'geometry.json',[g])
        for f in ['candidates.npz','reset_candidates.npz']:
            with np.load(src/f) as z:bank={k:z[k][:1].copy() for k in z.files}
            pose=bank['root_pose'];pose[...,:2]+=np.array([v['x'],v['y']])
            R=Rotation.from_euler('z',v['yaw_deg'],degrees=True)
            pose[...,3:7]=(R*Rotation.from_quat(pose[...,3:7].reshape(-1,4))).as_quat().reshape(pose[...,3:7].shape)
            for sl in [slice(0,3),slice(3,6)]:bank['root_velocity'][...,sl]=R.apply(bank['root_velocity'][...,sl].reshape(-1,3)).reshape(bank['root_velocity'][...,sl].shape)
            np.savez_compressed(folder/f,**bank);bindings[str(src/f)]=sha(src/f)
        args=original[3:]
        for opt,value in [('--bank',str(folder)),('--output',str(ROOT/'outputs'/HERE.name/v['id'])),('--num-envs','1'),('--eligible-indices','0')]:args[args.index(opt)+1]=value
        (HERE/'jobs').mkdir(exist_ok=True)
        write(HERE/'jobs'/(v['id']+'.json'),dict(entry='run_low_seat_current_timing_probe.py',arguments=args,task='sitting',bank=v['id']))
    bindings[str(HERE/'source_assets/1kwsjhpY84.glb')]=sha(HERE/'source_assets/1kwsjhpY84.glb')
    write(HERE/'preparation.json',dict(state='PREPARED_NOT_SIMULATED',scenarios=len(p['variants']),counts=p['counts'],source_bindings=bindings,
        originals_modified=False,ppo_started=False,demonstrations_admitted=0,physical_success_claimed=False))
    print('Prepared',len(p['variants']),'scenarios; no simulation executed',flush=True)

if __name__=='__main__':main()
