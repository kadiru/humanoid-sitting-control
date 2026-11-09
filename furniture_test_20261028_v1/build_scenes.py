"""CPU-only furniture conversion, placement, USD validation and preview.

Creates real mesh colliders, not a bounding-box couch. RobotPreview in the
inspection USD is a static collision-hull rendering of the saved standing pose;
run_scene.py uses the actual articulated G1 instead.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import json, hashlib, math
from pathlib import Path
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Gf, Vt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

HERE=Path(__file__).resolve().parent
ROOT=Path('/home/kadir/humanoid-sitting')
OUTPUT=HERE/'scenes';OUTPUT.mkdir(exist_ok=True)
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
sources=json.loads((HERE/'asset_sources.json').read_text())
bankpath=ROOT/'outputs/full_body_contact_imitation_v4/dataset/sitting'
bank=np.load(bankpath/'bank.npz');meta=json.loads((bankpath/'metadata.json').read_text())
trace=np.load(ROOT/'outputs/full_body_contact_imitation_v4/teacher_replay_sitting/trajectory.npz')
result=json.loads((ROOT/'outputs/full_body_contact_imitation_v4/teacher_replay_sitting/results.json').read_text())
pose=trace['body'][0,0].copy();pose[:,:3]-=np.array(result['origins'][0])
model=json.loads((ROOT/'runs/g1_kinematics_geometry_04/model.json').read_text())
robot=[]
for body in model['body_names']:
    pts=[np.asarray(c['points']) for c in model['collisions'] if c['body']==body and c['points']]
    if not pts:continue
    pts=np.concatenate(pts);i=model['body_names'].index(body)
    pts=pts@Rotation.from_quat(pose[i,3:7]).as_matrix().T+pose[i,:3]
    if len(pts)<4:continue
    try:m=trimesh.convex.convex_hull(pts)
    except Exception:continue
    robot.append((body,m))
save(HERE/'standing_pose.json',{'source':str(bankpath),'case':'known_old/0','root_pose':bank['reset_root_pose'][0].tolist(),'joint_names':meta['joint_names'],'q':bank['reset_q'][0].tolist(),'preview_pose':'teacher replay after first .02s control; simulator reset uses original saved standing state'})

def mesh_prim(stage,path,mesh,color,collision=False):
    g=UsdGeom.Mesh.Define(stage,path)
    g.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(mesh.vertices.astype(np.float32)))
    g.CreateFaceVertexCountsAttr([3]*len(mesh.faces));g.CreateFaceVertexIndicesAttr(mesh.faces.flatten().tolist())
    g.CreateSubdivisionSchemeAttr('none');g.CreateDoubleSidedAttr(True)
    g.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    g.CreateExtentAttr(Vt.Vec3fArray.FromNumpy(mesh.bounds.astype(np.float32)))
    if collision:
        UsdPhysics.CollisionAPI.Apply(g.GetPrim()).CreateCollisionEnabledAttr(True)
        UsdPhysics.MeshCollisionAPI.Apply(g.GetPrim()).CreateApproximationAttr('none')
    return g

def make_stage(path):
    stage=Usd.Stage.CreateNew(str(path));UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z);UsdGeom.SetStageMetersPerUnit(stage,1.)
    root=UsdGeom.Xform.Define(stage,'/Furniture');stage.SetDefaultPrim(root.GetPrim())
    return stage,root

cases=[];asset_audits=[]
for a in sources:
    name=a['asset_id'];f=HERE/'source_assets'/(Path(a['model_local']).name if a.get('model_local') else name+'_1k.gltf')
    scene=trimesh.load(f,force='scene',skip_materials=True)
    mesh=scene.to_geometry()
    T=np.eye(4);T[:3,:3]=[[1,0,0],[0,0,-1],[0,1,0]];mesh.apply_transform(T)
    original_bounds=mesh.bounds.copy()
    scale=2.2/mesh.extents[0] if name=='6MoOyPtetL' else 1.2 if name=='myd1WSucAz' else 1.
    mesh.apply_scale(scale);mesh.apply_translation([-mesh.bounds.mean(0)[0],-mesh.bounds.mean(0)[1],-mesh.bounds[0,2]])
    color=(.35,.16,.10) if name=='6MoOyPtetL' else (.18,.42,.48) if name=='myd1WSucAz' else (.57,.39,.22)
    variants=[('seat',90),('seat_oblique',60),('arm',0),('rear_back',-90)] if name in ['6MoOyPtetL','myd1WSucAz'] else [('top',0),('top_oblique',45)]
    asset_audits.append({'asset':name,'input_sha256':digest(f),'input_bounds_after_Y_up_to_Z_up':original_bounds.tolist(),'uniform_scale':scale,'dimensions_m':mesh.extents.tolist(),'faces':len(mesh.faces),'source_declared_millimeters':a.get('dimensions_mm'),'scale_policy':'Chosen test scale, not verified real-world size' if scale!=1 else 'Native mesh scale; compared with publisher dimensions','collision':'Exact triangles, kinematic rigid furniture; no cushion compliance modeled'})
    for variant,yaw in variants:
        m=mesh.copy();rot=Rotation.from_euler('z',yaw,degrees=True).as_matrix();T=np.eye(4);T[:3,:3]=rot;m.apply_transform(T)
        # Place nearest furniture edge behind the standing robot, then enforce
        # conservative triangle-AABB separation from every robot collision hull.
        shift=-.18-m.bounds[1,0];m.apply_translation([shift,0,0])
        extra=0.
        tri=m.triangles;lo=tri.min(1);hi=tri.max(1)
        for _,r in robot:
            rb=r.bounds
            active=(hi[:,1]>=rb[0,1])&(lo[:,1]<=rb[1,1])&(hi[:,2]>=rb[0,2])&(lo[:,2]<=rb[1,2])
            if active.any():extra=max(extra,float(hi[active,0].max()-rb[0,0]+.06))
        if extra>0:m.apply_translation([-extra,0,0])
        offset=np.array([shift-max(extra,0),0,0])
        key=f'{name}_{variant}';folder=OUTPUT/key;folder.mkdir(exist_ok=True)
        stage,root=make_stage(folder/'furniture.usda')
        body=UsdPhysics.RigidBodyAPI.Apply(root.GetPrim());body.CreateRigidBodyEnabledAttr(True);body.CreateKinematicEnabledAttr(True)
        UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(50.)
        material=UsdShade.Material.Define(stage,'/Furniture/Material');phys=UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        phys.CreateStaticFrictionAttr(.8);phys.CreateDynamicFrictionAttr(.6);phys.CreateRestitutionAttr(0.)
        g=mesh_prim(stage,'/Furniture/Mesh',m,color,True)
        UsdShade.MaterialBindingAPI.Apply(g.GetPrim()).Bind(material,UsdShade.Tokens.weakerThanDescendants,'physics')
        stage.GetRootLayer().Save()
        # Inspection stage: collision hulls are visibly labelled, not an
        # independently simulated robot or evidence of successful sitting.
        view=Usd.Stage.CreateNew(str(folder/'layout.usda'));UsdGeom.SetStageUpAxis(view,'Z');UsdGeom.SetStageMetersPerUnit(view,1.)
        world=UsdGeom.Xform.Define(view,'/World');view.SetDefaultPrim(world.GetPrim())
        furn=UsdGeom.Xform.Define(view,'/World/Furniture');furn.GetPrim().GetReferences().AddReference('furniture.usda')
        for j,(n,r) in enumerate(robot):mesh_prim(view,'/World/RobotPreview/'+n,r,(.67,.75,.81))
        floor=trimesh.creation.box(extents=[5,5,.04]);floor.apply_translation([-.8,0,-.02]);mesh_prim(view,'/World/Floor',floor,(.85,.88,.91))
        view.GetRootLayer().Save()
        np.savez_compressed(folder/'mesh.npz',vertices=m.vertices,faces=m.faces)
        # Region proposals for later sensor/contact validation. They are NOT
        # ground-truth actor inputs and do not yet admit a pass/fail score.
        c=m.triangles_center;n=m.face_normals
        if variant=='rear_back':
            mask=(n[:,0]>.65)&(c[:,2]>.55*m.bounds[1,2])
        elif variant=='arm':
            mask=(n[:,2]>.8)&(c[:,2]>.5*m.bounds[1,2])&(c[:,2]<.85*m.bounds[1,2])&(c[:,0]>m.bounds[1,0]-.18)
        elif variant.startswith('seat'):
            h=(.905 if name=='6MoOyPtetL' else .311)*scale
            mask=(n[:,2]>.8)&(abs(c[:,2]-h)<.03)
        else:mask=(n[:,2]>.8)&(c[:,2]>m.bounds[1,2]-.03)
        assert mask.any(),key
        target=np.average(c[mask],axis=0,weights=m.area_faces[mask]);normal=np.average(n[mask],axis=0,weights=m.area_faces[mask]);normal/=np.linalg.norm(normal)
        save(folder/'target_region.json',{'triangle_indices':np.flatnonzero(mask).tolist(),'center':target.tolist(),'mean_normal':normal.tolist(),'status':'Proposed region from mesh orientation; not an audited physical contact label','actor_truth_input':False})
        case={'id':key,'asset':name,'title':a['title'],'approach':variant,'yaw_deg':yaw,'task':'leaning' if variant=='rear_back' or name=='wooden_table_02' else 'sitting','mesh_usd':str(folder/'furniture.usda'),'mesh_npz':str(folder/'mesh.npz'),'bounds':m.bounds.tolist(),'target_center':target.tolist(),'target_normal':normal.tolist(),'translation':offset.tolist(),'uniform_scale':scale,'color':color,'initial_triangle_AABB_clearance_m':.06,'contact_scoring_validated':False}
        cases.append(case)

save(HERE/'cases.json',cases);save(HERE/'asset_audit.json',asset_audits)
# Round-trip USD topology, transforms, collision bindings and placement checks.
checks=[]
for c in cases:
    z=np.load(c['mesh_npz']);v=z['vertices'];faces=z['faces'];stage=Usd.Stage.Open(c['mesh_usd']);g=UsdGeom.Mesh(stage.GetPrimAtPath('/Furniture/Mesh'))
    np.testing.assert_allclose(np.asarray(g.GetPointsAttr().Get()),v,atol=1e-7,rtol=1e-7)
    np.testing.assert_array_equal(np.asarray(g.GetFaceVertexIndicesAttr().Get()).reshape(-1,3),faces)
    assert np.isfinite(v).all() and v[:,2].min()>-1e-7 and v[:,0].max()<0
    assert UsdPhysics.CollisionAPI(g.GetPrim()).GetCollisionEnabledAttr().Get()
    assert UsdPhysics.RigidBodyAPI(stage.GetDefaultPrim()).GetKinematicEnabledAttr().Get()
    lo=v[faces].min(1);hi=v[faces].max(1)
    overlaps=0
    for _,r in robot:overlaps+=int(((hi>=r.bounds[0]).all(1)&(lo<=r.bounds[1]).all(1)).sum())
    assert overlaps==0,c['id']
    checks.append({'case':c['id'],'usd_roundtrip':True,'initial_triangle_AABB_overlaps':overlaps,'behind_robot':True,'file_sha256':digest(Path(c['mesh_usd']))})
save(HERE/'static_validation.json',{'passed':True,'cases':len(cases),'checks':checks,'simulator_started':False,'GPU_used':False,'policy_evaluation_performed':False,'limitations':['No PhysX import/settling/contact sensor test yet','Anchored rigid furniture; no compliant cushions or tipping','Target regions require multi-surface contact validation; formal clean score disabled','This furniture set is separate from the protected 72-case benchmark']})

fig=plt.figure(figsize=(16,12),facecolor='#f2f5f8')
for index,c in enumerate(cases):
    ax=fig.add_subplot(4,4,index+1,projection='3d');ax.set_facecolor('#f2f5f8')
    z=np.load(c['mesh_npz']);tris=z['vertices'][z['faces']]
    normals=np.cross(tris[:,1]-tris[:,0],tris[:,2]-tris[:,0]);normals/=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-12)
    light=np.array([1.,-.6,1.6]);light/=np.linalg.norm(light)
    colors=np.asarray(c['color'])[None,:]*(.55+.45*np.maximum(normals@light,0))[:,None]
    ax.add_collection3d(Poly3DCollection(tris,facecolor=colors,edgecolor='none',alpha=1))
    for _,r in robot:
        rc=np.array([.48,.66,.78])[None,:]*(.55+.45*np.maximum(r.face_normals@light,0))[:,None]
        ax.add_collection3d(Poly3DCollection(r.triangles,facecolor=rc,edgecolor='none',alpha=1))
    t=np.array(c['target_center']);ax.scatter(*t,s=35,c='#ee6a25',depthshade=False)
    ax.quiver(.1,0,.03,.35,0,0,color='#246290',arrow_length_ratio=.25)
    xmin=float(z['vertices'][:,0].min())-.1;ymax=max(.45,float(abs(z['vertices'][:,1]).max())+.1)
    ax.set(xlim=(xmin,.45),ylim=(-ymax,ymax),zlim=(0,1.35));ax.set_box_aspect((.45-xmin,2*ymax,1.35));ax.view_init(elev=24,azim=-56)
    ax.set_title(c['title']+'\n'+c['approach'].replace('_',' '),fontsize=10,pad=0);ax.set_axis_off()
fig.suptitle('Furniture tests behind the G1 • 14 isolated layouts',fontsize=21,color='#183348',y=.98)
fig.text(.5,.025,'Actual furniture meshes + robot collision-hull preview. Orange: proposed target. Blue arrow: robot faces forward.\nCPU scene preparation only — no policy trial or physical success claimed.',ha='center',fontsize=11,color='#3b5061')
fig.subplots_adjust(left=.01,right=.99,bottom=.075,top=.93,wspace=.01,hspace=.12)
fig.savefig(HERE/'scene_preview.jpg',dpi=140);plt.close(fig)
print(json.dumps({'cases':len(cases),'static_validation':'passed','simulation':'not started'}))
