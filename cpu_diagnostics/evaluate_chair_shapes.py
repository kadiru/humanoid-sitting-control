"""Benchmark the frozen sitting controller on varied rigid furniture."""

import argparse
from collections import defaultdict
import csv
from pathlib import Path
import re
import shutil
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

from compare_touchdown_ppo import verify_audit
from evaluate_grip_fresh import TOUCHDOWN_REFERENCE
from study_live_sitting import (
    EVAL_WALL, GPU, MIN_DISK, PREPARED, ROOT, bank_from_geometry, read, sha, stage, write,
)


# Unique real dimensions for each construction. Back and arm values are
# heights above the seat. Uprights and rails are rigid and collision enabled.
CHAIRS = (
    ('square_stool', .36, .39, 'rect', 0., 0., 'legs'),
    ('round_stool', .39, .43, 'ellipse', 0., 0., 'legs'),
    ('pedestal_stool', .42, .45, 'ellipse', 0., 0., 'pedestal'),
    ('dining_chair', .43, .49, 'rect', .36, 0., 'legs'),
    ('ladder_back', .45, .52, 'rect', .45, 0., 'slats'),
    ('cantilever_chair', .46, .54, 'chamfered', .35, 0., 'sled'),
    ('padded_side_chair', .48, .57, 'chamfered', .42, 0., 'legs'),
    ('wooden_armchair', .49, .63, 'rect', .45, .18, 'legs'),
    ('club_armchair', .54, .71, 'chamfered', .37, .20, 'box'),
    ('backless_bench', .44, 1.03, 'rect', 0., 0., 'legs'),
    ('garden_bench', .47, 1.14, 'rect', .40, .15, 'slats'),
    ('two_seat_sofa', .56, 1.21, 'rect', .41, .20, 'box'),
)
HEIGHTS = (.337, .355, .378, .402, .425, .454)
PITCHES = (0., 6.5, 13., 19.75)
OFFSETS = ((-.039, -.055, -14.), (-.022, .032, 8.), (-.007, -.017, 0.),
           (.009, .016, 0.), (.027, -.037, -8.), (.039, .057, 14.))
YAW_OFFSETS = ((-.005,0.,-12.),(-.005,0.,0.),(-.005,0.,12.))
FROZEN = TOUCHDOWN_REFERENCE/'untouched.freeze.json'
ASSET_ROOT = ROOT/'experiments/furniture_test_20261028_v1/scenes'
MESHES = (('mesh_armchair', 'myd1WSucAz_seat'),
          ('mesh_l_couch', '6MoOyPtetL_seat'),
          ('mesh_wooden_stool', 'wooden_stool_01_top'),
          ('mesh_coffee_table', 'modern_coffee_table_01_top'))
GPU_BUDGET_S = 7*3600


def pieces(depth,width,height,rot,back,arms,base):
    """Boxes in the seat body's local frame, with grounded bases."""
    r = rot.as_matrix()
    root_z = height-.05*r[2,2]
    upright = (rot.inv()*Rotation.from_euler('z',rot.as_euler('zyx',degrees=False)[0])).as_quat()
    parts = []

    def box(name,position,size,orientation=(0.,0.,0.,1.)):
        parts.append(dict(name=name,center=list(position),size=list(size),xyzw=list(orientation)))

    def support(name,x,y,w,d,ground_gap=0.):
        underside = r@np.array([x,y,-.048])
        span = root_z+underside[2]-ground_gap
        center = r.T@(underside-[0.,0.,span/2])
        assert span>.09
        box(name,center,(d,w,span),upright)

    if base=='box':
        support('Plinth',-.018,0.,width*.89,depth*.80,.015)
    elif base=='pedestal':
        support('Pedestal',0.,0.,.085,.085,.027)
        center = r.T@np.array([0.,0.,.015-root_z])
        box('PedestalBase',center,(depth*.72,width*.72,.03),upright)
    else:
        for side in (-1,1):
            y = side*width*.34
            positions = (-depth*.31,) if base=='sled' else (-depth*.34,depth*.34)
            for i,x in enumerate(positions):
                support(f'Upright_{side+1}_{i}',x,y,.039,.041,.025 if base=='sled' else 0.)
            if base=='sled':
                center = r.T@np.array([0.,y,.017-root_z])
                box(f'Runner_{side+1}',center,(depth*.84,.047,.034),upright)

    if back:
        if base=='slats':
            for side in (-1,1):
                box(f'BackPost_{side+1}',(-depth/2+.024,side*width*.43,.05+back/2),
                    (.05,.045,back))
            for i,level in enumerate((.42,.66,.91)):
                box(f'BackRail_{i}',(-depth/2+.025,0.,.05+level*back),(.05,width*.89,.055))
        else:
            box('Back',(-depth/2+.03,0.,.043+back/2),(.075,width,back+.014))
    if arms:
        for side in (-1,1):
            y = side*(width/2+.025)
            box(f'Arm_{side+1}',(-depth*.07,y,.05+arms),(.80*depth,.058,.061))
            box(f'ArmPost_{side+1}',(depth*.20,y,.05+arms/2),(.049,.057,arms))
    return parts


def procedural_cases(template,height,pitch,offset_indices,tag,controls=False,only_control=False,
                     yaw_comparison=False):
    cases = []
    for offset in offset_indices:
        dx,dy,yaw = (YAW_OFFSETS if yaw_comparison else OFFSETS)[offset]
        rot = Rotation.from_euler('ZY',[yaw,pitch],degrees=True)
        u,_,n = rot.as_matrix().T
        edge_direction = Rotation.from_euler('y',pitch,degrees=True).as_matrix()[:,0] if yaw_comparison else u
        front = np.array([-.4+dx,dy,height])+.25*edge_direction
        for family,depth,width,shape,back,arms,base in (() if only_control else CHAIRS):
            top = front-(depth/2)*u
            spec = dict(template,label=f'{tag}/{family}/{offset+1}',family=family,shape=shape,
                        depth=depth,width=width,height=float(top[2]),pitch_deg=pitch,yaw=np.deg2rad(yaw),
                        yaw_deg=yaw,x=float(top[0]),y=float(top[1]),top_center=top.tolist(),
                        rotation_xyzw=rot.as_quat().tolist(),normal=n.tolist(),
                        fore_aft_offset_cm=100*dx,lateral_offset_cm=100*dy,
                        reference_height_cm=100*height)
            spec['parts'] = pieces(depth,width,float(top[2]),rot,back,arms,base)
            cases.append(spec)
        if controls or only_control:
            top = front-.25*u
            cases.append(dict(template,label=f'{tag}/plain_plate/{offset+1}',family='plain_plate',
                              depth=.50,width=.90,shape='rect',height=float(top[2]),pitch_deg=pitch,
                              yaw=np.deg2rad(yaw),yaw_deg=yaw,x=float(top[0]),y=float(top[1]),
                              top_center=top.tolist(),rotation_xyzw=rot.as_quat().tolist(),normal=n.tolist(),
                              fore_aft_offset_cm=100*dx,lateral_offset_cm=100*dy,
                              reference_height_cm=100*height,parts=[]))
    return cases


def mesh_spec(family,source):
    """Recover the actual top footprint and frame from the saved collision mesh."""
    root = ASSET_ROOT/source
    mesh = root/'furniture.usda'
    text = mesh.read_text()
    point_text = re.search(r'point3f\[\] points = \[(.*?)\]',text,re.S)[1]
    points = np.array([[float(x) for x in xyz.split(',')] for xyz in re.findall(r'\(([^()]*)\)',point_text)])
    face_text = re.search(r'int\[\] faceVertexIndices = \[(.*?)\]',text,re.S)[1]
    faces = np.fromstring(face_text,sep=',',dtype=int).reshape(-1,3)
    region = read(root/'target_region.json')
    triangles = points[faces[region['triangle_indices']]]
    n = np.asarray(region['mean_normal'])
    assert n[2]>.99
    n /= np.linalg.norm(n)
    heights = triangles@n
    xy = np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
    area = np.linalg.norm(xy,axis=1)/2
    candidate = (np.ptp(heights,axis=1)<.004)&(np.abs(xy@n)>.97*2*area)
    if not candidate.any():
        raise ValueError('No near-planar seat triangles in the saved surface region')
    z = heights[candidate].mean(1)
    plane = max(z,key=lambda a:area[candidate][np.abs(z-a)<.006].sum())
    selected = candidate & (np.max(np.abs(heights-plane),axis=1)<.009)
    if not selected.any():
        raise ValueError('No complete facets in the planar seat region')
    patch = triangles[selected].reshape(-1,3)
    u = np.cross([0,1,0],n)
    u /= np.linalg.norm(u)
    r = np.stack((u,np.cross(n,u),n),axis=1)
    projected = patch@r
    lo,hi = projected.min(0),projected.max(0)
    assert np.max(np.abs(projected[:,2]-plane))<.009
    center = r@np.r_[(lo[:2]+hi[:2])/2,plane]
    return mesh,center,r,hi[:2]-lo[:2]


def mesh_cases(template):
    cases = []
    exclusions = {}
    for family,source in MESHES:
        try:
            mesh,center,frame,extent = mesh_spec(family,source)
        except ValueError as error:
            exclusions[family] = str(error)
            continue
        depth,width = extent
        for i,(dx,dy,yaw) in enumerate((*OFFSETS,(0.,0.,0.)),1):
            pose = Rotation.from_euler('z',yaw,degrees=True)
            orientation = pose*Rotation.from_matrix(frame)
            n = orientation.as_matrix()[:,2]
            top = np.array([-.15+dx,dy,center[2]])-depth/2*orientation.as_matrix()[:,0]
            top[2] = center[2]
            root = top-pose.apply(center)
            # Keep the source furniture grounded and the mapped seat top coplanar.
            assert abs(root[2])<.002
            cases.append(dict(template,label=f'exact_mesh/{family}/{i}',family=family,shape='mesh',
                depth=float(depth),width=float(width),height=float(top[2]),x=float(top[0]),y=float(top[1]),
                pitch_deg=float(np.rad2deg(np.arctan2(n[0],n[2]))),yaw=float(np.deg2rad(yaw)),yaw_deg=yaw,
                normal=n.tolist(),rotation_xyzw=orientation.as_quat().tolist(),top_center=top.tolist(),
                mesh_file=str(mesh),mesh_sha256=sha(mesh),mesh_root=root.tolist(),
                mesh_root_xyzw=pose.as_quat().tolist(),fore_aft_offset_cm=100*dx,lateral_offset_cm=100*dy))
    return cases,exclusions


def scene_worker(launcher,options):
    sys.path.insert(0,str(launcher.parent))
    sys.path[:0] = [str(launcher.parent/'trainer'),str(ROOT/'scripts')]
    from grip_hold_ppo import evaluate_worker
    import run_supported_postures as support
    from humanoid_sitting.tasks import furniture
    from environment import FullBodyEnv
    import isaaclab.sim as sim
    from isaaclab.utils import configclass
    import torch

    original = support.make_cfg

    def spawn_chair(path,cfg,translation=None,orientation=None,**kwargs):
        prim = furniture.spawn_furniture(path,cfg,translation,orientation,**kwargs)
        for part in cfg.parts:
            sim.spawn_cuboid(path+'/'+part['name'],sim.CuboidCfg(size=part['size'],
                collision_props=cfg.collision_props,visual_material=cfg.visual_material,
                physics_material=cfg.physics_material),translation=part['center'],orientation=part['xyzw'])
        return prim

    @configclass
    class ChairSpawnCfg(furniture.FurnitureSpawnCfg):
        func = spawn_chair
        parts: tuple = ()

    def reset_mesh(env,env_ids):
        ids = torch.arange(env.num_envs,device=env.device)
        ids = ids if env_ids is None else ids[env_ids] if isinstance(env_ids,slice) else env_ids
        rows = [env.cfg.geometry_bank[i] for i in furniture.indices(env)[ids].cpu().tolist()]
        pose = torch.as_tensor([g['mesh_root']+g['mesh_root_xyzw'] for g in rows],
                               dtype=torch.float32,device=env.device)
        pose[:,:3] += env.scene.env_origins[ids]
        env.scene['seat'].write_root_pose_to_sim_index(root_pose=pose,env_ids=ids)
        env.scene['seat'].write_root_velocity_to_sim_index(
            root_velocity=torch.zeros(len(ids),6,device=env.device),env_ids=ids)

    def make_cfg(args,geometry):
        cfg = original(args,geometry)
        spawner = cfg.scene.seat.spawn
        assets = []
        for g,inherited in zip(geometry,spawner.assets_cfg,strict=True):
            if 'mesh_file' in g:
                assert sha(g['mesh_file'])==g['mesh_sha256']
                asset = sim.UsdFileCfg(usd_path=g['mesh_file'],activate_contact_sensors=True)
            else:
                asset = ChairSpawnCfg(size=inherited.size,shape=g['shape'],family='slab',
                    top_height=g['height'],parts=tuple(g['parts']),rigid_props=inherited.rigid_props,
                    collision_props=inherited.collision_props,mass_props=inherited.mass_props,
                    visual_material=inherited.visual_material,physics_material=inherited.physics_material,
                    activate_contact_sensors=True)
            assets.append(asset)
        spawner.assets_cfg = assets
        if 'mesh_file' in geometry[0]:
            assert all('mesh_file' in g for g in geometry)
            cfg.events.reset_seats.func = reset_mesh
            cfg.scene.env_spacing = 4.
        return cfg

    support.make_cfg = make_cfg
    initialize = FullBodyEnv.__init__

    def initialize_with_shape_audit(self,*args,**kwargs):
        initialize(self,*args,**kwargs)
        from pxr import Usd, UsdPhysics
        usd = sim.get_current_stage()
        audited = []
        for env_path,g in zip(self.raw.scene.env_prim_paths,self.cfg.geometry_bank,strict=True):
            root = usd.GetPrimAtPath(env_path+'/Seat')
            body = UsdPhysics.RigidBodyAPI(root)
            assert root.IsValid() and body.GetKinematicEnabledAttr().Get()
            prefixes = ['/Mesh'] if 'mesh_file' in g else ['/geometry',*['/'+p['name'] for p in g['parts']]]
            active = [str(prim.GetPath()).removeprefix(str(root.GetPath()))
                      for prim in Usd.PrimRange(root,Usd.TraverseInstanceProxies())
                      if prim.HasAPI(UsdPhysics.CollisionAPI)
                      and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()]
            assert all(any(c.startswith(k+'/') or c==k for c in active) for k in prefixes),(g['label'],active)
            audited.append(dict(id=g['label'],family=g['family'],enabled_colliders=active))
        write(self.out/'scene_assets_verified.json',audited)

    FullBodyEnv.__init__ = initialize_with_shape_audit
    assert options[0]=='evaluate'
    checkpoint = Path(options[1])
    sys.argv[1:] = options[2:]
    evaluate_worker(checkpoint)


def frozen_controller():
    frozen = read(FROZEN)
    for path,digest in frozen['inputs'].items():
        assert sha(path)==digest,path
    command = frozen['command']
    assert frozen['gpu']==GPU and Path(command[2]).name=='grip_hold_ppo.py'
    return frozen,command[3:command.index('--mode')]


def record(batch,geometry):
    audit,episodes = verify_audit(batch/'evaluation')
    manifest = read(batch/'evaluation/manifest.json')
    assert manifest['action_limits_unchanged'] and manifest['physical_criteria_unchanged']
    result = read(batch/'evaluation/results.json')
    assert result['success_criteria_unchanged'] and len(episodes)==len(geometry)
    asset_rows = read(batch/'evaluation/scene_assets_verified.json')
    assert [r['id'] for r in asset_rows]==[g['label'] for g in geometry]
    rows = []
    for g,r,w in zip(geometry,episodes,audit['final_absolute_mechanical_power_W'],strict=True):
        assert r['id']==g['label'] and r['full_standing_start'] and r['steps']==1000
        rows.append(dict(id=r['id'],family=g['family'],status='pass' if r['clean'] else 'fail',
                         height_cm=100*g['height'],pitch_deg=g['pitch_deg'],yaw_deg=g['yaw_deg'],
                         fore_aft_offset_cm=g['fore_aft_offset_cm'],lateral_offset_cm=g['lateral_offset_cm'],
                         hold_s=r['final_hold_s'],power_w=w,slip_ticks=r['slip_ticks'],
                         illegal_ticks=r['illegal_ticks'],forbidden_contact=r['physical_failures'][1]))
    return rows


def run_batch(batch,source,geometry,inputs,options,launcher,protocol,deadline,disk_reserve):
    bank = batch/'bank'
    assert time.monotonic()+EVAL_WALL<=deadline
    batch.mkdir()
    bank_from_geometry(bank,source,geometry)
    write(batch/'reservation.json',dict(files={str(p):sha(p) for p in bank.iterdir()}))
    try:
        stage(batch,'evaluation',bank,PREPARED/'library.npz',worker=Path(__file__),
              extra=('--worker',str(launcher),*options),protocol=protocol,
              inputs=[*inputs,*(Path(g['mesh_file']) for g in geometry if 'mesh_file' in g)],
              disk_reserve=disk_reserve)
    except AssertionError:
        startup = batch/'evaluation/startup.json'
        if not startup.exists():
            raise
        health = read(startup)
        invalid = [i for i,(warm,sensor) in enumerate(zip(health['warm_failure'],health['sensor_valid'],strict=True))
                   if warm or not sensor]
        if not invalid or (batch/'evaluation/trajectory.npz').exists():
            raise
        rows = [dict(id=geometry[i]['label'],family=geometry[i]['family'],
                     status='startup failure' if health['warm_failure'][i] else 'depth unavailable',
                     height_cm=100*geometry[i]['height'],pitch_deg=geometry[i]['pitch_deg'],
                     yaw_deg=geometry[i]['yaw_deg'],fore_aft_offset_cm=geometry[i]['fore_aft_offset_cm'],
                     lateral_offset_cm=geometry[i]['lateral_offset_cm'],hold_s=None,power_w=None,
                     slip_ticks=None,illegal_ticks=None,forbidden_contact=None) for i in invalid]
        remaining = [g for i,g in enumerate(geometry) if i not in invalid]
        if remaining:
            rows += run_batch(batch/'sensor_valid',source,remaining,inputs,options,launcher,protocol,deadline,
                              disk_reserve)
        return rows
    return record(batch,geometry)


def publish(out,rows):
    groups = defaultdict(list)
    for r in rows:
        groups[r['family']].append(r)
    assert len(rows)==len({r['id'] for r in rows})
    summary = {}
    for name,subset in groups.items():
        scored = [r for r in subset if r['status'] in ('pass','fail')]
        passed = [r for r in scored if r['status']=='pass']
        summary[name] = dict(attempted=len(subset),scored=len(scored),passed=len(passed),
                             slipped=sum(r['slip_ticks']>0 for r in scored),
                             forbidden=sum(r['forbidden_contact'] for r in scored),
                             passing_mean_power_w=float(np.mean([r['power_w'] for r in passed])) if passed else None)
    write(out/'summary.json',dict(attempted=len(rows),families=summary))
    temp = out/'results.csv.tmp'
    with temp.open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(out/'results.csv')


def existing_scores(path):
    with path.open(newline='') as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for count in ('slip_ticks','illegal_ticks'):
            row[count] = int(row[count]) if row[count] else None
        row['power_w'] = float(row['power_w']) if row['power_w'] else None
        row['forbidden_contact'] = {'True':True,'False':False,'':None}[row['forbidden_contact']]
    return rows


def run(out,mode,resume=False,disk_reserve=MIN_DISK):
    start = time.monotonic()
    frozen,options = frozen_controller()
    launcher = Path(frozen['command'][2])
    source = Path(read(TOUCHDOWN_REFERENCE/'candidate/manifest.json')['args']['bank'])
    template = read(source/'geometry.json')[0]
    exclusions = {}
    if mode=='pilot':
        batches = [('pilot',procedural_cases(template,.402,6.5,(2,3),tag='pilot',controls=True))]
    elif mode=='meshes':
        meshes,exclusions = mesh_cases(template)
        batches = [('exact_meshes',meshes)]
    elif mode=='yaw':
        batches = [(f'yaw_{i+1:02d}',[
            g for j,h in enumerate((.367,.419))
            for g in procedural_cases(template,h,p,range(len(YAW_OFFSETS)),
                                      f'yaw_{i+1:02d}/{j+1}',yaw_comparison=True)
        ]) for i,p in enumerate((1.,8.,17.5))]
    else:
        batches = []
        controls = []
        for offset in range(len(OFFSETS)):
            for phase in range(len(PITCHES)):
                name = f'batch_{len(batches)+1:02d}'
                geometry = []
                for i,height in enumerate(HEIGHTS):
                    pitch = PITCHES[(i+phase)%len(PITCHES)]
                    shift = ((i+offset)%len(OFFSETS),)
                    tag = f'{name}/{i+1}'
                    geometry += procedural_cases(template,height,pitch,shift,tag)
                    controls += procedural_cases(template,height,pitch,shift,tag,only_control=True)
                batches.append((name,geometry))
        batches += [('matched_plates_01',controls[:72]),('matched_plates_02',controls[72:])]
    protocol = out/'protocol.json'
    design = dict(mode=mode,controller_freeze=str(FROZEN),gpu=GPU,wall_budget_s=GPU_BUDGET_S,
                  frozen_before_rollouts=True,scorer='original support posture scorer',
                  source_mesh_exclusions=exclusions,banking={name:geometry for name,geometry in batches})
    rows = []
    deadline = start+GPU_BUDGET_S
    if resume:
        assert read(protocol)==design
        rows = existing_scores(out/'results.csv')
        elapsed_gpu = sum(read(p)['wall_s'] for p in out.rglob('evaluation.exit.json'))
        deadline -= elapsed_gpu
        index = len(list(out.glob('resume_receipt*.json')))+1
        write(out/f'resume_receipt_{index:02d}.json',dict(original_plan_sha256=sha(protocol),attempted=len(rows),
              output_reserve_bytes=disk_reserve,prior_gpu_worker_wall_s=elapsed_gpu))
    else:
        out.mkdir()
        write(protocol,design)
    inputs = [FROZEN,source/'bank.npz',*map(Path,frozen['inputs'])]
    known = {r['id'] for r in rows}
    for name,geometry in batches:
        assert len(geometry)<=72
        done = [g['label'] in known for g in geometry]
        if all(done):
            continue
        assert not any(done),name
        batch = out/name
        if resume and (batch/'evaluation/independent_audit.json').exists():
            saved = read(batch/'bank/geometry.json')
            assert saved==[dict(g,case=i,surface_id=i,start_stage=0) for i,g in enumerate(geometry)]
            rows += record(batch,geometry)
            publish(out,rows)
            continue
        # Allow 6 MB per trial above the reserve.
        if shutil.disk_usage(out).free<disk_reserve+6_000_000*len(geometry):
            break
        if time.monotonic()+EVAL_WALL>deadline:
            break
        if batch.exists():
            assert resume and not list(batch.rglob('independent_audit.json')),batch
            batch = out/(name+'_resumed')
        rows += run_batch(batch,source,geometry,inputs,options,launcher,protocol,deadline,disk_reserve)
        publish(out,rows)
        print(f'{len(rows)} of {sum(len(g) for _,g in batches)} attempted',flush=True)
    write(out/'status.json',dict(attempted=len(rows),planned=sum(len(g) for _,g in batches),
                                 elapsed_s=time.monotonic()-start))


if __name__=='__main__':
    if sys.argv[1]=='--worker':
        scene_worker(Path(sys.argv[2]),sys.argv[3:])
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('output',type=Path)
        parser.add_argument('--mode',choices=('pilot','meshes','campaign','yaw'),required=True)
        parser.add_argument('--resume',action='store_true')
        parser.add_argument('--reserve-gb',type=int,choices=(18,20),default=20)
        args = parser.parse_args()
        if args.reserve_gb==18:
            assert args.resume and args.output.resolve().is_relative_to(TOUCHDOWN_REFERENCE.parent)
        run(args.output.resolve(),args.mode,args.resume,args.reserve_gb*1_000_000_000)
