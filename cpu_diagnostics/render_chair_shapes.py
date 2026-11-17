"""Render representative audited chair rollouts and their actual colliders."""

import argparse
from collections import defaultdict
from pathlib import Path
import re
import subprocess

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

import render_steep_seat_clearance as replay
from evaluate_chair_shapes import CHAIRS, MESHES


def cases(studies):
    groups = defaultdict(list)
    for study in studies:
        for path in sorted(study.rglob('independent_audit.json')):
            episode = path.parent
            manifest = replay.read(episode/'manifest.json')
            geometry = replay.read(Path(manifest['args']['bank'])/'geometry.json')
            result = replay.read(episode/'results.json')
            audit = replay.read(path)
            assert audit['passed'] and audit['strict_predicates_exact']
            for i,(g,r,w) in enumerate(zip(geometry,result['episodes'],
                                          audit['final_absolute_mechanical_power_W'],strict=True)):
                assert g['label']==r['id']
                groups[g['family']].append((episode,i,g,r,w))
    selected = []
    order = [shape[0] for shape in (*CHAIRS,*MESHES)]+['plain_plate']
    for family in order:
        rows = groups[family]
        if not rows:
            continue
        yes = [x for x in rows if x[3]['clean']]
        no = [x for x in rows if not x[3]['clean']]
        def central(x):
            g = x[2]
            return abs(g['height']-.4)+abs(g['pitch_deg']-7)/500+abs(g['yaw_deg'])/800
        if yes and no:
            selected += [min(yes,key=central),min(no,key=central)]
        else:
            selected += sorted(rows,key=central)[:2]
    return selected


def physical_mesh(path):
    text = Path(path).read_text()
    data = re.search(r'point3f\[\] points = \[(.*?)\]',text,re.S)[1]
    vertices = np.array([[float(x) for x in xyz.split(',')] for xyz in re.findall(r'\(([^()]*)\)',data)])
    data = re.search(r'int\[\] faceVertexIndices = \[(.*?)\]',text,re.S)[1]
    faces = np.fromstring(data,sep=',',dtype=int).reshape(-1,3)
    return vertices,faces


def stage_vertices(g,cube,faces):
    if 'mesh_file' in g:
        v,f = physical_mesh(g['mesh_file'])
        transform = Rotation.from_quat(g['mesh_root_xyzw'])
        return [(transform.apply(v)+g['mesh_root'],f)]
    shape = g['shape']
    if shape=='ellipse':
        angle = np.arange(16)*2*np.pi/16
        xy = np.c_[np.cos(angle),np.sin(angle)]*.5
    elif shape=='chamfered':
        xy = [(-.35,-.5),(.35,-.5),(.5,-.35),(.5,.35),(.35,.5),(-.35,.5),(-.5,.35),(-.5,-.35)]
    else:
        xy = None
    v = cube if xy is None else np.array([[x,y,z] for z in (-.5,.5) for x,y in xy])
    f = faces if xy is None else ConvexHull(v).simplices
    rot = Rotation.from_quat(g['rotation_xyzw'])
    origin = np.asarray(g['top_center'])-.05*np.asarray(g['normal'])
    result = [(rot.apply(v*[g['depth'],g['width'],.10])+origin,f)]
    for part in g['parts']:
        part_rot = Rotation.from_quat(part['xyzw'])
        v = part_rot.apply(cube*part['size'])+part['center']
        result.append((rot.apply(v)+origin,faces))
    return result


def render(studies,destination):
    selected = cases(studies)
    assert selected and len(selected)<=32
    indices = defaultdict(list)
    for folder,index,*_ in selected:
        indices[folder].append(index)
    loaded = {}
    for folder,keep in indices.items():
        audit = replay.read(folder/'independent_audit.json')
        trace = folder/'trajectory.npz'
        assert replay.digest(trace)==audit['files']['trajectory.npz']
        with np.load(trace,allow_pickle=False) as z:
            body,held,no_slip,legal = (z[name][:,keep] for name in ('body','held_ticks','no_slip','legal'))
        result = replay.read(folder/'results.json')
        for j,i in enumerate(keep):
            pose = body[:,j]
            pose[:,:,:3] -= np.asarray(result['origins'][i])
            loaded[folder,i] = (pose,held[:,j]*.02,np.maximum.accumulate(~no_slip[:,j]|~legal[:,j]),
                                result['body_names'])
    replay.PANEL = (365,246)
    renderer = replay.Renderer()
    renderer.scale = 195
    renderer.offset = np.array([182.,128.,0.])-np.array([-.28,0.,.5])@replay.VIEW*renderer.scale
    frames = []
    for folder,index,g,row,power in selected:
        pose,held,events,names = loaded[folder,index]
        frames.append((stage_vertices(g,renderer.cube,renderer.faces),pose,held,events,names,g,row,power))

    size = (1600,820)
    batches = [frames[i:i+8] for i in range(0,len(frames),8)]
    with subprocess.Popen(['ffmpeg','-nostdin','-v','error','-n','-f','rawvideo','-pix_fmt','rgb24',
                           '-s','1600x820','-r','15','-i','-','-an','-c:v','libx264','-threads','2',
                           '-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',
                           str(destination)],stdin=subprocess.PIPE) as pipe:
        for group,rows in enumerate(batches,1):
            for tick in [*np.linspace(0,999,150).astype(int),*([999]*45)]:
                canvas = Image.new('RGB',size,replay.BACKGROUND)
                draw = ImageDraw.Draw(canvas)
                replay.label(draw,(35,21),'Sitting on varied rigid furniture',35,bold=True)
                replay.label(draw,(36,71),f'Saved 4090 rollouts   ·   Group {group}/{len(batches)}   ·   {(tick+1)*.02:.2f} s',22,replay.MUTED)
                for j,(parts,pose,held,events,names,g,row,w) in enumerate(rows):
                    x,y = 35+387*(j%4),111+335*(j//4)
                    replay.label(draw,(x,y),g['family'].replace('_',' ').title(),22,bold=True)
                    replay.label(draw,(x,y+28),f'{100*g["height"]:.1f} cm  ·  Tilt {g["pitch_deg"]:.1f}°  ·  Yaw {g["yaw_deg"]:.0f}°',
                                 17,replay.MUTED)
                    im = np.full((246,365,3),[230,237,240],np.uint8)
                    depth = np.full((246,365),-np.inf,np.float32)
                    renderer.draw(im,depth,renderer.cube*[1.65,1.4,.022]+[-.29,0.,-.011],
                                  renderer.faces,[204,216,221])
                    for k,(v,f) in enumerate(parts):
                        renderer.draw(im,depth,v,f,[102,145,165] if k==0 else [87,119,129])
                    for k,(pos,quat) in enumerate(zip(pose[tick,:,:3],pose[tick,:,3:],strict=True)):
                        if names[k] in renderer.parts:
                            v,f,color = renderer.parts[names[k]]
                            renderer.draw(im,depth,Rotation.from_quat(quat).apply(v)+pos,f,color)
                    canvas.paste(Image.fromarray(im),(x,y+51))
                    if tick==999:
                        score = f'{"PASS" if row["clean"] else "FAIL"}  ·  {row["final_hold_s"]:.2f} s  ·  {w:.2f} W'
                        color = replay.GREEN if row['clean'] else replay.RED
                    else:
                        score = f'{held[tick]:.2f} s qualifying hold'
                        color = replay.RED if events[tick] else replay.INK
                    replay.label(draw,(x,y+304),score,20,color,True)
                replay.label(draw,(35,792),'Representative trials  ·  Same scoring  ·  Mean mechanical power over the last 5 s',17,replay.MUTED)
                pipe.stdin.write(canvas.tobytes())
        pipe.stdin.close()
        assert pipe.wait(timeout=30)==0


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('studies',type=Path,nargs='+')
    args = parser.parse_args()
    render(args.studies,args.output)
