"""Build the catalog grid from recorded robot poses and the tested mesh assets."""
import argparse
from functools import lru_cache
from itertools import product
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import fast_simplification
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pxr import Usd, UsdGeom
from scipy.spatial.transform import Rotation

from furniture_skill_comparison import METHODS, SIZES, STUDY, read, sha, write
from gallery import vis


FPS = 10
WIDTH, SCENE, LABEL = 192, 182, 38
ROW_HEIGHT = SCENE + LABEL
BACKGROUND = '#f4f7f7'
INK = '#14262e'
SUBTLE = '#465d69'
PIECE_COLOR, UNIFIED_COLOR = '#095a55', '#34496a'
DISPLAY = dict(piecewise='Piece-wise', unified='Unified')
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans'


@lru_cache(maxsize=None)
def font(size, bold=False):
    return ImageFont.truetype(FONT+('-Bold' if bold else '')+'.ttf',size)


def paint_text(draw, xy, value, size, color=INK, bold=False):
    draw.text(xy,value,font=font(size,bold),fill=color)


def mesh(geometry):
    stage = Usd.Stage.Open(geometry['mesh_file'])
    prim = UsdGeom.Mesh.Get(stage,'/Furniture/Mesh')
    xyz = np.asarray(prim.GetPointsAttr().Get(),dtype='d')+geometry['mesh_root']
    triangles = np.array(prim.GetFaceVertexIndicesAttr().Get(),dtype='i',copy=True).reshape(-1,3)
    return xyz,triangles


def rgb(r,g,b):
    return np.asarray([r,g,b], dtype='B')


def camera():
    eye = np.array([.31,-.93,.195])
    eye /= np.linalg.norm(eye)
    across = np.cross([0.,0.,1.],eye)
    across /= np.linalg.norm(across)
    return np.stack([across,np.cross(across,eye),eye],axis=1)


def furniture_name(name):
    return name.replace('_',' ').replace('MapleBarStool','Maple Barstool').replace('EastRural','East Rural').replace('CapriceBackless','Caprice Backless')


def reason_label(result):
    if result['passed']:
        return f"PASS  {result['terminal_hold_s']:.2f} s"
    reason = result['reason']
    if 'Forbidden contact' in reason:
        return 'FAIL  Contact'
    if 'slip' in reason.lower():
        return 'FAIL  Slip'
    if 'impact' in reason.lower():
        return 'FAIL  Impact'
    if 'hold' in reason.lower():
        return 'FAIL  Short hold'
    if 'floor' in reason.lower():
        return 'FAIL  Foot lift'
    return 'FAIL  Safety stop'


class CatalogView:
    def __init__(self, body_names):
        vis.PANEL, vis.VIEW = (WIDTH, SCENE), camera()
        self.view = vis.VIEW
        self.renderer = vis.Renderer()
        verts,faces,links,colors = [],[],[],[]
        stop = 0
        for i,name in enumerate(body_names):
            if name not in self.renderer.parts:
                continue
            xyz, tri, _ = self.renderer.parts[name]
            color = rgb(32,112,137) if name == 'pelvis' else rgb(48,69,87)
            if name == 'head_link':
                color = rgb(96,126,145)
            verts.append(xyz)
            faces.append(tri+stop)
            links.extend([i]*len(xyz))
            colors.append(np.tile(color,(len(tri),1)))
            stop += len(xyz)
        self.verts, self.faces = np.vstack(verts), np.ascontiguousarray(np.vstack(faces), dtype='i')
        self.links = np.asarray(links)
        self.colors = np.ascontiguousarray(np.vstack(colors),dtype='B')

    def background(self, native, sampled):
        native_xyz, _ = mesh(native)
        xyz, faces = mesh(sampled)
        if len(faces)>7500:
            xyz, faces = fast_simplification.simplify(xyz,faces,target_count=7500)
        # Use the native-scale framing for both methods and all sizes so
        # furniture size changes remain visible against the same robot.
        robot_space = np.array(list(product((-.62,.38),(-.37,.37),(-.035,1.32))))
        projected = np.vstack((native_xyz,robot_space))@self.view
        low, high = projected.min(0), projected.max(0)
        r = self.renderer
        r.scale = min((WIDTH-12)/(high[0]-low[0]),(SCENE-14)/(high[1]-low[1]))
        r.offset = np.array([WIDTH/2,SCENE/2,0.])-(low+high)*.5*r.scale
        im = np.full((SCENE,WIDTH,3),rgb(222,233,238),'B')
        depth = np.full((SCENE,WIDTH),-np.inf,'f')
        r.draw(im,depth,r.cube*[5.,5.,.016]+[-.3,0,-.009],r.faces,rgb(200,213,220))
        r.draw(im,depth,xyz,faces,rgb(177,134,96))
        return im,depth,r.scale,r.offset.copy()

    def at(self, scene, pose, rotations):
        bg,depth,scale,offset = scene
        im, zbuf = bg.copy(), depth.copy()
        verts = np.einsum('nij,nj->ni',rotations[self.links],self.verts)+pose[self.links,:3]
        xy = np.ascontiguousarray(verts@(self.view*scale)+offset,dtype='f')
        self.renderer.paint(im,zbuf,WIDTH,SCENE,xy,self.faces,self.colors,len(self.faces))
        return Image.fromarray(im)


def encode_row(method, scale):
    out = STUDY/'rows'
    out.mkdir(exist_ok=True)
    dest = out/f'{method}_{scale}.mp4'
    if dest.is_file() and dest.with_suffix('.json').is_file():
        assert read(dest.with_suffix('.json'))['video_sha256']==sha(dest)
        return
    folder = STUDY/f'{method}_{scale}'
    ev = folder/'evaluation'
    result = read(ev/'results.json')
    scores = read(folder/'scores.json')
    trace_file = ev/'trajectory.npz'
    audit = read(ev/'independent_audit.json')
    digest = sha(trace_file)
    assert audit['passed'] and digest==audit['files']['trajectory.npz']
    current = read(folder/'bank'/'geometry.json')
    native = read(STUDY/'banks'/'100'/'geometry.json')
    assert [g['name'] for g in native]==[g['name'] for g in current]==[r['asset'] for r in scores]
    view = CatalogView(result['body_names'])
    backgrounds = [view.background(a,b) for a,b in zip(native,current,strict=True)]
    with np.load(trace_file,allow_pickle=False) as z:
        poses = z['body'].copy()
        poses[:,:,:,:3] -= np.asarray(result['origins'])[None,:,None,:]
        active = z['learn']
        guards = z['flags']
    last = []
    for i in range(len(current)):
        ended = np.flatnonzero(guards[:,i].any(1))
        if len(ended):
            last.append(int(ended[0]))
        else:
            ticks = np.flatnonzero(active[:,i])
            last.append(int(ticks[-1]) if len(ticks) else 0)
    snapshots = np.minimum(np.arange(0,1000,50//FPS)[:,None],np.asarray(last)[None,:])
    pose_frame = poses[snapshots,np.arange(len(current))[None,:]]
    rotations = Rotation.from_quat(pose_frame[...,3:].reshape(-1,4)).as_matrix()
    rotations = rotations.reshape(*pose_frame.shape[:-1],3,3)
    row_width = WIDTH*len(scores)
    base = Image.new('RGB',(row_width,ROW_HEIGHT),BACKGROUND)
    d = ImageDraw.Draw(base)
    for i,r in enumerate(scores):
        x = WIDTH*i
        fill, ink = ('#d9ede7','#11564b') if r['passed'] else ('#f3e5df','#873e35')
        d.rectangle((x+1,SCENE,x+WIDTH-2,ROW_HEIGHT-2),fill=fill)
        paint_text(d,(x+9,SCENE+6),reason_label(r),21,ink,True)
    pipe = subprocess.Popen(['ffmpeg','-nostdin','-v','error','-n','-f','rawvideo','-pix_fmt','rgb24',
        '-s',f'{row_width}x{ROW_HEIGHT}','-r',str(FPS),'-i','-','-an','-c:v','libx264',
        '-threads','3','-preset','veryfast','-crf','17','-pix_fmt','yuv420p',str(dest)],stdin=subprocess.PIPE)
    cache = [None]*len(scores)
    previous = [-1]*len(scores)
    try:
        with pipe.stdin as stream:
            for j,k in enumerate(range(0,1000,50//FPS)):
                frame = base.copy()
                for i,bg in enumerate(backgrounds):
                    if previous[i]!=snapshots[j,i]:
                        cache[i] = view.at(bg,pose_frame[j,i],rotations[j,i])
                        if last[i] < 999 and k>=last[i]:
                            draw = ImageDraw.Draw(cache[i])
                            draw.rounded_rectangle((7,6,94,30),radius=5,fill='#253b47')
                            paint_text(draw,(12,9),f'Stop {last[i]/50:.1f} s',14,'#ffffff',True)
                        previous[i] = snapshots[j,i]
                    frame.paste(cache[i],(i*WIDTH,0))
                stream.write(frame.tobytes())
        assert pipe.wait()==0
    finally:
        if pipe.poll() is None:
            pipe.kill()
            pipe.wait()
    write(dest.with_suffix('.json'),dict(method=method,scale_pct=scale,frames=len(snapshots),
          trace_sha256=digest,video_sha256=sha(dest),body_visuals_sha256=sha(vis.VISUALS)))
    print('rendered',method,scale,flush=True)


def wrap_name(draw, value, max_width, size):
    words = value.split()
    lines = ['']
    for word in words:
        merged = (lines[-1]+' '+word).strip()
        if lines[-1] and draw.textlength(merged,font=font(size,True))>max_width:
            lines.append(word)
        else:
            lines[-1] = merged
    return lines


def layout(title, start, count, tile, row_height, stem, out):
    protocol = read(STUDY/'protocol.json')
    roster = protocol['roster'][start:start+count]
    header, left, footer = 134, 208, 42
    width, height = left+tile*count+16, header+8*row_height+footer
    width += width%2
    height += height%2
    backdrop = Image.new('RGB',(width,height),BACKGROUND)
    d = ImageDraw.Draw(backdrop)
    title_size = next(s for s in range(30,17,-1) if d.textlength(title,font=font(s,True)) <= width-36)
    subtitle = 'Matched standing starts  ·  Recorded simulation  ·  20 s at real time'
    sub_size = next(s for s in range(17,10,-1) if d.textlength(subtitle,font=font(s)) <= width-36)
    paint_text(d,(18,13),title,title_size,bold=True)
    paint_text(d,(19,54),subtitle,sub_size,SUBTLE)
    d.line((18,78,width-16,78),fill='#b7c8cf',width=1)
    for j,name in enumerate(roster):
        x = left+j*tile
        size = 15 if tile<=128 else 17
        for line,value in enumerate(wrap_name(d,furniture_name(name),tile-10,size)[:2]):
            paint_text(d,(x+6,90+line*18),value,size,bold=True)
    row_order = list(product(SIZES,METHODS))
    for i,(scale,method) in enumerate(row_order):
        y = header+i*row_height
        color = PIECE_COLOR if method=='piecewise' else UNIFIED_COLOR
        d.rectangle((16,y,left-6,y+row_height-3),fill=color)
        scores = read(STUDY/f'{method}_{scale}'/'scores.json')[start:start+count]
        paint_text(d,(28,y+12),f'{scale}%',31,'#ffffff',True)
        paint_text(d,(29,y+49),DISPLAY[method],20,'#ffffff',True)
        paint_text(d,(29,y+80),f'{sum(r["passed"] for r in scores)}/{count} passed',16,'#ffffff')
    paint_text(d,(18,height-31),'Pass requires a 5-second supported hold with both feet planted.',15,SUBTLE)
    card = out/(stem+'_layout.png')
    backdrop.save(card)
    command = ['ffmpeg','-nostdin','-v','error','-n','-filter_complex_threads','3','-loop','1','-framerate',str(FPS),'-i',str(card)]
    filters = []
    for i,(scale,method) in enumerate(row_order,1):
        source = STUDY/'rows'/f'{method}_{scale}.mp4'
        assert read(source.with_suffix('.json'))['video_sha256']==sha(source)
        command += ['-threads','1','-i',str(source)]
        filters.append(f'[{i}:v]crop={WIDTH*count}:{ROW_HEIGHT}:{WIDTH*start}:0,scale={tile*count}:{row_height}[r{i}]')
    filters.append(''.join(f'[r{i}]' for i in range(1,9))+'vstack=inputs=8[rows]')
    filters.append(f'[0:v][rows]overlay={left}:{header}:shortest=1,format=yuv420p[out]')
    path = out/(stem+'.mp4')
    command += ['-filter_complex',';'.join(filters),'-map','[out]','-an','-t','20','-threads','4','-c:v','libx264',
                '-preset','fast','-crf','18','-movflags','+faststart',str(path)]
    subprocess.run(command,check=True)
    return dict(inspect_video(path,20), cases=roster)


def inspect_video(path,seconds):
    probe = json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-count_frames',
        '-show_entries','stream=width,height,duration,nb_read_frames','-of','json',str(path)]))['streams'][0]
    assert float(probe['duration'])==seconds and int(probe['nb_read_frames'])==FPS*seconds
    return dict(file=path.name,sha256=sha(path),geometry=probe)


def closeup(out):
    sections = []
    video = out/'sitting_grid_closeup.mp4'
    with TemporaryDirectory(dir=out) as directory:
        temp = Path(directory)
        listing = temp/'sections.txt'
        for start in range(0,30,5):
            section = layout(f'Furniture {start+1:02d}–{start+5:02d} of 30',start,5,192,220,
                             f'section_{start:02d}',temp)
            sections.append(dict(start_s=4*start,cases=section['cases']))
        listing.write_text(''.join(f"file 'section_{i:02d}.mp4'\n" for i in range(0,30,5)))
        subprocess.run(['ffmpeg','-nostdin','-v','error','-n','-f','concat','-safe','1','-i',str(listing),
                        '-c','copy','-movflags','+faststart',str(video)],check=True)
    return dict(inspect_video(video,120),chapters=sections)


def assemble():
    out = STUDY/'media'
    out.mkdir(exist_ok=True)
    files = [layout('Sitting on 30 furniture pieces',0,30,120,140,'sitting_grid_30',out),
             layout('Sitting on Monarch and Petite Chair',30,2,192,220,'sitting_grid_supplement',out),
             closeup(out)]
    write(out/'video_manifest.json',dict(replay=True,scales=SIZES,methods=METHODS,files=files,
          renderer_sha256=sha(Path(__file__)),catalog_protocol_sha256=sha(STUDY/'protocol.json'),
          rows=[read(STUDY/'rows'/f'{method}_{scale}.json') for scale,method in product(SIZES,METHODS)]))
    print('verified 8-row grid and supplement',flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=('row','assemble'))
    parser.add_argument('--method',choices=METHODS)
    parser.add_argument('--scale',type=int,choices=SIZES)
    args = parser.parse_args()
    if args.operation == 'row':
        encode_row(args.method,args.scale)
    else:
        assemble()
