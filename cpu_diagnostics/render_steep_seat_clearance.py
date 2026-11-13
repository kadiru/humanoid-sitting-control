"""Render a before/after replay from the two saved steep-seat trajectories."""

import argparse
import ctypes
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation


ROOT = Path('/home/kadir/humanoid-sitting')
VISUALS = ROOT/'outputs/capability_review_20261105_v1/visuals.npz'
RASTER = ROOT/'experiments/furniture_saved_review_20261104_v1/raster.so'
FPS = 20
SIZE = (1440, 840)
PANEL = (670, 520)
BACKGROUND = '#f3f6f7'
INK = '#182e39'
MUTED = '#526570'
GREEN = '#076850'
RED = '#aa362b'
RIGHT = np.array([.92, np.sqrt(1-.92**2), 0.])
EYE = np.array([RIGHT[1]*.99,-RIGHT[0]*.99,.19]); EYE /= np.linalg.norm(EYE)
VIEW = np.stack([RIGHT,-np.cross(EYE,RIGHT),EYE],axis=1)


def read(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


@lru_cache(None)
def font(size, bold=False):
    suffix = '-Bold' if bold else ''
    return ImageFont.truetype(f'/usr/share/fonts/truetype/dejavu/DejaVuSans{suffix}.ttf',size)


def label(draw, at, value, size, color=INK, bold=False):
    draw.text(at,value,font=font(size,bold),fill=color)


def data(folder):
    result, audit = read(folder/'results.json'), read(folder/'independent_audit.json')
    trace = folder/'trajectory.npz'
    assert audit['passed'] and digest(trace)==audit['files']['trajectory.npz']
    indices = [next(i for i,r in enumerate(result['episodes']) if r['id']==name)
               for name in ('known_old/6','known_old/7')]
    with np.load(trace,allow_pickle=False) as z:
        poses = z['body'][:,indices].copy()
        poses[:,:,:,:3] -= np.asarray(result['origins'])[indices][None,:,None,:]
        rows = [result['episodes'][i] for i in indices]
        force = z['physical_force_w'][:,indices][:,:,:5,2].sum(-1)/z['physical_bodyweight_N'][:,indices]
        return dict(poses=poses, rows=rows, bodies=result['body_names'], source=str(trace),
                    sha256=audit['files']['trajectory.npz'], load=force, held=z['held_ticks'][:,indices]*.02,
                    forbidden=z['wrong_force'][:,indices], flags=z['flags'][:,indices],
                    geometry=[read(ROOT/'outputs/full_body_contact_imitation_v4/dataset/sitting/geometry.json')[i] for i in (6,7)])


class Renderer:
    def __init__(self):
        self.paint = ctypes.CDLL(str(RASTER)).paint
        ptr = np.ctypeslib.ndpointer
        self.paint.argtypes = [ptr(np.uint8,flags='C'),ptr(np.float32,flags='C'),ctypes.c_int,ctypes.c_int,
                               ptr(np.float32,flags='C'),ptr(np.int32,flags='C'),ptr(np.uint8,flags='C'),ctypes.c_int]
        self.paint.restype = None
        self.parts = {}
        with np.load(VISUALS,allow_pickle=False) as z:
            for key in z.files:
                if key.endswith('_v'):
                    name = key[:-2]
                    color = [191,206,215]
                    if name in ('left_hip_pitch_link','right_hip_pitch_link'):
                        color = [203,114,52]
                    if name == 'pelvis':
                        color = [43,134,119]
                    if name in ('head_link','left_ankle_roll_link','right_ankle_roll_link'):
                        color = [82,107,126]
                    self.parts[name] = (z[key],z[name+'_f'],color)
        self.cube = np.array([[x,y,z] for x in (-.5,.5) for y in (-.5,.5) for z in (-.5,.5)])
        self.faces = ConvexHull(self.cube).simplices
        self.scale = 365
        self.offset = np.array([PANEL[0]/2,PANEL[1]/2,0])-np.array([-.20,0,.57])@VIEW*self.scale

    def draw(self, image, depth, vertices, faces, color):
        projected = np.ascontiguousarray(vertices@VIEW*self.scale+self.offset,np.float32)
        self.paint(image,depth,*PANEL,projected,np.ascontiguousarray(faces,np.int32),
                   np.tile(color,(len(faces),1)).astype(np.uint8),len(faces))

    def panel(self, sample, case, tick):
        w,h = PANEL
        image = np.full((h,w,3),[230,237,240],np.uint8)
        depth = np.full((h,w),-np.inf,np.float32)
        g = sample['geometry'][case]
        surface = (self.cube*[g['depth'],g['width'],.1]) @ Rotation.from_quat(g['rotation_xyzw']).as_matrix().T
        surface += np.array(g['top_center'])-.05*np.array(g['normal'])
        self.draw(image,depth,self.cube*[1.40,1.2,.022]+[-.21,0,-.011],self.faces,[204,216,221])
        self.draw(image,depth,surface,self.faces,[104,145,165])
        pose = sample['poses'][tick,case]
        rotations = Rotation.from_quat(pose[:,3:]).as_matrix()
        for i,name in enumerate(sample['bodies']):
            if name in self.parts:
                v,f,c = self.parts[name]
                self.draw(image,depth,v@rotations[i].T+pose[i,:3],f,c)
        return Image.fromarray(image)


def render(before, after, result, out):
    old, new = data(before), data(after)
    report = read(result)
    renderer = Renderer()
    output = out/'steep_seat_clearance.mp4'
    pipe = subprocess.Popen(['ffmpeg','-nostdin','-v','error','-f','rawvideo','-pix_fmt','rgb24',
        '-s',f'{SIZE[0]}x{SIZE[1]}','-r',str(FPS),'-i','-','-an','-c:v','libx264','-threads','2',
        '-preset','fast','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(output)],stdin=subprocess.PIPE)
    frames = 0
    try:
        for case in range(2):
            for i in range(20*FPS):
                tick = min(999,int((i+1)*50/FPS)-1)
                seconds = (tick+1)*.02
                im = Image.new('RGB',SIZE,BACKGROUND)
                d = ImageDraw.Draw(im)
                label(d,(40,22),f"20° low seat   /   {new['rows'][case]['id']}",39,bold=True)
                label(d,(42,76),f'{seconds:05.2f} / 20 s   ·   Standing start   ·   Same action limits and scoring',22,MUTED)
                for col,(sample,title) in enumerate(((old,'Previous approach'),(new,'Clearance-first approach'))):
                    x = 40+690*col
                    label(d,(x,123),title,28,bold=True)
                    im.paste(renderer.panel(sample,case,tick),(x,161))
                    failed = bool(sample['flags'][tick,case].any())
                    held = sample['held'][tick,case]
                    label(d,(x,698),f'Upward seat load   {100*sample["load"][tick,case]:4.1f}% of body weight',20)
                    if failed:
                        status = 'Forbidden contact recorded' if sample['flags'][tick,case,1] else 'Physical failure recorded'
                        label(d,(x,739),status,23,RED,True)
                    else:
                        label(d,(x,739),f'Continuous qualifying hold   {held:.2f} s',23,GREEN,True)
                label(d,(42,802),'Saved 4090 simulation replay  ·  Orange: hip housings  ·  Green: pelvis',18,MUTED)
                pipe.stdin.write(im.tobytes()); frames += 1
        im = Image.new('RGB',SIZE,BACKGROUND); d = ImageDraw.Draw(im)
        label(d,(58,65),'Standing-start sitting benchmark' if report['total']==72 else 'Targeted clearance check',43,bold=True)
        label(d,(55,179),f'{report["clean"]} / {report["total"]} clean',110,GREEN if report['clean']==72 else INK,True)
        label(d,(62,343),f'{len(report["repairs"])} new repairs   ·   {len(report["regressions"])} regressions',35)
        for i,r in enumerate(new['rows']):
            label(d,(64,455+75*i),f'{r["id"]}     {"PASS" if r["clean"] else "FAIL"}     {r["final_hold_s"]:.2f} s qualifying hold',29,
                  GREEN if r['clean'] else RED,True)
        label(d,(64,741),'All trials ran for 20 seconds in simulation.',24,MUTED)
        for _ in range(4*FPS):
            pipe.stdin.write(im.tobytes()); frames += 1
        pipe.stdin.close()
        assert pipe.wait(timeout=20)==0
    finally:
        if pipe.poll() is None:
            pipe.kill(); pipe.wait()
    probe = json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-count_frames',
        '-show_entries','stream=nb_read_frames,duration,width,height','-of','json',str(output)]))['streams'][0]
    assert int(probe['nb_read_frames'])==frames and float(probe['duration'])==44.
    (out/'video_manifest.json').write_text(json.dumps(dict(video_sha256=digest(output),stream=probe,
        trajectories={old['source']:old['sha256'],new['source']:new['sha256']},visuals_sha256=digest(VISUALS),
        replay=True,full_trial_real_time=True),indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('before','after','result','out'):
        parser.add_argument(name,type=Path)
    a = parser.parse_args()
    a.out.mkdir(exist_ok=False)
    render(a.before,a.after,a.result,a.out)
