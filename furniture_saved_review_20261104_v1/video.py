"""Draw the saved depth, robot poses, collision meshes and measured forces on CPU."""
import ctypes
from functools import lru_cache
import json
import os
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation
from recover import ASSETS, OUT, SAVED, SOURCE, publish, read, sha

HERE = Path(__file__).resolve().parent
W, H, FPS = 2560, 1440, 10
PW, PH, GH = 620, 298, 210
BG=(239,243,245)
INK="#152e3e"
MUTED="#405b68"
GREEN="#116c4c"
RED="#af3b35"
AMBER="#87570d"
ar = np.array
right = ar([.85, np.sqrt(1-.85**2), 0])
eye = ar([right[1]*.96, -right[0]*.96, .28])
up = np.cross(eye, right)
VIEW = np.stack([right, -up, eye], axis=1)
paint = ctypes.CDLL(str(HERE/"raster.so")).paint
arrp = np.ctypeslib.ndpointer
paint.argtypes = [arrp(np.uint8, flags="C"), arrp(np.float32, flags="C"), ctypes.c_int, ctypes.c_int,
                  arrp(np.float32, flags="C"), arrp(np.int32, flags="C"), arrp(np.uint8, flags="C"), ctypes.c_int]
paint.restype = None


@lru_cache(None)
def font(size, bold=False):
    suffix = "-Bold" if bold else ""
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/DejaVuSans{suffix}.ttf", size)


def label(draw, xy, string, size=21, color=INK, bold=False):
    draw.text(xy, string, font=font(size, bold), fill=color)


def mesh(vertices, faces, color):
    faces = np.ascontiguousarray(faces, dtype=np.int32)
    return np.asarray(vertices), faces, np.broadcast_to(color, (len(faces),3)).astype(np.uint8).copy()


def group_shapes(shapes):
    bybody = {}
    for shape in shapes:
        bybody.setdefault(shape["body"], []).append(shape)
    out = {}
    for name, items in bybody.items():
        verts, faces = [], []
        offset = 0
        for s in items:
            verts.extend(s["vertices"])
            faces.append(np.asarray(s["faces"])+offset)
            offset += len(s["vertices"])
        out[name] = mesh(verts, np.concatenate(faces), [96,171,205])
    return out


class Panel:
    def __init__(self, case, result, shapes):
        self.result = result
        self.title = case["title"].replace("Rectangular ", "").replace("Round wooden stool","Wooden stool")
        self.title = self.title[0].upper()+self.title[1:]
        self.face = case["approach"].replace("rear_back","rear back").replace("_oblique"," at an angle")
        self.name = self.title+"  /  "+self.face
        folder = SAVED / ("robot_"+case["id"])
        self.frames = []
        for j in range(3):
            with np.load(folder / f"frame_{j}.npz") as z:
                rgb = z["rgb"][...,:3].astype(np.uint8).copy()
                pixels = tuple(z["selected_pixels"].T)
                rgb[pixels] = .40*rgb[pixels] + .60*ar([23,227,128])
                depth = z["depth"]
                shade = 1 - np.clip((depth-.05)/1.5,0,1)
                shade = np.where(np.isfinite(shade), shade, 0)
                gray = np.repeat((255*shade).astype(np.uint8)[...,None],3,-1)
                self.frames.append([Image.fromarray(x).resize((260,195),Image.Resampling.NEAREST) for x in [rgb,gray]])
        self.depth = read(folder/"depth_audit.json")["frames"]
        self.robot = result["robot"]
        self.shapes = shapes
        names = read(folder/"results.json")["body_names"]
        contacts = read(folder/"manifest.json")["contact_bodies"][:5]
        self.contact_bodies = [names.index(b) for b in contacts]
        self.indices = [(names.index(b), m) for b,m in shapes.items()]
        with np.load(folder/"trajectory.npz") as z:
            self.poses = z["body"][4::5,0]
            self.forces = z["physical_force_w"][4::5,0,:5]
        with np.load(case["mesh_npz"]) as z:
            v,f,col = mesh(z["vertices"],z["faces"],[172,178,174])
        target=read(Path(case["mesh_npz"]).with_name("target_region.json"))["triangle_indices"]
        col[target] = [245,183,77]
        bounds = np.concatenate([v, self.poses[:,:,:3].reshape(-1,3)]) @ VIEW
        lo, hi = bounds.min(0), bounds.max(0)
        self.scale = min((PW-30)/(hi[0]-lo[0]+.18), (GH-10)/(hi[1]-lo[1]+.15))
        self.offset = ar([PW/2,GH/2,0])-(lo+hi)*.5*self.scale
        self.base = np.full((GH,PW,3), [248,250,249],np.uint8)
        self.z = np.full((GH,PW), -np.inf, np.float32)
        paint(self.base,self.z,PW,GH,self.project(v),f,col,len(f))

    def project(self,v):
        return np.ascontiguousarray((v@VIEW)*self.scale+self.offset, dtype=np.float32)

    def geometry(self, items):
        rgb, depth = self.base.copy(), self.z.copy()
        for pose, (verts,faces,col) in items:
            rot = Rotation.from_quat(pose[3:]).as_matrix()
            uv = np.ascontiguousarray((verts@(rot.T@VIEW)+pose[:3]@VIEW)*self.scale+self.offset,dtype=np.float32)
            paint(rgb,depth,PW,GH,uv,faces,col,len(faces))
        return Image.fromarray(rgb)

    def arrow(self,draw,point,force):
        if np.linalg.norm(force)<1:
            return
        start,end = self.project(np.asarray([point,point+.002*force]))[:,:2]
        delta=end-start
        length=np.linalg.norm(delta)
        if length<.7:
            return
        draw.line([tuple(start), tuple(end)],fill="#cc302c",width=4)
        unit=delta/length
        across=np.array([-unit[1],unit[0]])
        tip=min(length,9)
        draw.polygon([tuple(end),tuple(end-tip*unit+.4*tip*across),tuple(end-tip*unit-.4*tip*across)],fill="#cc302c")

    def panel(self, phase, index, fixture=None):
        im=Image.new("RGB",(PW,PH),"white")
        d=ImageDraw.Draw(im)
        label(d,(12,3),self.name,24,bold=True)
        if phase=="depth":
            rgb,gray=self.frames[index]
            im.paste(rgb,(32,43)); im.paste(gray,(325,43))
            fraction=self.depth[index]["selected_target_fraction"]
            stat="No selection" if fraction is None else f"{fraction:.0%} of selected pixels on target"
            label(d,(12,254),stat,19)
        elif phase=="robot":
            poses=self.poses[index]
            geo=self.geometry((poses[j],m) for j,m in self.indices)
            g=ImageDraw.Draw(geo)
            for force,b in zip(self.forces[index],self.contact_bodies):
                self.arrow(g,poses[b,:3],force)
            im.paste(geo,(0,38))
            t=(index+1)/10
            reason="Contact score ambiguous" if self.robot["clean"] is None else "Failed"
            if self.robot["warm_failure"] or (self.robot["first_recorded_failure_s"] is not None and t>=self.robot["first_recorded_failure_s"]):
                label(d,(12,250),reason,20,AMBER if self.robot["clean"] is None else RED,True)
            loads=np.linalg.norm(self.forces[index],axis=-1)
            label(d,(12,277),f"Peak current channel   {loads.max():.0f} N",15,MUTED)
        else:
            row, origin, shape, check = fixture
            pose=np.asarray(row["pose"]).copy()
            pose[:3]-=origin
            geo=self.geometry([(pose,shape)])
            anchor=np.mean(row["points"],axis=0) if row["points"] else pose[:3]
            self.arrow(ImageDraw.Draw(geo),anchor,np.asarray(row["force"]))
            im.paste(geo,(0,38))
            if check["physical_support"] and check["intended_contact_patch"]:
                status,color="Load carried on target",GREEN
            elif check["physical_support"]:
                status,color="Load carried; patch check failed",AMBER
            else:
                status,color="Load check failed",RED
            label(d,(12,250),status,20,color,True)
            label(d,(12,277),f"Whole-press result   /   current force {np.linalg.norm(row['force']):.0f} N",15,MUTED)
        return im


def main():
    cases = read(ASSETS/"cases.json")
    summary = read(OUT/"audit_summary.json")
    results = {r["id"]:r for r in summary["trials"]}
    shapes=group_shapes(read(SOURCE/"render_colliders.json")["shapes"])
    fixtures=read(SAVED/"body_support/measurements.json")
    fixture_manifest=read(SAVED/"body_support/manifest.json")
    body_shapes=group_shapes([s for group in fixture_manifest["bodies"].values() for s in group])
    assert fixture_manifest["load_bodyweight_fraction"]==.5
    assert [c["id"] for c in fixture_manifest["cases"]]==[c["id"] for c in cases]
    panels=[Panel(c,results[c["id"]],shapes) for c in cases]
    media=OUT/"media"; media.mkdir(exist_ok=True)
    path=media/"updated_furniture_depth_contacts.mp4"
    if path.exists():
        raise FileExistsError(path)
    tmp=media/f"render-{os.getpid()}.mp4"
    cmd=["ffmpeg","-nostdin","-v","error","-f","rawvideo","-pix_fmt","rgb24","-s",f"{W}x{H}","-r",str(FPS),"-i","-",
         "-an","-c:v","libx264","-threads","3","-preset","veryfast","-crf","18","-pix_fmt","yuv420p","-movflags","+faststart",str(tmp)]
    proc=subprocess.Popen(cmd,stdin=subprocess.PIPE)
    frames=0

    def emit(im,n=1):
        nonlocal frames
        data=im.tobytes()
        for _ in range(n):
            proc.stdin.write(data)
            frames+=1

    def grid(phase,index,group=0):
        im=Image.new("RGB",(W,H),BG); d=ImageDraw.Draw(im)
        title={"depth":"What the rear camera saw","robot":"What the whole robot did","fixture":"Which surfaces carried a controlled load"}[phase]
        label(d,(24,16),title,48,bold=True)
        sub={"depth":"Green shows the selected patch in RGB. In the depth image, nearer returns are brighter.",
             "robot":"Full 20-second attempts at 1×   ·   Blue robot colliders   ·   Gold target faces   ·   Red measured furniture forces",
             "fixture":"Body-shaped fixtures at 1×   ·   Target load 158 N (half the robot’s weight)   ·   Gold target faces   ·   Red measured forces"}[phase]
        label(d,(26,81),sub,25,MUTED)
        stamp=(f"Camera at {index*.2:.1f} s" if phase=="depth" else
               f"Replay   {(index+1)/10:4.1f} s" if phase=="robot" else f"Press {group+1} of 3   /   {(index+1)/10:.1f} s")
        label(d,(2070,30),stamp,33,bold=True)
        for i,p in enumerate(panels):
            fixture=None
            if phase=="fixture":
                row=fixtures[group][index*5+4][i]
                assert row["case"]==cases[i]["id"] and row["round"]==group
                fixture=(row,ar([0,i*8,0]),body_shapes[row["body"]],p.result["fixture"][group])
            im.paste(p.panel(phase,index,fixture),(24+(PW+10)*(i%4),142+(PH+12)*(i//4)))
        x=24+2*(PW+10);y=142+3*(PH+12)
        lines={"depth":["Three recorded frames per layout","Left: RGB with selected surface.   Right: optical depth.",
                        "The target is the specified mesh region for that approach."],
               "robot":["Reading the robot trials","Five seconds of stable, legal support required at the end.",
                        "Force arrows start at the measured body’s origin (2 mm/N).",
                        "Floor forces omitted. Ambiguous contacts receive no clean score."],
               "fixture":["Reading the presses","Sitting layouts: pelvis, left thigh, right thigh.   Leaning: torso three times.",
                          "Load and target contact must both pass during the steady window.",
                          "Fixture pose is controlled. The whole-robot test is scored separately."]}[phase]
        for j,line in enumerate(lines):label(d,(x+14,y+28+j*48),line,27 if j==0 else 23,INK if j==0 else MUTED,j==0)
        return im

    try:
        for j in range(3): emit(grid("depth",j),15)
        for k in range(200): emit(grid("robot",k))
        for j in range(3):
            for k in range(60): emit(grid("fixture",k,j))
        im=Image.new("RGB",(W,H),BG);d=ImageDraw.Draw(im)
        label(d,(120,110),"What supported the robot?",68,bold=True)
        label(d,(122,240),"No whole-robot attempt reached the five-second hold.",48,bold=True)
        label(d,(122,330),"14 standing starts   ·   5 scored failures   ·   9 ambiguous contact scores   ·   0 validated successes",32,MUTED)
        label(d,(122,525),"Controlled presses that carried the load on the intended surface",42,bold=True)
        lines=[("Box armchair","Seat 1/3 in each direction   ·   Arm 1/3   ·   Rear back 3/3"),
               ("Box-arm couch","Seat 0/3 in each direction   ·   Arm 0/3   ·   Rear back 0/3"),
               ("Wooden table","Top 3/3 in each direction, using the torso fixture"),
               ("Stool and coffee table","Top 0/3 in each direction")]
        for idx,(name,stat) in enumerate(lines):
            y=655+idx*106
            label(d,(122,y),name,38,bold=True);label(d,(730,y),stat,37)
        label(d,(122,1200),"23 of 42 presses carried the load. 12 of those also passed the contact-location check.",33,MUTED)
        emit(im,65)
    finally:
        proc.stdin.close()
        code=proc.wait()
    if code: raise RuntimeError(f"ffmpeg returned {code}")
    probe=json.loads(subprocess.check_output(["ffprobe","-v","error","-show_entries","format=duration:stream=width,height,nb_frames","-of","json",str(tmp)]))
    assert int(probe["streams"][0]["nb_frames"])==frames==490 and float(probe["format"]["duration"])==49
    os.link(tmp,path);tmp.unlink()
    publish(media/"video_manifest.json",dict(sha256=sha(path),frames=frames,duration_s=49,size=[W,H],fps=FPS,
            robot_seconds=20,fixture_seconds=18,physics_steps=0,ppo_updates=0,
            force_arrow_scale_m_per_N=.002,robot_anchor="body origin, furniture channels only",fixture_anchor="mean contact point if available, else fixture origin",
            audit_summary_sha256=sha(OUT/"audit_summary.json"),collider_sha256=sha(SOURCE/"render_colliders.json"),
            source_trial_hashes={c["id"]:sha(SAVED/("robot_"+c["id"])/"independent_audit.json") for c in cases}))
    print(path,flush=True)


if __name__=="__main__":
    main()
