"""Create a success video only from three independently audited full attempts."""
import sys,json,subprocess,hashlib
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from scipy.spatial.transform import Rotation
ROOT=Path('/home/kadir/humanoid-sitting')
sys.path.insert(0,str(ROOT/'experiments/furniture_saved_review_20261104_v1'))
import video as raster
from recover import sha,read
from video import mesh,group_shapes,label
raster.PW,raster.GH=608,375

class View(raster.Panel):
    def __init__(self,folder,case):
        self.audit=read(folder/'independent_audit.json')
        assert self.audit['qualifying_lean'] and self.audit['mapping_validated'] and self.audit['row']['clean']
        assert self.audit['source_hashes']['trajectory.npz']==sha(folder/'trajectory.npz')
        self.dy=read(folder/'manifest.json')['lateral_start_m']
        with np.load(folder/'trajectory.npz') as z:
            self.pose=z['body'][4::5,0];self.force=z['force'][4::5,0,3:5]
            self.hold=z['held_ticks'][4::5,0]*.02
            fn=(z['force']*z['physical_normal_w']).sum(-1)
            self.load=(fn[:,:,3:5].clip(min=0)*z['physical_inside_patch'][:,:,3:5]).sum(-1)[4::5,0]/z['weight'][4::5,0]
        names=read(folder/'results.json')['body_names']
        shapes=group_shapes(read(ROOT/'experiments/furniture_perception_repair_20261029_v1/render_colliders.json')['shapes'])
        self.indices=[(names.index(k),v) for k,v in shapes.items()]
        self.contacts=[names.index(k) for k in ['torso_link','logo_link']]
        with np.load(case['mesh_npz']) as z:v,f,col=mesh(z['vertices'],z['faces'],[170,178,178])
        col[read(Path(case['mesh_npz']).with_name('target_region.json'))['triangle_indices']]=[246,179,71]
        bounds=np.concatenate([v,self.pose[:,:,:3].reshape(-1,3)])@raster.VIEW
        lo,hi=bounds.min(0),bounds.max(0)
        self.scale=min(590/(hi[0]-lo[0]+.10),352/(hi[1]-lo[1]+.13))
        self.offset=np.array([304,188,0])-(lo+hi)*.5*self.scale
        self.base=np.full((375,608,3),[247,249,250],np.uint8)
        self.z=np.full((375,608),-np.inf,np.float32)
        raster.paint(self.base,self.z,608,375,self.project(v),f,col,len(f))
        self.caps={}
        for fi in folder.glob('frame_*.npz'):
            tick=int(fi.stem.split('_')[-1])
            with np.load(fi) as z:
                im=z['rgb'][...,:3].astype(np.uint8).copy();pix=tuple(z['selected_pixels'].T)
                im[pix]=im[pix]*.6+np.array([33,215,118])*.4
                self.caps[tick]=Image.fromarray(im).resize((224,168),Image.Resampling.NEAREST)
    def draw(self,i):
        panel=Image.new('RGB',(608,700),'white');d=ImageDraw.Draw(panel)
        title="Centered start" if self.dy==0 else f"Standing start  {self.dy*1000:+.0f} mm lateral"
        label(d,(12,8),title,30,bold=True)
        pos=self.pose[i];geo=self.geometry([(pos[j],m) for j,m in self.indices])
        g=ImageDraw.Draw(geo)
        for b,force in zip(self.contacts,self.force[i]):self.arrow(g,pos[b,:3],force)
        panel.paste(geo,(0,60))
        label(d,(12,451),f"Qualifying hold   {self.hold[i]:.1f} s",29,'#136c54' if self.hold[i]>=5 else '#263e4a',True)
        tick=max(k for k in self.caps if k<=(i+1)*5)
        panel.paste(self.caps[tick],(12,514))
        label(d,(251,522),'Rear camera',23,bold=True)
        label(d,(251,558),f"Captured at {tick*.02:.1f} s",22)
        label(d,(251,606),'Torso load',21)
        label(d,(251,634),f"{self.load[i]:.1%} of body weight",24,bold=True)
        return panel

def main():
    folder=Path(sys.argv[1]); dest=folder/'video';dest.mkdir(exist_ok=False)
    case=next(c for c in read(ROOT/'experiments/furniture_test_20261028_v1/cases.json') if c['id']=='myd1WSucAz_rear_back')
    views=[View(folder/f'trial_{j}',case) for j in range(3)]
    file=dest/'repeatable_armchair_lean.mp4'
    proc=subprocess.Popen(['ffmpeg','-loglevel','error','-nostdin','-f','rawvideo','-pix_fmt','rgb24','-s','1920x1080','-r','10','-i','-','-an','-c:v','libx264','-threads','3','-preset','veryfast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(file)],stdin=subprocess.PIPE)
    for i in range(200):
        im=Image.new('RGB',(1920,1080),(233,240,243));d=ImageDraw.Draw(im)
        label(d,(30,22),'Leaning against the back of the box armchair',48,bold=True)
        label(d,(32,91),'Three independent standing starts  ·  recorded simulation at normal speed',30)
        for j,view in enumerate(views):im.paste(view.draw(i),(32+624*j,175))
        label(d,(36,920),'Gold: target back face     Blue: robot collision shapes     Red: measured torso contact force',26)
        label(d,(36,968),'Green pixels show the detector’s candidate. The controller rejects near-range jumps. Camera insets show capture times.',23)
        label(d,(1620,96),f'{(i+1)/10:.1f} / 20 s',28,bold=True)
        proc.stdin.write(im.tobytes())
    end=Image.new('RGB',(1920,1080),(233,240,243));d=ImageDraw.Draw(end)
    label(d,(88,110),'Three independent five-second holds',55,bold=True)
    label(d,(92,226),'Final uninterrupted qualifying hold in each 20-second trial',32)
    for j,v in enumerate(views):
        x=92+j*600
        label(d,(x,417),'Centered' if v.dy==0 else f'{v.dy*1000:+.0f} mm lateral',36,bold=True)
        label(d,(x,492),f'{v.hold[-1]:.2f} seconds',49,'#11694d',True)
        label(d,(x,587),'Zero slip or illegal-contact ticks',24)
    label(d,(92,858),'Verified on this box-armchair back in simulation. Same controller and scoring rules for all three starts.',29)
    for _ in range(30):proc.stdin.write(end.tobytes())
    proc.stdin.close();assert proc.wait()==0
    info=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-count_frames','-show_entries','stream=width,height,nb_read_frames,duration','-of','json',str(file)]))
    assert int(info['streams'][0]['nb_read_frames'])==230 and float(info['streams'][0]['duration'])==23.
    (dest/'manifest.json').write_text(json.dumps(dict(sha256=sha(file),streams=info['streams'],audits=[sha(folder/f'trial_{j}/independent_audit.json') for j in range(3)],scope='Camera-guided controller, all 20s of three simulations, never a physical hardware test.'),indent=2))
if __name__=='__main__':main()
