"""Four synchronized representative rigid-furniture transitions, matched camera."""
from pathlib import Path
import json,subprocess
r=Path(__file__).resolve().parents[1];out=r/'runs/furniture_progress_01';out.mkdir(exist_ok=True)
cmd=['ffmpeg','-nostdin','-n']
for i in range(4):cmd += ['-i',str(r/f'runs/furniture_family_{i}_video_04/transition_0000.mp4')]
labels=['Chair','Bench','Box','Rigid couch - no cushion compliance']
filters=[]
for i,label in enumerate(labels):filters.append(f"[{i}:v]drawtext=text='{label}':x=20:y=20:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.7[v{i}]")
filters.append('[v0][v1][v2][v3]xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0[v]')
cmd += ['-filter_complex',';'.join(filters),'-map','[v]','-an','-c:v','libx264','-crf','20','-pix_fmt','yuv420p',str(out/'families.mp4')]
subprocess.run(cmd,check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
(out/'video_command.json').write_text(json.dumps(cmd,indent=2)+'\n')
print(out/'families.mp4')
