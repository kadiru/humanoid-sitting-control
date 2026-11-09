"""Four-view video: preserved failure and new geometry-conditioned transitions."""
import json,subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[1];out=root/'runs/geometry_reference_progress_01';out.mkdir(exist_ok=True)
runs=['furniture_stage2_failure_video_01','geometry_reference_v12_low_chair_parent_video_01','geometry_reference_v12_shallow_box_scripted_video_01','geometry_reference_v12_heldout_couch_scripted_video_01']
labels=['Preserved PPO, low chair','Same PPO + geometry reference, low chair','Geometry scripted, shallow box','Geometry scripted, held-out rigid couch']
cmd=['ffmpeg','-hide_banner','-loglevel','error','-n']
filters=[]
for i,(name,label) in enumerate(zip(runs,labels)):
 run=root/'runs'/name;movie=next(run.glob('*.mp4'));r=json.loads((run/'results.json').read_text());cmd+=['-i',str(movie)]
 text=f"{label}: {'PASS' if r['success_rate']==1 else 'FAIL'}, peak {r['maximum_seat_impact_N']:.0f} N"
 filters.append(f"[{i}:v]scale=640:360,setsar=1,drawtext=text='{text}':fontcolor=white:fontsize=15:box=1:boxcolor=black@0.65:x=8:y=8[v{i}]")
filters.append('[v0][v1][v2][v3]xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0[v]')
cmd+=['-filter_complex',';'.join(filters),'-map','[v]','-c:v','libx264','-crf','23','-preset','fast','-pix_fmt','yuv420p',str(out/'transitions.mp4')]
(out/'video_command.json').write_text(json.dumps(cmd,indent=2)+'\n')
subprocess.run(cmd,check=True)
print(out/'transitions.mp4')
