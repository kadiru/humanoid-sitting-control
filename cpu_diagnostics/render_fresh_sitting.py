"""Render all 24 saved simulation trajectories with their measured outcomes."""

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image, ImageDraw

import render_steep_seat_clearance as replay


SIZE = (1600, 820)
PANEL = (366, 274)
FPS = 15
ROLL_FRAMES = 150
END_FRAMES = 45


def render(run, destination, original_bank=None):
    evaluation = run/('evaluation' if original_bank is None else 'fresh')
    trace = evaluation/'trajectory.npz'
    audit = replay.read(evaluation/'independent_audit.json')
    assert audit['passed'] and replay.digest(trace) == audit['files']['trajectory.npz']
    results = replay.read(evaluation/'results.json')
    comparison = replay.read(run/'comparison.json')
    geometry = replay.read((original_bank or run/'bank')/'geometry.json')
    is_repair = original_bank is not None
    repaired = set(comparison['splits']['fresh']['repairs']) if is_repair else set()
    if is_repair:
        scores = comparison['splits']
        rows = [dict(r,height_cm=round(g['height']*100,3),slope_deg=g['pitch_deg'])
                for r,g in zip(scores['fresh']['cases'],geometry)]
        all_passed = sum(s['clean'] for s in scores.values())
        all_cases = sum(s['total'] for s in scores.values())
        overall = f'{all_passed}/{all_cases} overall  ·  {len(repaired)} repairs'
    else:
        rows = comparison['rows']
        overall = f'{comparison["score"]["clean"]}/24 passed'
    positioned = all('fore_aft_offset_cm' in row and 'lateral_offset_cm' in row for row in rows)
    assert [r['id'] for r in rows] == [r['id'] for r in results['episodes']]
    with np.load(trace,allow_pickle=False) as z:
        poses = z['body'].copy()
        poses[:,:,:,:3] -= np.asarray(results['origins'])[None,:,None,:]
        hold = z['held_ticks']*.02
        slipped = np.maximum.accumulate(~z['no_slip'],axis=0)
        illegal = np.maximum.accumulate(~z['legal'],axis=0)
        failed = np.maximum.accumulate(z['failure'],axis=0)
    assert poses.shape[:2] == (1000,24)
    assert len(rows) == len(geometry) == 24
    sample = dict(poses=poses,geometry=geometry,bodies=results['body_names'])
    panel = (PANEL[0],250) if positioned else PANEL
    replay.PANEL = panel
    renderer = replay.Renderer()
    renderer.scale = 218
    renderer.offset = np.array([panel[0]/2,panel[1]/2,0])-np.array([-.20,0,.48])@replay.VIEW*renderer.scale

    with subprocess.Popen(['ffmpeg','-nostdin','-v','error','-n','-f','rawvideo','-pix_fmt','rgb24',
                           '-s',f'{SIZE[0]}x{SIZE[1]}','-r',str(FPS),'-i','-','-an','-c:v','libx264','-threads','2',
                           '-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(destination)],
                          stdin=subprocess.PIPE) as pipe:
        for group in range(3):
            indices = range(group*8,group*8+8)
            for frame in range(ROLL_FRAMES+END_FRAMES):
                complete = frame >= ROLL_FRAMES
                tick = 999 if complete else round(frame*999/(ROLL_FRAMES-1))
                canvas = Image.new('RGB',SIZE,replay.BACKGROUND)
                d = ImageDraw.Draw(canvas)
                title = 'Sitting touchdown repair' if is_repair else 'Fresh sitting trials'
                if positioned:
                    title = 'Sitting on shifted seats'
                replay.label(d,(35,20),title,36,bold=True)
                replay.label(d,(1000 if is_repair else 1110,26),f'{overall}  ·  Batch {group+1}/3',24,bold=True)
                time_label = 'Final scores' if complete else f'{(tick+1)*.02:05.2f} / 20 s   ·   2× replay'
                replay.label(d,(37,67),time_label,22,replay.MUTED)
                for slot,case in enumerate(indices):
                    x, y = 35+387*(slot%4), 109+335*(slot//4)
                    row = rows[case]
                    title = f'{case+1:02d}   {row["height_cm"]:g} cm  /  +{row["slope_deg"]:g}°'
                    replay.label(d,(x,y),title,22,bold=True)
                    if positioned:
                        offsets = f'X {row["fore_aft_offset_cm"]:+g} cm  /  Y {row["lateral_offset_cm"]:+g} cm'
                        replay.label(d,(x,y+26),offsets,17,replay.MUTED)
                    canvas.paste(renderer.panel(sample,case,tick),(x,y+(50 if positioned else 30)))
                    if complete:
                        state = 'REPAIRED' if row['id'] in repaired else 'PASS' if row['passed'] else 'FAIL'
                        text = f'{state}  ·  {row["hold_s"]:.2f} s final hold'
                        color = replay.GREEN if row['passed'] else replay.RED
                    else:
                        event = ('Physical failure' if failed[tick,case] else 'Illegal contact' if illegal[tick,case]
                                 else 'Slip recorded' if slipped[tick,case] else '')
                        text = f'{event}  ·  {hold[tick,case]:.2f} s hold' if event else f'{hold[tick,case]:.2f} s qualifying hold'
                        color = replay.RED if event else replay.INK
                    replay.label(d,(x,y+307),text,20,color,True)
                replay.label(d,(35,793),'Saved RTX 4090 simulation replay  ·  Standing start  ·  Final hold must be ≥5 seconds',17,replay.MUTED)
                pipe.stdin.write(canvas.tobytes())
        pipe.stdin.close()
        assert pipe.wait(timeout=30) == 0
    probe = json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0',
        '-show_entries','stream=nb_read_frames,duration,width,height','-of','json',str(destination)]))['streams'][0]
    assert int(probe['nb_read_frames']) == 3*(ROLL_FRAMES+END_FRAMES)
    replay_manifest = dict(stream=probe,sha256=replay.digest(destination),trajectory_sha256=replay.digest(trace),
                           all_24_cases=True,simulation_replay=True)
    destination.with_suffix('.json').write_text(json.dumps(replay_manifest,indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--original-bank',type=Path,help='Original 24-case bank used for the repair evaluation.')
    args = parser.parse_args()
    render(args.run,args.output,args.original_bank)
