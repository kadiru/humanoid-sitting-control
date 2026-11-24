"""Render a short, matched replay from measured robot poses and forces."""
import argparse
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image, ImageDraw

from run import P, read
from stance23_visible import visible_clip
from video_confirmed import vis

STUDY = P/'sitting_skills_01'


def replay(arm, row):
    folder = STUDY/arm
    result = read(folder/'measures.json')[row]
    geometry = read(folder/'bank'/'geometry.json')[row]
    data = dict(evaluation_folder=str(folder), asset=geometry['name'], trajectory_index=row)
    with np.load(folder/'evaluation'/'trajectory.npz') as trace:
        power = np.abs(trace['physical_torque'][:,row,:23]*trace['physical_joint_velocity'][:,row,:23]).sum(1)
        load = trace['support_bw'][:,row]*100
        live = np.flatnonzero(trace['learn'][:,row])[-1]
    return visible_clip(data), result, load, power, live


def render(staged, direct, output):
    vis.PANEL = (606,350)
    width,height = 1280,550
    guarded = read(STUDY/staged/'skills.json').get('guard_from')
    names = ('Guarded transfer + hold' if guarded else 'Four staged skills','One full-horizon actor')
    arms = (staged,direct)
    outcomes = [read(STUDY/n/'measures.json') for n in arms]
    passes = [sum(bool(x['supported'] and x['descent_feet_planted']) for x in o) for o in outcomes]
    proc = subprocess.Popen(
        ['ffmpeg','-nostdin','-v','error','-n','-f','rawvideo','-pix_fmt','rgb24',
         '-s',f'{width}x{height}','-r','10','-i','-','-an','-c:v','libx264',
         '-crf','18','-pix_fmt','yuv420p','-threads','4','-movflags','+faststart',str(output)],
        stdin=subprocess.PIPE)
    with proc.stdin as stream:
        for row in (1,10):
            clips = [replay(arm,row) for arm in arms]
            case = clips[0][1]['case'].replace('sit/','').replace('_',' ').replace('/',' · ')
            for t in range(4,1000,8):
                frame = Image.new('RGB',(width,height),'#F4F8F6')
                draw = ImageDraw.Draw(frame)
                vis.label(draw,(24,11),'Learning to sit in stages',28,bold=True)
                vis.label(draw,(25,53),f'{case}   |   Recorded simulation   |   {(t+1)*.02:.1f} s of 20 s',17)
                for i,(clip,result,load,power,live) in enumerate(clips):
                    x = 24+i*628
                    vis.label(draw,(x,88),f'{names[i]}   ·   {passes[i]}/12 supported',21,bold=True)
                    img,status = clip(t)
                    frame.paste(img,(x,121))
                    k = min(t,live)
                    y = power[max(0,k-24):k+1].mean()
                    vis.label(draw,(x,483),f'Seat load {load[k]:.0f}%  ·  Mechanical power {y:.1f} W',16,bold=True)
                    vis.label(draw,(x,511),status,15)
                stream.write(frame.tobytes())
        card = Image.new('RGB',(width,height),'#12302B')
        draw = ImageDraw.Draw(card)
        from PIL import ImageFont
        path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
        font = ImageFont.truetype(path,25)
        big = ImageFont.truetype(path,38)
        draw.text((52,75),'A supported sit is the first test.',font=big,fill='#ffffff')
        draw.text((52,164),'Compare seat load, torque and power over the full hold.',font=font,fill='#E0EFEA')
        draw.text((52,240),f'{names[0]}  {passes[0]}/12     {names[1]}  {passes[1]}/12',
                  font=font,fill='#E0EFEA')
        draw.text((52,332),'Four furniture assets · 12 matched simulation starts',font=font,fill='#E0EFEA')
        for _ in range(38):
            stream.write(card.tobytes())
    assert proc.wait() == 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('staged')
    parser.add_argument('direct')
    parser.add_argument('output',type=Path)
    args = parser.parse_args()
    render(args.staged,args.direct,args.output)
