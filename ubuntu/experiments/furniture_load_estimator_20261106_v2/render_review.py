"""CPU-only video of existing recordings; no simulation or model inference."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation

ROOT = Path('/home/kadir/humanoid-sitting')
OUT = ROOT / 'outputs/furniture_load_review_20261106_v1'
sys.path.insert(0, str(ROOT / 'experiments/furniture_trials_review_20261105_v1'))
import render as prior

read = lambda p: json.loads(Path(p).read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
W, H, FPS = 1280, 720, 25
BG = (245, 247, 250)
INK, MUTED, GREEN, RED = '#183444', '#546d7c', '#14765b', '#b04237'


def text(d, xy, value, size=23, color=INK, bold=False):
    d.text(xy, value, font=prior.font(size, bold), fill=color)


class Scene(prior.Scene):
    def __init__(self, folder, row, reference=False):
        self.w, self.h, self.static = 900, 525, False
        self.folder = folder
        name = 'trajectory.npz' if reference else 'furniture_trajectory.npz'
        audit = read(folder / 'independent_audit.json')
        hashes = audit['source_hashes'] if reference else audit['files']
        assert sha(folder / name) == hashes[name]
        capture = read(folder / ('results.json' if reference else 'furniture_capture.json'))
        with np.load(folder / name) as z:
            data = dict(z)
        self.pose = data['body'][:, 0].copy()
        self.pose[:, :, :3] -= np.asarray(capture['origins'][0])
        self.parts = [(capture['body_names'].index(n), m)
                      for n, m in prior.visual.items() if n in capture['body_names']]
        physical = prior.original.physical(data)
        held = 0
        self.hold = []
        for good in physical['good'][:, 0]:
            held = held + 1 if good and not audit.get('warm_failure', False) else 0
            self.hold.append(held * .02)
        expected = audit['row']['final_hold_s'] if reference else row['final_hold_s']
        assert abs(self.hold[-1] - expected) < 1e-9
        self.slip = np.cumsum(~physical['no_slip'][:, 0])
        case = prior.cases[prior.variants[row['id']]['case']]
        with np.load(case['mesh_npz']) as z:
            self.v, self.f = z['vertices'], z['faces']
        self.c = np.tile((np.array(case['color']) * 255).astype(np.uint8), (len(self.f), 1))
        bounds = [self.v @ prior.VIEW]
        for j, (v, f, c) in self.parts:
            lo, hi = v.min(0), v.max(0)
            corners = np.array([[x, y, z] for x in [lo[0], hi[0]]
                                for y in [lo[1], hi[1]] for z in [lo[2], hi[2]]])
            rot = Rotation.from_quat(self.pose[:, j, 3:7]).as_matrix()
            pts = np.einsum('tij,kj->tki', rot, corners) + self.pose[:, j, None, :3]
            bounds.append(pts.reshape(-1, 3) @ prior.VIEW)
        points = np.concatenate(bounds)
        lo, hi = points.min(0), points.max(0)
        self.scale = min((self.w - 50) / (hi[0] - lo[0]), (self.h - 25) / (hi[1] - lo[1]))
        self.offset = np.array([self.w / 2, self.h / 2, 0]) - (lo + hi) * self.scale / 2
        self.base = np.full((self.h, self.w, 3), BG, np.uint8)
        self.depth = np.full((self.h, self.w), -np.inf, np.float32)
        ground = np.array([[-2.5, -2.5, 0], [2.5, -2.5, 0], [2.5, 2.5, 0], [-2.5, 2.5, 0]])
        prior.paint(self.base, self.depth, self.w, self.h, self.project(ground),
                    np.array([[0, 1, 2], [0, 2, 3]], np.int32),
                    np.tile([235, 240, 244], (2, 1)).astype(np.uint8), 2)
        prior.paint(self.base, self.depth, self.w, self.h, self.project(self.v),
                    np.ascontiguousarray(self.f, np.int32), self.c, len(self.f))


def page(title, subtitle):
    im = Image.new('RGB', (W, H), BG)
    d = ImageDraw.Draw(im)
    text(d, (30, 22), title, 31, bold=True)
    text(d, (30, 66), subtitle, 20, MUTED)
    d.line((30, 670, 1250, 670), fill='#d5dee5', width=2)
    text(d, (30, 686), 'Recorded simulation | original rules | no new PPO | CPU video replay', 18, MUTED)
    return im


def main():
    OUT.mkdir(exist_ok=False)
    rows = read(ROOT / 'outputs/furniture_trials_review_20261105_v1/results.json')['rows']
    row1 = next(r for r in rows if r['id'] == 'lean_myd1WSucAz_right')
    row2 = next(r for r in rows if r['id'] == 'sit_6MoOyPtetL_angled')
    folder1 = ROOT / 'outputs/furniture_repair_20261106_v1/motor_axis_readout'
    folder2 = Path(row2['source'])
    measurement = read(folder1 / 'motor_axis_audit.json')
    validation = read(Path(__file__).parent / 'validation.json')
    couch = next(c for c in validation['cases'] if c['label'] == row2['id'])
    assert not measurement['passed'] and not validation['passed']
    assert couch['false_load'] == 14
    movie = OUT / 'recorded_behavior_and_load_estimator.mp4'
    proc = subprocess.Popen(['ffmpeg', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                             '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-', '-an', '-c:v', 'libx264',
                             '-threads', '2', '-preset', 'fast', '-crf', '20', '-pix_fmt', 'yuv420p',
                             '-movflags', '+faststart', str(movie)], stdin=subprocess.PIPE)
    count = 0
    def emit(im, repeat=1):
        nonlocal count
        for _ in range(repeat):
            proc.stdin.write(im.tobytes())
            count += 1
    intro = page('Current capability and the measurement problem', 'November 6 review of recorded trials; no learned-policy success claimed')
    d = ImageDraw.Draw(intro)
    for i, line in enumerate(['1. Known armchair lean: full 20-second replay.',
                              '2. Angled couch sitting: an earlier failed trial.',
                              'The new load estimator is not approved for control.',
                              'The requested 20 successful demos are still incomplete.']):
        text(d, (55, 185 + 90 * i), line, 27)
    emit(intro, 3 * FPS)
    for section, folder, row, reference in [(1, folder1, row1, True), (2, folder2, row2, False)]:
        scene = Scene(folder, row, reference)
        for frame in range(20 * FPS):
            tick = min(999, frame * 2 + 1)
            title = 'Known armchair lean — PASS' if reference else 'Earlier angled couch sitting — FAIL'
            subtitle = ('November 6 reference recording' if reference else 'November 5 saved trial') + f' | {(tick + 1) * .02:04.2f} / 20.00 seconds | real time'
            im = page(title, subtitle)
            im.paste(scene.frame(tick), (15, 120))
            d = ImageDraw.Draw(im)
            color = GREEN if reference else RED
            text(d, (935, 140), 'Qualifying hold', 23, bold=True)
            text(d, (935, 180), f'{scene.hold[tick]:.2f} seconds', 29, color, True)
            lines = (['Final hold: 7.00 s', 'Loaded-slip ticks: 0', 'Illegal contact: 0', '',
                      'Behavior passed.', 'Force estimator failed:', 'normal error 6.64 N', 'shear error 25.31 N'] if reference else
                     [f"Final hold: {row['final_hold_s']:.2f} s", f"Loaded-slip ticks:", str(int(scene.slip[tick])), '',
                      'New estimator also failed:', '14 false load detections', 'across this saved trial.', 'Not used for control.'])
            for j, line in enumerate(lines):
                text(d, (935, 245 + 38 * j), line, 19, MUTED)
            emit(im)
            if frame in [250, 499]:
                im.save(OUT / f'section_{section}_{frame}.jpg', quality=94)
        print('Rendered section', section, flush=True)
    end = page('Training has not restarted', 'Estimator rejection is a measurement result, not a repaired behavior')
    d = ImageDraw.Draw(end)
    for j, line in enumerate(['Load estimator: 9 of 21 validation starts failed.',
                              'Five-object loaded validation coverage is incomplete.',
                              'New 10 sitting + 10 leaning successes: not collected.',
                              'Latest full 72-case result remains 64 clean; 8 unresolved.',
                              'Action limits and physical pass/fail rules are unchanged.',
                              'The 4070 remains reserved for AIWorker.']):
        text(d, (45, 145 + 75 * j), line, 25)
    end.save(OUT / 'summary.jpg', quality=94)
    emit(end, 7 * FPS)
    proc.stdin.close()
    assert proc.wait() == 0 and count == 50 * FPS
    sources = [folder1 / 'trajectory.npz', folder1 / 'independent_audit.json',
               folder1 / 'motor_axis_audit.json', folder2 / 'furniture_trajectory.npz',
               folder2 / 'independent_audit.json', Path(__file__).parent / 'validation.json',
               Path(__file__)]
    manifest = dict(duration_s=50, frames=count, fps=FPS, resolution=[W, H],
                    video_sha256=sha(movie), source_sha256={str(p): sha(p) for p in sources},
                    physics_steps=0, PPO_updates=0, GPU_used=False, playback_speed=1,
                    physical_passes_depicted=1, newly_distinct_demonstrations=0,
                    truth_only_for_offline_display=True, estimator_admitted=False)
    (OUT / 'video_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest), flush=True)


if __name__ == '__main__':
    main()
