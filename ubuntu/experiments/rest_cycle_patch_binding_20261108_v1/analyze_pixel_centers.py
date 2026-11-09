"""One saved-frame projection audit. Truth transforms are evaluation-only.

Tests whether translating the old observed patch is equivalent to obtaining
a fresh observation after the robot moves closer. No model solve or physics.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

R = Path('/home/kadir/humanoid-sitting')
H = Path(__file__).resolve().parent
OLD = R / 'outputs/furniture_examples_20261105_v2/sit_modern_coffee_table_01_center/frame_0.npz'
NEW = R / 'outputs/rest_cycle_startup_fix_20261108_v1/sit_coffee_center/frame_0.npz'
AUDIT = NEW.parent / 'fresh_geometry_admission.json'
PRIOR = R / 'experiments/furniture_load_repair_20261106_v3/seat_models/sit_modern_coffee_table_01_center.audit.json'
CAM_R = np.array([[0., .707106781, -.707106781], [1., 0., 0.], [0., -.707106781, -.707106781]])
CAM_T = np.array([-.07, 0., .38])


def project(points, K):
    p = (points - CAM_T) @ CAM_R
    pixels = p[:, :2] / p[:, 2, None] * np.array([K[0, 0], K[1, 1]]) + K[:2, 2]
    return pixels, p[:, 2]


def main():
    start = time.monotonic()
    with np.load(OLD) as z: old = dict(z)
    with np.load(NEW) as z: new = dict(z)
    old_pixels, _ = project(old['selected_points'], old['K'])
    saved_pixels = old['selected_pixels'][:, ::-1] + .5
    projection_error = float(np.max(np.linalg.norm(old_pixels - saved_pixels, axis=1)))
    assert projection_error < .01, ('SAVED_CAMERA_PROJECTION_MISMATCH', projection_error)
    # Evaluation-only registration. This is never a localization provider for
    # the robot, nor permission to mark past observations as current.
    old_in_new = (old['selected_points'] @ old['root_rotation'].T + old['root_position']
                  - new['root_position']) @ new['root_rotation']
    uv, depth = project(old_in_new, new['K'])
    height, width = new['depth'].shape
    in_view = ((depth > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < width)
               & (uv[:, 1] >= 0) & (uv[:, 1] < height))
    front = old['selected_points'][:, 0] >= old['selected_points'][:, 0].max() - .003
    prior = json.loads(PRIOR.read_text())['geometry_audit']
    current = json.loads(AUDIT.read_text())['geometry']
    result = dict(
        source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [OLD, NEW, AUDIT, PRIOR, Path(__file__)]},
        actual_root_translation_m=(new['root_position'] - old['root_position']).tolist(),
        old_point_count=len(uv), previous_points_outside_current_image=int((~in_view).sum()),
        front_boundary_points=int(front.sum()), front_boundary_points_outside_current_image=int((front & ~in_view).sum()),
        old_front_boundary_pixel_rows=old_pixels[front, 1].tolist(),
        projected_front_boundary_pixel_rows=uv[front, 1].tolist(), image_height=height,
        camera_projection_max_error_pixels=projection_error,
        old_observed_front_x_m=float(old['selected_points'][:, 0].max()),
        translated_old_front_x_m=float(old_in_new[:, 0].max()),
        fresh_observed_front_x_m=float(new['selected_points'][:, 0].max()),
        saved_model_minimum_contact_inset_m=prior['min_patch_inset'],
        live_model_minimum_contact_inset_m=current['min_patch_inset'],
        required_inset_m=.02, original_threshold_preserved=True,
        reset_q_max_difference_rad=float(np.max(abs(new['q'] - old['q']))),
        conclusion='Reset translation occurred correctly. The model used a translated OLD visible patch. Its front boundary projects outside the new image. A rigid geometric translation is not a fresh camera observation; no qualified map localization was available.',
        corrected_software_passed=True, camera_patch_gate_passed=False,
        motion_authorized_from_this_analysis=False, full_cycle_success=False,
        simulator_truth_used='evaluation-only saved-pose transform',
        physics_steps=0, model_solves=0, PPO_updates=0)
    image = Image.new('RGB', (1200, 730), '#f5f7fa'); draw = ImageDraw.Draw(image)
    font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font = lambda size: ImageFont.truetype(font_path, size)
    draw.text((32, 22), 'Why the sitting attempt stayed in standing', font=font(32), fill='#183444')
    draw.text((32, 72), 'Saved rear-camera images — no sitting motion was executed', font=font(22), fill='#526976')
    for offset, frame, label in [(32, old, 'Earlier standing view'), (632, new, 'After moving the reset 5 cm closer')]:
        rgb = frame['rgb'][..., :3].astype(np.uint8)
        picture = Image.fromarray(rgb).resize((512, 384), Image.Resampling.NEAREST)
        image.paste(picture, (offset, 145))
        draw.text((offset, 110), label, font=font(21), fill='#183444')
        draw.rectangle((offset, 145, offset + 511, 528), outline='#526976', width=2)
    # The old patch's front edge is plotted in its original image and where it
    # would project now, including outside the current image when clipped.
    for offset, pixels in [(32, old_pixels), (632, uv)]:
        for x, y in pixels[front]:
            x, y = offset + x * 4, 145 + y * 4
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill='#e74031')
    lines = [
        'Red: the previously observed front boundary. It now falls below the camera image.',
        f"Planned contact inset: {prior['min_patch_inset']*100:.2f} cm before / {current['min_patch_inset']*100:.2f} cm now.",
        'Required: more than 2 cm inside the observed patch. The unchanged gate correctly rejected it.',
        'Next: observe and localize a persistent surface map before approaching; do not widen the guard.']
    for i, line in enumerate(lines): draw.text((32, 570 + i * 36), line, font=font(19), fill='#183444')
    image.save(H / 'camera_patch_diagnosis.png')
    result['wall_seconds'] = time.monotonic() - start
    with (H / 'result.json').open('x') as f: json.dump(result, f, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k not in ['source_sha256', 'old_front_boundary_pixel_rows', 'projected_front_boundary_pixel_rows']}))


if __name__ == '__main__': main()
