"""Sensor-only snapshots for diagnosing seat/vertical-surface junctions."""
import argparse
import json
from pathlib import Path
import numpy as np
from humanoid_sitting.shared_frame import BodyModel, SharedFrameTracker, SensorFrame
from run_shared_frame import serial

p=argparse.ArgumentParser();p.add_argument('--output',required=True)
p.add_argument('--capture',default='shared_frame_rear_down_six_capture_01')
a=p.parse_args();root=Path(__file__).resolve().parents[1]
out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
with np.load(root/'runs'/a.capture/'sensors.npz') as ar:
    d={k:ar[k] for k in ('depth','K','q','gravity','delta_rotation','stamp','T_P_C')}
model=BodyModel(root/'runs/g1_kinematics_geometry_04/model.json')
for case in (0,3):
    observer=SharedFrameTracker(model,d['T_P_C'])
    for k,stamp in enumerate(d['stamp']):
        row=observer.update(SensorFrame(stamp,stamp,d['depth'][k,case],d['K'][case],d['q'][k,case],
                           d['gravity'][k,case],d['delta_rotation'][k,case],stamp,stamp))
        if k in (0,40,80,120,160):
            np.savez_compressed(out/f'case_{case}_frame_{k:03d}.npz',points=observer.map_points,
                normals=observer.map_normals,edges=observer.map_edges,seat_T=row['seat']['T'],
                cells=row['seat']['cells'],depth=d['depth'][k,case])
            print('SNAPSHOT',case,k,json.dumps(serial(row['quality']['seat_boundary'])),flush=True)
