"""Build the sensor-indexed motion bank from the 144 verified trajectories."""

import argparse
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from repair_fresh_sitting import checked_trace
from run_repaired_sitting_ppo import EVAL_SEED, HERE, MODEL, read, sha, write
from study_live_sitting import BASE, PREPARED
from common import OBS
from policy import Learner


def prepare(out):
    torch.set_num_threads(1)
    build = read(PREPARED/'construction.json')
    checkpoint = Path(build['checkpoint'])
    library = PREPARED/'library.npz'
    assert sha(checkpoint)==build['checkpoint_sha256'] and sha(library)==build['library_sha256']
    sources = {str(p):sha(p) for p in (checkpoint,library,Path(__file__),MODEL,HERE/'study_live_sitting.py')}
    cases, observations, desired, surfaces, slip = [], [], [], [], []
    for label in ('development','test','fresh','positioned'):
        folder = BASE/label
        results, geometry, trace = checked_trace(folder,sources)
        names = results['body_names']
        feet_indices = [names.index(f'{side}_ankle_roll_link') for side in ('left','right')]
        for i,(row,seat) in enumerate(zip(results['episodes'],geometry)):
            assert row['clean']
            up = np.array([0.,0.,1.])
            feet = trace['body'][0,i,feet_indices,:3]-np.asarray(results['origins'][i])
            f = np.cross(feet[0]-feet[1],up)
            f /= np.linalg.norm(f)
            n = np.asarray(seat['normal'])
            e = f-(f@n)*n
            e /= np.linalg.norm(e)
            top = np.asarray(seat['top_center'])
            front = top+Rotation.from_quat(seat['rotation_xyzw']).as_matrix()[:,0]*seat['depth']/2
            center = feet.mean(0)
            surfaces.append([(front-center)@e,(top-center)@n/(up@n),
                             np.rad2deg(np.arctan2(n@f,n@up))])
            cases.append(row['id'])
            observations.append(trace['obs'][0,i].copy())
            desired.append(trace['desired'][:750,i].copy())
            slip.append(float(trace['loaded_slip'][:,i].max()))
    obs = torch.tensor(np.asarray(observations))
    assert obs.shape==(144,OBS) and len(set(cases))==144
    agent = Learner(np.zeros(OBS),np.ones(OBS),EVAL_SEED)
    agent.load(checkpoint)
    with np.load(library,allow_pickle=False) as z:
        agent.bind_sitting({**agent.sitting,**dict(z)})
    paths = []
    with torch.no_grad():
        for tick in range(1000):
            obs[:,335] = tick/1000
            paths.append(agent.prior(obs).numpy())
    paths = np.stack(paths,axis=1)
    np.testing.assert_array_equal(paths[:,:750],np.asarray(desired))
    out.mkdir()
    payload = out/'live_library.npz'
    np.savez_compressed(payload,paths=paths,features=np.asarray(surfaces),ids=np.asarray(cases),
                        protected_signatures=obs[:,336:].numpy(),scale=agent.sitting['signature_scale'].numpy(),
                        peak_slip=np.asarray(slip))
    write(out/'live_inputs.json',dict(paths=144,source_hashes=sources,payload_sha256=sha(payload),
         feature_order=['front_edge_along_seat_from_feet_m','plane_height_above_ankle_centroid_m','slope_degrees']))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out',type=Path)
    prepare(parser.parse_args().out.resolve())
