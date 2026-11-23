"""Estimate a joint-space loading direction from a guarded, physically supported descent."""
import numpy as np
from stance23_control import P

def fit():
    data = np.load(P/'supported_corner_monarch/evaluation/trajectory.npz')
    joints = np.load(P/'supported_corner_transfer.npz')['joints']
    times = np.arange(650, 700)
    assert not data['physical_failure'][times, 0].any()
    assert data['no_slip'][times, 0].all()
    assert data['physical_inside_patch'][times, 0, 0].all()
    load = data['support_bw'][times, 0]
    command = data['command'][times, 0][:, joints]
    centered = load - load.mean()
    slope = centered @ (command - command.mean(axis=0)) / (centered @ centered)
    assert .04 < np.ptp(load) < .07 and .1 < slope.max() < .36
    np.savez(P/'loaded_corner_fit.npz', slope=slope, joints=joints,
             interval=times[[0,-1]], support_range=load[[0,-1]])
    print('seat-load fraction', load[[0,-1]], 'joint slope', slope, flush=True)

if __name__ == '__main__':
    fit()
