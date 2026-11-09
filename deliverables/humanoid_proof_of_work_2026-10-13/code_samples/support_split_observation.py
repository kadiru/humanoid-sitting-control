"""Independent clearance contract, appended to the preserved 144-channel input.

144:148 clearance (m) and seat normal in the current pelvis frame;
148 distance-valid bit; 149 age of the supporting measurement (s);
150 conditional engineering error budget (m, NOT a confidence probability).
The old 140 patch-valid bit retains its original complete-patch meaning.
Unavailable distance is [0,0,0,0,0,1,.02]; it is never silently held.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

LEGACY_DIM = 144
DISTANCE_DIM = 7
OBS_DIM = LEGACY_DIM + DISTANCE_DIM


def distance_features(result, stamp):
    """Pack only current, accepted sensor-tracker estimates; no patch dependency."""
    missing = np.array([0., 0., 0., 0., 0., 1., .02], dtype=np.float32)
    if not result.get('valid', False):
        return missing
    try:
        normal = np.asarray(result['normal_P'], dtype=float)
        clearance = float(result['clearance_m'])
        age = float(result['measurement_age_s'])
        budget = float(result['allowance_m'])
        current = abs(float(result['stamp']) - float(stamp)) < 1.e-7
        finite = np.isfinite([clearance, age, budget]).all() and np.isfinite(normal).all()
        valid = (current and finite and normal.shape == (3,) and
                 abs(np.linalg.norm(normal)-1.) < .001 and
                 0. <= age <= .1+1.e-7 and 0. <= budget < .02)
    except (KeyError, TypeError, ValueError):
        valid = False
    if not valid:
        return missing
    return np.array([clearance, *normal, 1., age, budget], dtype=np.float32)


def expand_parent_state(state):
    """Zero new input weights, retain every old weight and normalizer statistic."""
    assert state['mlp.0.weight'].shape[1] == LEGACY_DIM
    expanded = {k: v.clone() for k, v in state.items()}
    w = state['mlp.0.weight']
    expanded['mlp.0.weight'] = torch.cat((w, w.new_zeros(w.shape[0], DISTANCE_DIM)), 1)
    scale = w.new_tensor([[.3, 1., 1., 1., 1., .1, .02]])
    for suffix, tail in [('mean', torch.zeros_like(scale)), ('var', scale.square()), ('std', scale)]:
        key = 'obs_normalizer._'+suffix
        expanded[key] = torch.cat((state[key], tail), 1)
    return expanded


class SplitInputLinear(nn.Linear):
    """Keep the old GEMM shape to preserve pre-training GPU action parity.

The added branch starts at zero and receives ordinary PPO gradients. Parameters
keep standard Linear state-dict keys; the loader must install this computation.
"""
    def forward(self, value):
        old = F.linear(value[..., :LEGACY_DIM].contiguous(),
                       self.weight[:, :LEGACY_DIM].contiguous(), self.bias)
        extra = F.linear(value[..., LEGACY_DIM:].contiguous(),
                         self.weight[:, LEGACY_DIM:].contiguous())
        return old + extra


def install_split_linear(model):
    layer = model.mlp[0]
    assert type(layer) is nn.Linear and layer.in_features == OBS_DIM
    # No parameter replacement: optimizer references and RNG remain untouched.
    layer.__class__ = SplitInputLinear
