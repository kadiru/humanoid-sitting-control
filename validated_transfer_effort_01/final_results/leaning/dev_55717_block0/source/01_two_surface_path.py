"""Frozen motion-library interpolation and bounded residual actions.

The library is offline joint-command data, not a live simulator-state input.
Policy execution uses the requested motion, elapsed control ticks and sensors.
"""
import torch

def sample_path(values,stage):
    stage=stage.clamp(0,values.shape[1]-1)
    lower=stage.floor().long();upper=(lower+1).clamp_max(values.shape[1]-1)
    alpha=stage-lower;alpha=alpha.square()*(3-2*alpha)
    ids=torch.arange(values.shape[0],device=values.device)
    return values[ids,lower]*(1-alpha[:,None])+values[ids,upper]*alpha[:,None]

def bounded_path_command(reference,previous,actions,lower,upper,span):
    desired=(reference+span*torch.tanh(actions)).clamp(min=lower+.03,max=upper-.03)
    return previous+(.1*(desired-previous)).clamp(-.002,.002)

def residual_span(names,device='cpu'):
    return torch.tensor([.01 if ('hip_roll' in n or 'ankle_roll' in n) else .05 if ('shoulder' in n or 'elbow' in n) else .025 for n in names],device=device)
