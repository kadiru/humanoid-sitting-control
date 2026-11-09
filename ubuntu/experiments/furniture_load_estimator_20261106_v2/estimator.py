"""Causal sensor-only contact-report estimator; force truth is never an input."""
import torch
from torch import nn

class Estimator(nn.Module):
    def __init__(self,mean,scale):
        super().__init__()
        self.register_buffer('mean',torch.as_tensor(mean,dtype=torch.float32))
        self.register_buffer('scale',torch.as_tensor(scale,dtype=torch.float32).clamp_min(.01))
        self.net=nn.Sequential(nn.Linear(333,128),nn.ELU(),nn.Linear(128,128),nn.ELU(),nn.Linear(128,3))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)

    def forward(self,history):
        assert history.shape[-1]==333
        return self.net(((history-self.mean)/self.scale).clamp(-10,10))

def predict(models,history,calibration_bound):
    """Returns surface-normal / two tangent components, in bodyweight units."""
    with torch.no_grad():
        samples=torch.stack([m(history) for m in models]);mean=samples.mean(0)
        radius=calibration_bound+3*samples.std(0,unbiased=False)
    return mean,radius
