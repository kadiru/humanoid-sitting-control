"""Control-rate torque proxy over the final five seconds; no physical classifier."""
import torch

class RestEffortWindow:
    def __init__(self, environments, device):
        self.samples=torch.zeros((environments,250),device=device,dtype=torch.float32)
        self.cursor=0
        self.count=0

    def update(self, torque_squared):
        if torque_squared.shape!=self.samples[:,0].shape or not torch.isfinite(torque_squared).all() or (torque_squared<0).any():
            raise ValueError('Invalid torque-squared proxy')
        self.samples[:,self.cursor]=torque_squared
        self.cursor=(self.cursor+1)%250
        self.count=min(250,self.count+1)
        return self.samples.sum(1)/self.count
