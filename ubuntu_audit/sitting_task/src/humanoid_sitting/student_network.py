"""Small full-target student. This module has no simulator or teacher imports."""
import torch
from torch import nn

class Student(nn.Module):
    def __init__(self, inputs=99, outputs=37, width=256):
        super().__init__()
        self.register_buffer('mean',torch.zeros(inputs))
        self.register_buffer('std',torch.ones(inputs))
        self.register_buffer('target_mean',torch.zeros(outputs))
        self.register_buffer('target_std',torch.ones(outputs))
        self.net=nn.Sequential(nn.Linear(inputs,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU(),nn.Linear(width,outputs))
    def forward(self,x):
        return self.net(((x-self.mean)/self.std).clamp(-20,20))*self.target_std+self.target_mean
    @classmethod
    def load(cls,path,device='cpu'):
        data=torch.load(path,map_location=device,weights_only=False)
        model=cls(**data['architecture']).to(device);model.load_state_dict(data['state_dict']);model.eval();return model,data

class StructuredStudent(nn.Module):
    """Learned geometry/time/contact trajectory plus a fitted bounded balance head.

The base network receives static geometry and timing, avoiding spurious feedback
from imitation-only joint-state correlations. Balance gains are demonstration-fit.
"""
    def __init__(self,inputs=99,outputs=37,width=256,ankle_indices=(15,16),minimal_geometry=False):
        super().__init__();self.ankle_indices=list(ankle_indices);self.minimal_geometry=minimal_geometry
        self.register_buffer('mean',torch.zeros(10));self.register_buffer('std',torch.ones(10))
        self.register_buffer('target_mean',torch.zeros(outputs));self.register_buffer('target_std',torch.ones(outputs))
        self.register_buffer('balance_gain',torch.zeros(2))
        self.net=nn.Sequential(nn.Linear(10,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU(),nn.Linear(width,outputs))
    def features(self,x):
        return torch.cat(((x[:,85]+x[:,91]+.05).unsqueeze(1),torch.zeros_like(x[:,86:88]) if self.minimal_geometry else x[:,86:88],x[:,89:91],x[:,93:95],x[:,92:93],x[:,97:99]),dim=1)
    def base(self,x):
        return self.net((self.features(x)-self.mean)/self.std)*self.target_std+self.target_mean
    def forward(self,x):
        q=self.base(x)
        balance=(x[:,95]*self.balance_gain[0]+x[:,0]*self.balance_gain[1]).clamp(-.35,.35)*(x[:,96]<.25)
        q[:,self.ankle_indices]+=balance[:,None]
        return q

# Both formats remain loadable; the initial failed BC checkpoint is preserved.
def load_student(path,device='cpu'):
    data=torch.load(path,map_location=device,weights_only=False)
    cls=StructuredStudent if data.get('variant')=='structured' else Student
    model=cls(**data['architecture']).to(device);model.load_state_dict(data['state_dict']);model.eval();return model,data
Student.load=staticmethod(load_student)
