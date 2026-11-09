"""CPU/GPU existing-query interception, mutation and contact-index fixtures."""
from pathlib import Path
import json,hashlib
import numpy as np
import torch
from low_seat_contact_recorder_257 import ContactReturnRecorder
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');O=D/'contact_recorder_preparation_258'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 O.mkdir(exist_ok=False);checks=[]
 for device in ['cpu','cuda:1']:
  for kind in ['normal','friction']:
   rec=ContactReturnRecorder();N=96;cnt=np.full((24,1),2,np.uint32);start=np.arange(24,dtype=np.uint32)[:,None]*4
   raw=[np.ones((N,1 if kind=='normal' else 3),np.float32),np.arange(N*3,dtype=np.float32).reshape(N,3)]
   if kind=='normal':raw.extend([np.tile([0.,0.,1.],(N,1)).astype(np.float32),np.zeros((N,1),np.float32)])
   raw += [cnt,start]
   # Adapter matches Warp's numpy() copy API on both CPU/GPU; tensors never
   # enter runtime here. Validate identity and private-copy isolation.
   class A:
    def __init__(self,a):self.x=torch.as_tensor(a.astype(np.int64) if a.dtype==np.uint32 else a,device=device)
    def numpy(self):return self.x.cpu().numpy()
   original=tuple(A(a) for a in raw);calls=[0]
   def query(dt):calls[0]+=1;return original
   answer=rec.observe(query,kind,'left',1600,8.,dt=.005);assert answer is original and calls[0]==1
   old=rec.records[0]['arrays'][-2].copy();original[-2].x.fill_(0);assert np.array_equal(rec.records[0]['arrays'][-2],old)
   rec.observe(query,kind,'left',1500,7.5,dt=.005);assert len(rec.records)==1 and calls[0]==2
   checks.append(dict(device=device,kind=kind,identity=True,private_count_copy=True,no_extra_query=True,outside_window_passthrough=True))
   original[-2].x.fill_(200)
   try:rec.observe(query,kind,'right',1601,8.005,dt=.005)
   except AssertionError:checks.append(dict(device=device,kind=kind,bad_active_bounds_rejected=True))
   else:raise AssertionError('bad count not rejected')
 out=dict(passed=True,checks=checks,physics_ticks=0,optimizer_updates=0,source_sha256={str(p.resolve()):sha(p) for p in [Path(__file__),Path('scripts/low_seat_contact_recorder_257.py')]})
 (O/'results.json').write_text(json.dumps(out,indent=2));(O/'SHA256.json').write_text(json.dumps({'results.json':sha(O/'results.json')},indent=2));print(json.dumps(dict(passed=True,fixtures=len(checks))))
if __name__=='__main__':main()
