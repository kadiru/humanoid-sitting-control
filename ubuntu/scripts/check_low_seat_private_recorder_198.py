"""Synthetic identity/permutation/private-storage checks; zero physics."""
from pathlib import Path
from types import SimpleNamespace as NS
import datetime,hashlib,json
import numpy as np
import torch
import warp as wp
from low_seat_private_recorder_198 import PassiveJointRecorder
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');O=D/'private_recorder_preparation_198'

def fixture(device,ordered,mode):
    q=torch.arange(12,dtype=torch.float32,device=device).reshape(3,4);qd=q/10
    publicq=torch.full_like(q,-2);publicqd=torch.full_like(q,-3)
    order=torch.tensor([2,0,3,1],device=device,dtype=torch.int32)
    data=NS(has_joint_ordering=ordered,joint_ordering=NS(user_to_backend=wp.from_torch(order)) if ordered else None,
        _joint_pos=NS(data=wp.from_torch(publicq),timestamp=3.),_joint_vel=NS(data=wp.from_torch(publicqd),timestamp=3.))
    def desc(x):return NS(data_address=int(data._joint_pos.data.ptr) if mode=='bad_descriptor' else int(x.ptr),tensor=x)
    def getq(d):
        if mode=='mutate':publicq[0,0]=5
        wp.copy(d.tensor,wp.from_torch(q));return True
    def getqd(d):wp.copy(d.tensor,wp.from_torch(qd));return True
    data._root_view=NS(count=3,max_dofs=4,_frontend=NS(get_tensor_desc=desc),_backend=NS(get_dof_positions=getq,get_dof_velocities=getqd))
    collection=NS(_implicit_executor=NS(kernel_inputs=[None]*3+[data._joint_pos.data,data._joint_vel.data]),_applied_effort=wp.from_torch(torch.zeros_like(q)))
    caught=False
    try:
        rec=PassiveJointRecorder(data,collection);rec.capture(1600,8.,torch.ones_like(q))
    except AssertionError:caught=True
    if mode!='good':assert caught
    else:
        assert not caught
        ix=[2,0,3,1] if ordered else [0,1,2,3]
        assert np.array_equal(rec.records[0]['q'],q.cpu().numpy()[:,ix])
        q.add_(.1);qd.add_(.2);rec.capture(1601,8.005,torch.ones_like(q))
        assert np.array_equal(rec.records[1]['q'],q.cpu().numpy()[:,ix])
        assert np.all(rec.records[1]['public_q_after']==-2)
    return dict(device=device,ordered=ordered,mode=mode,passed=True)

def main():
    O.mkdir(exist_ok=False);wp.init()
    checks=[fixture(device,ordered,mode) for device in ['cpu','cuda:1'] for ordered in [False,True] for mode in ['good','bad_descriptor','mutate']]
    src=[Path(__file__).resolve(),Path(__file__).with_name('low_seat_private_recorder_198.py')]
    result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,checks=checks,physics_ticks=0,optimizer_updates=0,physical_admission=False,sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in src},scope='Synthetic caller-owned destination descriptors, identity/permuted ordering, corrupt-descriptor and cache-mutation rejection. Real API and full-prefix replay remain unvalidated.')
    (O/'results.json').write_text(json.dumps(result,indent=2));(O/'SHA256.json').write_text(json.dumps({'results.json':hashlib.sha256((O/'results.json').read_bytes()).hexdigest()},indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
