"""Synthetic storage/order/cache tests only; no simulator or physics."""
from pathlib import Path
from types import SimpleNamespace as NS
import datetime, hashlib, json
import numpy as np
import torch
import warp as wp
from low_seat_passive_recorder_196 import PassiveJointRecorder

D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')
O=D/'passive_recorder_preparation_v2_196'

def fixture(device, mode='good'):
    q=torch.arange(12,dtype=torch.float32,device=device).reshape(3,4)
    qd=q/10; order=torch.tensor([2,0,3,1],device=device,dtype=torch.int32)
    publicq=torch.full_like(q,-2);publicqd=torch.full_like(qd,-3)
    data=NS(has_joint_ordering=True,joint_ordering=NS(user_to_backend=wp.from_torch(order)),
        _joint_pos=NS(data=wp.from_torch(publicq),timestamp=3.),_joint_vel=NS(data=wp.from_torch(publicqd),timestamp=3.))
    def qgetter():
        if mode=='mutate':publicq[0,0]=5
        return data._joint_pos.data if mode=='alias' else wp.from_torch(q)
    data._root_view=NS(get_dof_positions=qgetter,get_dof_velocities=lambda:wp.from_torch(qd))
    collection=NS(_implicit_executor=NS(kernel_inputs=[None]*3+[data._joint_pos.data,data._joint_vel.data]),_applied_effort=wp.from_torch(torch.zeros_like(q)))
    rec=PassiveJointRecorder(data,collection)
    if mode=='good':
        rec.capture(1600,8.,torch.ones_like(q))
        assert np.array_equal(rec.records[0]['q'],q.cpu().numpy()[:,[2,0,3,1]])
        q.add_(.1);qd.add_(.2)
        rec.capture(1601,8.005,torch.ones_like(q))
        assert np.array_equal(rec.records[1]['q'],q.cpu().numpy()[:,[2,0,3,1]])
        assert np.all(rec.records[0]['public_q_after']==-2)
        assert np.all(rec.records[0]['public_qd_after']==-3)
    else:
        caught=False
        try:rec.capture(1600,8.,torch.ones_like(q))
        except AssertionError:caught=True
        assert caught,mode
    return dict(device=device,mode=mode,passed=True)

def main():
    O.mkdir(exist_ok=False)
    wp.init()
    checks=[fixture(device,mode) for device in ['cpu','cuda:1'] for mode in ['good','alias','mutate']]
    sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__).resolve(),Path(__file__).with_name('low_seat_passive_recorder_196.py')]}
    r=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,checks=checks,physics_ticks=0,optimizer_updates=0,physical_admission=False,sources=sources,
        interpretation='Synthetic CPU/GPU storage checks, not runtime replay or physical admission. Real ordering, pointer separation, cache preservation and full prefix exactness remain required.')
    (O/'results.json').write_text(json.dumps(r,indent=2));(O/'SHA256.json').write_text(json.dumps({'results.json':hashlib.sha256((O/'results.json').read_bytes()).hexdigest()},indent=2));print(json.dumps(r),flush=True)

if __name__=='__main__':main()
