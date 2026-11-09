"""CPU-only motion-library preparation and exact command reconstruction audit."""
import argparse
from pathlib import Path
import numpy as np
import torch
from common import read,write,sha
from policy import Learner,command,teacher_targets

def main():
    p=argparse.ArgumentParser();p.add_argument('--bank',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    torch.set_num_threads(1);bank=Path(a.bank);out=Path(a.output);out.mkdir(exist_ok=False)
    meta=read(bank/'metadata.json')
    with np.load(bank/'bank.npz',allow_pickle=False) as z:
        keep=np.flatnonzero(z['admitted']);assert len(keep)==64
        obs=z['obs'][:,keep].copy();previous=z['previous'][:,keep,:23].copy();applied=z['command'][:,keep,:23].copy()
    initial=obs[0,:,222:333].copy()
    aug=np.concatenate((obs,np.broadcast_to(initial[None],(1000,64,111))),axis=-1)
    desired=teacher_targets(torch.from_numpy(previous),torch.from_numpy(applied),torch.tensor(meta['lower']),torch.tensor(meta['upper'])).numpy()
    library=dict(signature=initial,signature_scale=np.maximum(initial.std(0),.05),desired=desired.transpose(1,0,2).copy())
    # Full-clock recurrence audit. A small isolated fit error must not accumulate.
    running=torch.from_numpy(previous[0].copy());errors=[]
    for tick in range(1000):
        running=command(running,torch.from_numpy(desired[tick]),torch.tensor(meta['lower']),torch.tensor(meta['upper']))
        errors.append((running-torch.from_numpy(applied[tick])).abs().max().item())
    assert max(errors)<=2e-7,('NOMINAL_RECURRENCE_MISMATCH',max(errors))
    x=aug.reshape(-1,447).astype('float32');mean=x.mean(0);scale=x.std(0)
    records=[]
    for seed in [64111,64117]:
        agent=Learner(mean,scale,seed,'cpu');agent.bind_library(library)
        with torch.no_grad():
            _,choice,dist=agent.prior(torch.from_numpy(aug[0]),details=True)
            selected=library['desired'][choice.numpy()]
            identity_error=float(np.max(abs(selected-library['desired'])))
            assert identity_error<=2e-7,('AMBIGUOUS_INITIAL_SENSOR_MATCH',identity_error)
            pred=agent.choose(torch.from_numpy(x[:64]),False)[0]
            np.testing.assert_allclose(pred.numpy(),desired[0],rtol=0,atol=2e-7)
        ck=out/f'initial_{seed}.pt';agent.save(ck);records.append(dict(seed=seed,checkpoint=ck.name,sha256=sha(ck)))
    np.savez_compressed(out/'demonstrations.npz',obs=x)
    pair=((initial[:,None]-initial[None,:])/library['signature_scale'])**2
    pair=pair.mean(-1);np.fill_diagonal(pair,np.inf)
    write(out/'preparation.json',dict(passed=True,records=records,original_clean_cases=[meta['ids'][i] for i in keep],
        motion_library_size=64,sensor_dimensions=111,samples=len(x),recurrence_max_error_rad=max(errors),
        minimum_other_signature_distance=float(pair.min()),library_selection_uses_case_ids=False,
        validation_scope='Saved reconstruction only; no physical or novel-furniture success claim.',
        source={str(bank/'bank.npz'):sha(bank/'bank.npz'),str(bank/'metadata.json'):sha(bank/'metadata.json')}))

if __name__=='__main__':main()
