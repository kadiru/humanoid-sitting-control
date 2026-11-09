"""One frozen three-member fit; no hyperparameter search or physical rollout."""
import json,time,copy
from pathlib import Path
import numpy as np
import torch
from estimator import Estimator,predict

HERE=Path(__file__).resolve().parent
def main():
    torch.set_num_threads(2);started=time.monotonic()
    with np.load(HERE/'dataset.npz') as z:d=dict(z)
    selected=(d['split']==0)&d['valid'];x=torch.tensor(d['x'][selected]);y=torch.tensor(d['y'][selected]);groups=d['group'][selected]
    unique=np.unique(groups);weights=np.zeros(len(groups))
    for g in unique:
        ix=groups==g;w=1+4*(abs(y[ix,0].numpy())>.01);weights[ix]=w/w.sum()
    weights=torch.tensor(weights,dtype=torch.double)
    mean=x.mean(0);scale=x.std(0,unbiased=False).clamp_min(.01);models=[];records=[]
    for seed in (66201,66202,66203):
        torch.manual_seed(seed);model=Estimator(mean,scale);optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-5)
        initial=copy.deepcopy(model.state_dict());losses=[]
        for epoch in range(80):
            ids=torch.multinomial(weights,16384,replacement=True)
            total=0.
            for ix in ids.split(512):
                prediction=model(x[ix]);loss=torch.nn.functional.smooth_l1_loss(prediction,y[ix],beta=.02)
                optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();total+=float(loss.detach())
            losses.append(total/32)
            if (epoch+1)%20==0:print(dict(seed=seed,epoch=epoch+1,loss=losses[-1],seconds=time.monotonic()-started),flush=True)
            assert time.monotonic()-started<900,'FIT_WALL_CAP'
        assert all(torch.isfinite(p).all() for p in model.parameters())
        assert any(not torch.equal(initial[k],v) for k,v in model.state_dict().items())
        models.append(model.eval());torch.save(model.state_dict(),HERE/f'estimator_{seed}.pt')
        records.append(dict(seed=seed,epochs=80,optimizer_steps=2560,initial_loss=losses[0],final_loss=losses[-1]))
    full=torch.tensor(d['x']);predictions=[]
    with torch.no_grad():
        for batch in full.split(1024):predictions.append(torch.stack([m(batch) for m in models]).numpy())
    samples=np.concatenate(predictions,axis=1);mu=samples.mean(0);std=samples.std(0)
    calibration=(d['split']==1)&d['valid']
    bound=np.quantile(np.maximum(abs(mu[calibration]-d['y'][calibration])-3*std[calibration],0),.99,axis=0,method='higher')
    np.savez_compressed(HERE/'predictions.npz',mean=mu,std=std,radius=bound[None]+3*std,calibration_bound=bound)
    (HERE/'fit_result.json').write_text(json.dumps(dict(members=records,calibration_bound_BW=bound.tolist(),seconds=time.monotonic()-started,physical_PPO_updates=0,simulator_steps=0,admitted=False),indent=2))
if __name__=='__main__':main()
