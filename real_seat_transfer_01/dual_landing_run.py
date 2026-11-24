"""Use diagnostic replicas for identification and the exact archived banks for confirmation."""
from run import *
import numpy as np
import shutil
from pathlib import Path

def bank_rows(target,rows):
    original=[P/('corner_traction_'+str(s))/'bank' for s in (0,16)]
    source=[read(f/'geometry.json') for f in original]
    metadata=read(original[0]/'metadata.json')
    geoms=[]
    for j,i in enumerate(rows):
        g=dict(source[i//16][i%16])
        g.update(case=j,surface_id=j)
        geoms.append(g)
    target.mkdir()
    write(target/'geometry.json',geoms)
    write(target/'metadata.json',metadata|dict(ids=[g['label'] for g in geoms],cases=len(rows)))
    with np.load(original[0]/'bank.npz') as a,np.load(original[1]/'bank.npz') as b:
        vals={}
        for k in a.files:
            if k=='reference':vals[k]=np.arange(len(rows));continue
            if k=='q_post':vals[k]=np.stack([(a,b)[i//16][k][:,i%16] for i in rows],axis=1)
            else:vals[k]=np.stack([(a,b)[i//16][k][i%16] for i in rows])
        np.savez_compressed(target/'bank.npz',**vals)

def launch_trial(name,profile,half=None,worker='dual_landing_worker.py',extras=()):
    src=read(P/'corner_traction_0/evaluation.freeze.json')
    target=P/name;target.mkdir();bank=target/'bank'
    if half is None:bank_rows(bank,[x['asset_id'] for x in profile['diagnostic_rows']])
    else:shutil.copytree(P/('corner_traction_'+str(half))/'bank',bank)
    write(target/'landing_profile.json',profile)
    write(target/'reservation.json',dict(files={str(f):sha(f) for f in bank.iterdir()}))
    files=[*map(Path,src['inputs']),P/'dual_landing_motion.py',P/'dual_landing_run.py',target/'landing_profile.json',*[P/n for n in extras],*[Path(g['mesh_file']) for g in read(bank/'geometry.json')]]
    cmd=src['command']
    stage(target,'evaluation',bank,PREPARED/'library.npz',worker=P/worker,extra=cmd[3:cmd.index('--mode')],inputs=files,disk_reserve=DISK_RESERVE)

if __name__=='__main__':
    name,fn,*half=sys.argv[1:]
    launch_trial(name,read(P/fn),int(half[0]) if half else None)

