"""Read-only timeline for interpreting one armchair attempt."""
import json,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
p=Path(sys.argv[1]);z=np.load(p/'trajectory.npz');s=np.load(p/'sensor_control.npy');mapping=json.loads((p/'contact_mapping.json').read_text())['frames'][25:]
names=json.loads((p/'manifest.json').read_text())['contact_bodies']
event=np.union1d(np.flatnonzero(z['first_failure'][:,0]),np.flatnonzero(~z['no_slip'][:,0])[:1])
print('first failure/slip',event,'flags',z['flags'][-1,0],'good',z['good'].sum(),'cam valid',s[:,18].sum())
def show(i):
    load=np.linalg.norm(z['force'][i,0,:5],axis=-1)
    print(round(.02*(i+1),2),'pitch,r,gyro,feedback,opt,gap_ref,gap_cam',np.round(np.r_[s[i,1:7],s[i,8]],3),'q',np.round(z['q'][i,0,[0,11,15]],3),'cmd',np.round(z['command'][i,0,[0,11,15]],3),'load',np.round(load),'peak/wrong/slip',np.round([z['peak_bw'][i,0],z['wrong_force'][i,0],z['loaded_slip'][i,0]],3),'inside',z['physical_inside_patch'][i,0,:5].astype(int),'hold',z['held_ticks'][i,0])
for i in np.union1d(np.arange(0,1000,50),np.r_[event,999]):
    show(i)
for i in event:
    f=mapping[i];print('at event',i, [(b,labels[:1]) for b,labels in zip(names,f['mapped']) if labels])
    print('slip',z['slip'][i,0], 'force',np.linalg.norm(z['force'][i,0],axis=-1))
