import tempfile,zipfile,io
from pathlib import Path
import numpy as np
from produce import choose
from independent import array,pick
with tempfile.TemporaryDirectory() as t:
 p=Path(t)/'fixture.npz';v=np.array([-9.,-2.,-2.,-1.,-100.]);k=np.array([False,True,True,True,True]);s=np.array([False,False,False,False,True])
 np.savez(p,v=v,k=k,s=s,u=np.array(['gap','normal']),i=np.array([1,2],dtype=np.int32))
 vv,shape=array(p,'v');kk,_=array(p,'k');ss,_=array(p,'s')
 assert choose(v,k,s)==pick(vv,kk,ss)==1
 assert array(p,'u')[0]==['gap','normal'] and array(p,'i')[0]==[1,2]
 assert array(p,'v')[0]==list(v)
print('PASS: independent parsing, hard mask, first-tie selection; synthetic data only')
