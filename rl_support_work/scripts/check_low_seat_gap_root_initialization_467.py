"""Additional bounded root-initialization validation, not a preparation rerun.
466's two predictor-initialized batches needed zero Newton updates; their small
central difference primarily replayed the supplied tangent. Preserve those
results and test the SAME declared controls from UNPERTURBED dependent states.
This is new root work, not a repeated full-path or finite-work loop.
"""
from pathlib import Path
import json,datetime
import numpy as np
from low_seat_clock_transcription_444 import D,sha,load_inputs,Mechanics
from low_seat_gap_coordinates_466 import root,DEP,IND
P=D/'gap_coordinate_preparation_v3_466';O=D/'gap_root_initialization_467'
def main():
 O.mkdir(exist_ok=False);v=dict(np.load(P/'implicit_map.npz'));f=dict(np.load(P/'root_direction_fixture.npz'));mesh=dict(np.load(D/'clock_transcription_preparation_444/mesh.npz'));blocks=f['blocks'];d=f['direction'];h=float(f['h']);old=json.loads((P/'root_fixture_journal.json').read_text());assert all(len(x)==1 for x in old)
 s=load_inputs();oracle=Mechanics(s,mesh['canonical'][blocks]);xb=v['referenceX'][blocks];gb=v['referenceY'][blocks,47];outs=[];journal=[]
 for sign in [-1,1]:
  start=xb.copy();start[:,IND]+=sign*h*d[:,1:];target=gb+sign*h*d[:,0]
  xx,yy,history=root(oracle,start,target);outs.append(xx);journal.append(history)
  np.savez_compressed(O/('minus.npz' if sign<0 else 'plus.npz'),initial=start,X=xx,Y=yy,target_gap=target)
  (O/'journal.json').write_text(json.dumps(journal))
 assert all(len(x)>1 for x in journal),journal
 fd=(outs[1]-outs[0])/(2*h);analytic=np.einsum('nij,nj->ni',v['T'][blocks],d);error=float(abs(fd-analytic).max());np.savez_compressed(O/'comparison.npz',fd=fd,analytic=analytic,blocks=blocks,direction=d,h=h)
 assert error<2e-6,error
 out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope=__doc__,passed=True,original_predictor_fixture_newton_updates=0,new_batches=2,new_signed_states=12,newton_updates=[len(x)-1 for x in journal],derivative_error=error,unchanged_bound=2e-6,new_complete_paths=0,new_optimizers=0,physical_admission=False)
 (O/'result.json').write_text(json.dumps(out,indent=2));(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2));print(json.dumps(out,indent=2),flush=True)
if __name__=='__main__':main()
