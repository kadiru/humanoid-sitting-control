"""Saved-only row/acceptance localization; no model, root, AD or optimizer."""
from pathlib import Path
import json,datetime
import numpy as np
from scipy import sparse
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')
def main():
 V=D/'clock_row_preparation_v2_446';O=D/'simplex_path_462';a=dict(np.load(V/'row_arrays.npz'));M=sparse.load_npz(V/'all_rows_matrix.npz');mesh=dict(np.load(D/'clock_transcription_preparation_444/mesh.npz'));records=json.loads((D/'contact_lift_440/records.json').read_text());f=np.load(O/'final.npz');r=f['rows'];g=a['groups'];keep=a['keep'];out=[]
 def locate(i,rr):
  n=np.unique(M.indices[M.indptr[i]:M.indptr[i+1]]//75);labels=mesh['canonical'][n]
  return dict(row=int(i),group=str(g[i]),scaled_residual=float(rr[i]),unscaled_residual=float(rr[i]*a['scale'][i]),scale=float(a['scale'][i]),states=n.tolist(),labels=labels.tolist(),times_s=[records[j]['time_s'] for j in labels])
 for name in sorted(set(g.tolist())):
  ids=np.flatnonzero((g==name)&keep)
  if len(ids)==0:
   out.append(dict(group=name,rows=0,negative=0,minimum=None,scope='Audited constant/time rows omitted from optimizer only; unchanged full rows remain saved'));continue
  j=ids[np.argmin(r[ids])];out.append(dict(group=name,rows=len(ids),negative=int(np.sum(r[ids]<0)),minimum=locate(j,r)))
 history=[json.loads(x) for x in (O/'history.jsonl').read_text().splitlines()];trials=[]
 for h in history:
  for t in h['attempts']:
   P=O/f"linear_{t['linear_index']:02d}";v=dict(t)
   if (P/'nonlinear_rows.npz').exists():
    rr=np.load(P/'nonlinear_rows.npz')['rows'];hard=keep&~a['soft'];ids=np.flatnonzero(hard&(rr<0));v['negative_hard_groups']={name:int(np.sum(g[ids]==name)) for name in sorted(set(g[ids].tolist()))};v['worst_hard_rows']=[locate(i,rr) for i in ids[np.argsort(rr[ids])[:12]]]
   trials.append(v)
 result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope=__doc__,final_groups=out,attempts=trials,model_calls=0,optimizer_calls=0,physical_admission=False)
 p=D/'simplex_saved_localization_465.json';assert not p.exists();p.write_text(json.dumps(result,indent=2));print(json.dumps(dict(negative_groups=[x for x in out if x['negative']],rejected=[dict(linear=x['linear_index'],hard=x.get('negative_hard_groups'),equality=x.get('total_normal_maximum'),hover=x.get('hover_normalized_error')) for x in trials if not x.get('accepted')]),indent=2))
if __name__=='__main__':main()
