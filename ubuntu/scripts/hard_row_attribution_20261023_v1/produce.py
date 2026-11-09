"""Saved-row attribution only; no matrix products, model, derivative or optimizer."""
from common import *
import sys

def choose(actual,keep,soft):
 import numpy as np
 ids=np.flatnonzero(keep & ~soft)
 assert len(ids)>0 and np.isfinite(actual).all()
 return int(ids[np.argmin(actual[ids])])

def main(origin):
 c=admit('producer',origin)
 for p,h in c['numpy_runtime_sha256'].items():assert sha(p)==h,p;check(origin)
 import numpy as np
 ra=np.load(R/'row_arrays.npz',allow_pickle=False);mesh=np.load(G/'mesh.npz',allow_pickle=False)
 mat=np.load(R/'all_rows_matrix.npz',allow_pickle=False)
 old=np.load(OLD/'actual.npz',allow_pickle=False);new=np.load(NEW/'actual.npz',allow_pickle=False)
 of=np.load(OLD/'mapped.npz',allow_pickle=False)['features'];nf=np.load(NEW/'mapped.npz',allow_pickle=False)['features']
 keep=ra['keep'];soft=ra['soft'];assert keep.shape==soft.shape==(331888,)
 aold=old['actual_rows'];anew=new['actual_rows'];assert aold.shape==anew.shape==(331888,)
 selected={n:choose(a,keep,soft) for n,a in [('before',aold),('after',anew)]}
 assert float(aold[selected['before']])==read(OLD/'decision.json')['summary']['actual_hard_min']
 assert float(anew[selected['after']])==read(NEW/'decision.json')['summary']['actual_hard_min']
 rowmap=read(R/'original_row_map.json');times=read(G/'exact_times.json')
 indices=mat['indices'];ptr=mat['indptr'];data=mat['data'];canon=mesh['canonical'];labels=mesh['label_to_state'];groups=ra['groups']
 assert list(mat['shape'])==[331888,105750] and len(data)==527492
 records=[]
 for r in sorted(set(selected.values())):
  terms=[]
  for k in range(int(ptr[r]),int(ptr[r+1])):
   col=int(indices[k]);s,j=divmod(col,75);lab=int(canon[s]);check(origin)
   terms.append(dict(column=col,state=s,feature=j,coefficient=float(data[k]),canonical_label=lab,exact_time=times[lab],
    alias_labels=[int(i) for i in np.flatnonzero(labels==s)],
    auxiliary_before=float(of[s,j]),auxiliary_after=float(nf[s,j]),
    actual_before=float(of[s,j] if j<24 else old['Y'][s,j-24]),actual_after=float(nf[s,j] if j<24 else new['Y'][s,j-24])))
  scale=float(ra['scale'][r]);assert scale>0
  records.append(dict(row=r,unfiltered_446_row=int(rowmap[r]),group=str(groups[r]),scale=scale,constant=float(ra['constant'][r]),
   actual_before=float(aold[r]),actual_after=float(anew[r]),auxiliary_before=float(old['auxiliary_rows'][r]),auxiliary_after=float(new['auxiliary_rows'][r]),
   physical_signed_residual_before=float(aold[r])*scale,physical_signed_residual_after=float(anew[r])*scale,terms=terms))
 report=dict(classification='Executed attribution of existing stored residuals, not new model validation',selected=selected,records=records,
  criterion='keep AND NOT soft; smallest stored normalized residual, first original row on ties',
  source_model_calls=0,new_matrix_products=0,new_J_calls=0,solver_calls=0,candidate_admission=False,physical_success=False)
 assert len(json.dumps(report))<2*1024**2
 write(O/'attribution.json',report);check(origin)
if __name__=='__main__':main(float(sys.argv[1]))
