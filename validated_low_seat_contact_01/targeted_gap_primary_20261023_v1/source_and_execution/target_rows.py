"""Two fixed saved-failure associations, no target search or model evaluation."""
import numpy as np
from scipy import sparse
TARGETS=((214643,622,1273),(215327,1400,631))
def build(P,J,actual_rows,row_scale):
 ri=[];ci=[];vv=[];rhs=[]
 for k,(r,a,b) in enumerate(TARGETS):
  # Physical gap output47. Original parameter-to-physical direction P stays intact.
  for state,sign in ((a,-1.),(b,1.)):
   for j in range(24):
    coefficient=-sign*float(J[state,47,j])/float(row_scale[r])
    if coefficient==0:continue
    z=P.getrow(state*75+j)
    for col,value in zip(z.indices,z.data):
     term=coefficient*float(value)
     if term:ri.append(k);ci.append(int(col));vv.append(term)
  rhs.append(float(actual_rows[r])-2.**-20)
 G=sparse.coo_matrix((vv,(ri,ci)),shape=(2,P.shape[1]+1)).tocsr();G.sort_indices()
 assert G[:,-1].nnz==0 and np.isfinite(G.data).all()
 return G,np.array(rhs)
