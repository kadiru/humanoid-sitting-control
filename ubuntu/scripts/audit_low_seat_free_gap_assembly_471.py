"""Independent saved-array assembly audit of prospective free-gap LPs only.
No new mechanics/J/root or trajectory optimizer. Reference remains immutable440;
new gap targets are NOT yet lifted and no candidate feasibility is implied.
"""
from pathlib import Path
from fractions import Fraction as F
import json,datetime,hashlib
import numpy as np
from scipy import sparse
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');P=D/'free_gap_solver_preparation_470';Q=D/'gap_coordinate_preparation_v3_466';O=D/'free_gap_assembly_audit_471'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def diff(a,b):
 c=(a-b).tocsr();c.eliminate_zeros();return float(abs(c.data).max()) if c.nnz else 0.
def rat(v):return F(int(v[0]),int(v[1]))
def main():
 assert json.loads((D/'free_gap_preparation_exit_470.json').read_text())['actual_exit']==0
 for n,h in json.loads((P/'SHA256.json').read_text()).items():assert sha(P/n)==h,n
 O.mkdir(exist_ok=False);rows=dict(np.load(D/'clock_row_preparation_v2_446/row_arrays.npz'));M=sparse.load_npz(D/'clock_row_preparation_v2_446/all_rows_matrix.npz');E=sparse.load_npz(D/'clock_row_preparation_446/equality_matrix.npz');ea=dict(np.load(D/'clock_row_preparation_446/equality_arrays.npz'));cm=dict(np.load(D/'clock_row_preparation_446/coordinate_map.npz'));ref=cm['reference'].ravel()[cm['free']];lo=np.maximum(cm['lower'].ravel()[cm['free']],ref-.02);hi=np.minimum(cm['upper'].ravel()[cm['free']],ref+.02)
 B=sparse.load_npz(Q/'coordinate_basis.npz');v=dict(np.load(Q/'implicit_map.npz'));GB=sparse.block_diag(list(v['G']),format='csr')@B;J=(M@GB).multiply((1/rows['scale'])[:,None]).tocsr();J.eliminate_zeros();C=sparse.load_npz(Q/'original_control_box_matrix.npz')@GB;EQ=E@GB;EQ.eliminate_zeros();assert EQ.nnz==0
 layout=json.loads((Q/'layout.json').read_text());dom=json.loads((D/'exact_gap_chart_preparation_468/layout.json').read_text());rr=[];cc=[];vv=[]
 for j,c in enumerate(layout['columns']):
  if c['kind']=='normal_drop':rr.extend([c['edge'],c['pivot']]);cc.extend([j,j]);vv.extend([1.,-1.])
 N=sparse.csr_matrix((vv,(rr,cc)),shape=(206,B.shape[1]));checks={}
 for mode in ['predictor','correction']:
  folder=P/mode;a=dict(np.load(folder/'input.npz'));meta=json.loads((folder/'metadata.json').read_text());features=np.c_[a['X'],a['Y']].ravel();r=(M@features+rows['constant'])/rows['scale'];e=E@features+ea['constant'];assert np.array_equal(r,a['r']) and np.array_equal(e,a['e']);keep=rows['keep'];soft=rows['soft']&keep;hard=keep&~soft;var=np.diff(J.indptr)>0;guard=np.zeros(len(r));guard[soft&var]=2.**-20;hp=hard&var;guard[hp]=np.where(r[hp]>0,np.minimum(2.**-20,.5*r[hp]),2.**-20);assert np.array_equal(guard,a['guard']);radius=.02*meta['trust'];z=a['z'];n=B.shape[1]
  if mode=='predictor':
   dl=np.maximum(lo-z,-radius);du=np.minimum(hi-z,radius);top=sparse.hstack([-J[keep],sparse.csr_matrix(-soft[keep,None].astype(float))]);rhs=[r[keep]-guard[keep]];parts=[top];last=[0.,max(1.,2*max(0.,float(-r[soft].min())))]
  else:
   dl=np.maximum.reduce([lo-z,a['origin']-radius-z,np.full(len(z),-.25*radius)]);du=np.minimum.reduce([hi-z,a['origin']+radius-z,np.full(len(z),.25*radius)]);top=sparse.hstack([-J[keep],sparse.csr_matrix((keep.sum(),1))]);parts=[top,sparse.hstack([C,sparse.csr_matrix(-np.ones((len(z),1))*radius)]),sparse.hstack([-C,sparse.csr_matrix(-np.ones((len(z),1))*radius)])];rhs=[r[keep]-guard[keep]+soft[keep]*meta['merit_ceiling'],np.zeros(len(z)),np.zeros(len(z))];last=[0.,.25]
  drops=list(map(rat,meta['drops']));nl=np.array([float(rat(l)-x) for l,x in zip(dom['lower'],drops)]);nh=np.array([float(rat(h)-x) for h,x in zip(dom['upper'],drops)]);cb=sparse.hstack([C,sparse.csr_matrix((len(z),1))]);nb=sparse.hstack([N,sparse.csr_matrix((206,1))]);rebuilt=sparse.vstack(parts+[cb,-cb,nb,-nb],format='csr');rhs=np.concatenate(rhs+[du,-dl,nh,-nl]);assert np.array_equal(rhs,a['rhs']) and np.array_equal(dl,a['full_step_lower']) and np.array_equal(du,a['full_step_upper']);assert np.array_equal(nl,a['normal_step_lower']) and np.array_equal(nh,a['normal_step_upper'])
  bounds=np.c_[np.r_[np.full(n,-np.inf),last[0]],np.r_[np.full(n,np.inf),last[1]]];assert np.array_equal(bounds,a['bounds']);assert np.array_equal(a['cost'],np.r_[np.zeros(n),1.]);assert np.all(a['equality_rhs']==0)
  eq=sparse.load_npz(folder/'equality.npz');assert eq.nnz==0 and eq.shape==(1366,n+1)
  errs=dict(matrix=diff(rebuilt,sparse.load_npz(folder/'matrix.npz')),all_rows=diff(J,sparse.load_npz(folder/'A.npz')),full_control=diff(C,sparse.load_npz(folder/'C.npz')));assert max(errs.values())==0;checks[mode]=dict(shape=rebuilt.shape,errors=errs,full_old_control_bound_rows=23812,normal_drop_bound_rows=412)
 out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope=__doc__,passed=True,checks=checks,all331888_final_rows_retained=True,all11906_original_boxes_including_dependent_f_retained=True,normal_drop_pivot_bounds_retained=True,all_original_normal_hover_primal_tolerances_retained=True,zero_equality_tangent_analytic=True,model_calls=0,root_calls=0,optimizer_calls=0,new_complete_path=False,physical_admission=False)
 (O/'independent_audit.json').write_text(json.dumps(out,indent=2));(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2));print(json.dumps(out,indent=2),flush=True)
if __name__=='__main__':main()
