"""Independent stdlib NPY decoding and scalar row selection; no NumPy/producer import."""
from common import *
import ast,math,struct,sys,zipfile

def array(path,member):
 with zipfile.ZipFile(path) as z:
  with z.open(member+'.npy') as f:
   assert f.read(6)==b'\x93NUMPY';v=f.read(2)[0];assert v in (1,2,3)
   n=struct.unpack('<H' if v==1 else '<I',f.read(2 if v==1 else 4))[0]
   h=ast.literal_eval(f.read(n).decode());assert not h['fortran_order'];payload=f.read()
 dt=h['descr'];shape=h['shape'];count=math.prod(shape)
 if dt.startswith('<U'):
  width=int(dt[2:])*4;assert len(payload)==count*width
  vals=[payload[i*width:(i+1)*width].decode('utf-32-le').rstrip('\0') for i in range(count)]
 else:
  fmt={'<f8':'<d','<i4':'<i','<i8':'<q','|b1':'<?'}[dt]
  assert len(payload)==count*struct.calcsize(fmt);vals=[x[0] for x in struct.iter_unpack(fmt,payload)]
 return vals,shape

def pick(values,keep,soft):
 best=None
 for i,(x,k,s) in enumerate(zip(values,keep,soft)):
  assert math.isfinite(x)
  if k and not s and (best is None or x<values[best]):best=i
 assert best is not None
 return best

def main(origin):
 admit('audit',origin)
 keep,shape=array(R/'row_arrays.npz','keep');soft,_=array(R/'row_arrays.npz','soft');assert shape==(331888,)
 groups,_=array(R/'row_arrays.npz','groups');scale,_=array(R/'row_arrays.npz','scale');constant,_=array(R/'row_arrays.npz','constant')
 old,_=array(OLD/'actual.npz','actual_rows');new,_=array(NEW/'actual.npz','actual_rows');oa,_=array(OLD/'actual.npz','auxiliary_rows');na,_=array(NEW/'actual.npz','auxiliary_rows');oy,_=array(OLD/'actual.npz','Y');ny,_=array(NEW/'actual.npz','Y')
 of,osh=array(OLD/'mapped.npz','features');nf,nsh=array(NEW/'mapped.npz','features');assert osh==nsh==(1410,75)
 selection={'before':pick(old,keep,soft),'after':pick(new,keep,soft)};report=read(O/'attribution.json');assert report['selected']==selection
 ids,_=array(R/'all_rows_matrix.npz','indices');ptr,_=array(R/'all_rows_matrix.npz','indptr');coef,_=array(R/'all_rows_matrix.npz','data')
 canon,_=array(G/'mesh.npz','canonical');labels,_=array(G/'mesh.npz','label_to_state');times=read(G/'exact_times.json');rowmap=read(R/'original_row_map.json')
 assert len(report['records'])==len(set(selection.values()))<=2
 for record,r in zip(report['records'],sorted(set(selection.values()))):
  terms=[]
  for k in range(ptr[r],ptr[r+1]):
   col=ids[k];s=col//75;j=col%75;label=canon[s]
   terms.append(dict(column=col,state=s,feature=j,coefficient=coef[k],canonical_label=label,exact_time=times[label],alias_labels=[i for i,v in enumerate(labels) if v==s],auxiliary_before=of[col],auxiliary_after=nf[col],actual_before=of[col] if j<24 else oy[s*51+j-24],actual_after=nf[col] if j<24 else ny[s*51+j-24]))
  expected=dict(row=r,unfiltered_446_row=rowmap[r],group=groups[r],scale=scale[r],constant=constant[r],actual_before=old[r],actual_after=new[r],auxiliary_before=oa[r],auxiliary_after=na[r],physical_signed_residual_before=old[r]*scale[r],physical_signed_residual_after=new[r]*scale[r],terms=terms)
  # Hex float comparison preserves signs of zero as well as numerical values.
  def exact(a,b):
   if isinstance(a,float):return isinstance(b,float) and a.hex()==b.hex()
   if isinstance(a,dict):return a.keys()==b.keys() and all(exact(v,b[k]) for k,v in a.items())
   if isinstance(a,list):return len(a)==len(b) and all(exact(x,y) for x,y in zip(a,b))
   return type(a)==type(b) and a==b
  assert exact(expected,record);check(origin)
 assert old[selection['before']]==read(OLD/'decision.json')['summary']['actual_hard_min']
 assert new[selection['after']]==read(NEW/'decision.json')['summary']['actual_hard_min']
 write(O/'audit.json',dict(passed=True,independent_implementation='stdlib NPY parser and scalar minimum with hex-value comparison',selected=selection,records=len(report['records']),new_matrix_products=0,model_calls=0,solver_calls=0,candidate_accepted=False,physical_success=False));check(origin)
if __name__=='__main__':main(float(sys.argv[1]))
