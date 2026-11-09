"""Observe EXISTING contact-query returns; no extra query or control use.
Copies arrays immediately, before PhysX reuses count/start buffers for friction.
Normal and friction application points are kept separately. Never estimates a
wrench as average contact position cross resultant force.
"""
import numpy as np
class ContactReturnRecorder:
 def __init__(self):self.records=[];self.calls=0
 def observe(self,original,kind,side,substep,t,*args,**kwargs):
  result=original(*args,**kwargs);self.calls+=1
  if not 8.-1e-9<=t<=13.+1e-9:return result
  arrays=[np.asarray(a.numpy() if hasattr(a,'numpy') else a).copy() for a in result]
  assert len(arrays)==(6 if kind=='normal' else 4)
  counts=arrays[-2];starts=arrays[-1];assert counts.shape==(24,1) and starts.shape==(24,1)
  active=[]
  for i in range(24):
   start=int(starts[i,0]);count=int(counts[i,0]);assert count>=0
   if count:assert 0<=start and start+count<=len(arrays[0])
   active.extend(range(start,start+count))
  for a in arrays[:-2]:assert np.isfinite(a[active]).all()
  # Copy checks must not change any returned object or storage, including
  # unused capacity (which is not interpreted as active contacts).
  for a,b in zip(result,arrays):assert np.array_equal(np.asarray(a.numpy() if hasattr(a,'numpy') else a),b,equal_nan=True)
  self.records.append(dict(kind=kind,side=side,substep=int(substep),t=float(t),arrays=arrays,dt=float(kwargs.get('dt',args[0] if args else 0))))
  return result
 def save(self,path):
  values={};meta=[]
  for i,r in enumerate(self.records):
   meta.append({k:v for k,v in r.items() if k!='arrays'})
   for j,a in enumerate(r['arrays']):values[f'r{i}_a{j}']=a
  np.savez_compressed(path,**values)
  return dict(records=meta,query_calls=self.calls,record_count=len(meta),extra_queries=0,query_return_identity_preserved=True,all_return_buffer_values_unchanged=True)
