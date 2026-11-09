"""Complete nominal path with an event-aligned20ms clock; offline only.
Reuse379's FIVE event budgets exactly, but free the114 interior segment durations
through109 log-ratios. Softmax plus20ms segment floor preserves positive durations;
only mandatory contact/hover events, not every geometry knot, align to the clock.
Every original row remains checked.64 min rows encode ALL43200 clock inequalities;
their deterministic active-row derivatives are piecewise, not a smoothness proof.
Actual float32 forward replay and intermediate geometry remain mandatory later gates.
"""
from pathlib import Path
import json,numpy as np
from low_seat_support_problem_371 import make_problem
from low_seat_support_rows_371 import rows
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')
GROUPS=((0,16),(16,64),(64,99),(99,104),(104,114))

def timing(logits,budgets):
 dt=np.empty(114);J=np.zeros((114,109));off=0
 for (a,b),ticks in zip(GROUPS,budgets):
  n=b-a;z=np.r_[logits[off:off+n-1],0.];w=np.exp(z-z.max());w/=w.sum();extra=float(ticks)*.02-n*.02
  assert extra>0
  dt[a:b]=.02+extra*w;J[a:b,off:off+n-1]=extra*(np.diag(w)-np.outer(w,w))[:,:-1];off+=n-1
 assert off==109
 return dt,J

def clock_samples(C,dt,budgets):
 values=[C[0].copy()];locations=[(-1,0.,0,0)]
 for g,((a,b),ticks) in enumerate(zip(GROUPS,budgets)):
  cumulative=np.r_[0.,np.cumsum(dt[a:b])]
  for k in range(1,int(ticks)+1):
   if k==ticks:values.append(C[b].copy());locations.append((b,0.,a,1));continue
   t=k*.02;seg=min(int(np.searchsorted(cumulative,t,side='right')-1),b-a-1);idx=a+seg;alpha=(t-cumulative[seg])/dt[idx]
   assert -1e-12<=alpha<=1+1e-12
   values.append((1-alpha)*C[idx]+alpha*C[idx+1]);locations.append((idx,float(alpha),a,0))
  if b==99:
   for _ in range(25):values.append(C[b].copy());locations.append((b,0.,a,1))
 assert len(values)==676
 return np.asarray(values),np.asarray(locations,float)

def clock_rows(C,lo,hi,rates,ulp):
 delta=np.diff(C,axis=0);target=10*C[1:]-9*C[:-1]
 raw=np.concatenate([rates[None,:]*.02-delta,rates[None,:]*.02+delta,hi[None,:]-target,target-lo[None,:]],axis=1)
 tight=np.r_[4*ulp,4*ulp,32*ulp,32*ulp]
 full=raw-tight;active=np.argmin(full,axis=0);return full[active,np.arange(64)],raw,full,active,target

def make_event():
 p=make_problem();source=np.load(D/'support_path_373/final.npz');seed=source['x'].copy()
 record=json.loads((D/'discrete_path_379/results.json').read_text());counts=np.array(record['discrete']['segment_ticks']);budgets=np.array([sum(counts[a:b]) for a,b in GROUPS]);assert np.array_equal(budgets,[93,194,246,101,16])
 shape_free=np.array([k for k in p['free'] if k<1026]);logits=[]
 for a,b in GROUPS:
  # A positive research initialization, not clipping an old solution into admission.
  excess=np.maximum(seed[1026+a:1026+b]-.02,1e-6);logits.extend(np.log(excess[:-1]/excess[-1]))
 z0=np.r_[seed[shape_free],logits];assert len(z0)==895
 base_scale,base_keep,base_hard,groups,proof=rows(p);base_keep[19759]=False
 eps=float(np.finfo(np.float32).eps);ulp=np.spacing(np.maximum(abs(p['desired_lo']),abs(p['desired_hi']))).astype(float)
 static=np.zeros(20284)
 for i in range(114):
  for j in range(8):
   for off in [64,72]:
    if base_keep[i*80+off+j]:static[i*80+off+j]=8*eps
 static[12004:12212]=8*eps;static[19760:]=8*eps
 scale=np.r_[base_scale,p['rates']*.02,p['rates']*.02,np.full(32,.03)]
 keep=np.r_[base_keep,np.ones(64,bool)];hard=np.r_[base_hard,np.zeros(64,bool)]
 bounds=[p['bounds'][k] for k in shape_free]+[(None,None)]*109
 last={}
 def expand(z):
  x=seed.copy();x[shape_free]=z[:786];dt,T=timing(z[786:],budgets);x[1026:]=dt;return x,T
 def evaluate(z,derivative=False):
  if 'z' not in last or not np.array_equal(last['z'],z):
   x,T=expand(z);eq,raw=p['evaluate'](x);Y=p['last']['Y'];C=np.vstack([p['basecmd'][0,p['moving']],Y[:,14:30]]);samples,loc=clock_samples(C,x[1026:],budgets)
   cr,craw,cfull,active,targets=clock_rows(samples,p['command_lo'],p['command_hi'],p['rates'],ulp)
   tighten=static.copy()
   for start in [12212,14036]:tighten[start:start+1824]=(x[1026:,None]/.02*4*ulp).ravel()
   for start in [15860,17684]:tighten[start:start+1824]=(x[1026:,None]*32*ulp).ravel()
   last.clear();last.update(z=z.copy(),x=x,T=T,eq=eq,raw=raw,iq=np.r_[raw-tighten,cr],tighten=tighten,C=C,samples=samples,locations=loc,clock_raw=craw,clock_full=cfull,active=active,targets=targets,X=p['last']['X'].copy(),Y=Y.copy())
  if not derivative:return last['eq'].copy(),last['iq'].copy()
  if 'E' in last:return last['E'],last['I']
  E,I=p['evaluate'](last['x'],True);I=I.copy()
  for start in [12212,14036]:
   for k in range(114):I[start+k*16:start+(k+1)*16,1026+k]-=4*ulp/.02
  for start in [15860,17684]:
   for k in range(114):I[start+k*16:start+(k+1)*16,1026+k]-=32*ulp
  def pull(A):return np.c_[A[:,shape_free],A[:,1026:]@last['T']]
  EE=pull(E);II=pull(I);J=p['last']['J'];memo={}
  def sample_jac(k,j):
   key=(k,j)
   if key in memo:return memo[key]
   result=np.zeros(1140);idx,alpha,a,endpoint=last['locations'][k];idx=int(idx);a=int(a)
   if idx<0:pass
   elif endpoint:
    if idx>0:result[(idx-1)*9:idx*9]=J[idx-1,14+j]
   else:
    if idx>0:result[(idx-1)*9:idx*9]+=(1-alpha)*J[idx-1,14+j]
    result[idx*9:(idx+1)*9]+=alpha*J[idx,14+j]
    slope=(last['C'][idx+1,j]-last['C'][idx,j])/last['x'][1026+idx]
    result[1026+a:1026+idx]-=slope;result[1026+idx]-=alpha*slope
   memo[key]=result;return result
  clockI=[]
  for col,k in enumerate(last['active']):
   j=col%16;before=sample_jac(int(k),j);after=sample_jac(int(k)+1,j)
   clockI.append(before-after if col<16 else after-before if col<32 else 9*before-10*after if col<48 else 10*after-9*before)
  II=np.vstack([II,pull(np.asarray(clockI))]);last.update(E=EE,I=II);return EE,II
 return locals()
