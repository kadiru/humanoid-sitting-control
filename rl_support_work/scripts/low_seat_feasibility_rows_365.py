"""Explicit units and structural row identities for a feasibility-first nominal path.
Scaling changes solver coordinates only; all original rows remain final admission checks.
"""
import numpy as np

def rows(p):
 N=114;NP=103;rate=np.asarray(p['rates']);effort=np.asarray(p['effort']);scales=[];groups=[]
 def add(name,a,unit,reason):
  a=np.asarray(a,dtype=float).ravel();start=sum(len(x) for x in scales);scales.append(a);groups.append(dict(name=name,start=start,stop=start+len(a),raw_unit=unit,scale_reason=reason))
 local=np.r_[np.full(32,.03),effort,effort,np.ones(16)]
 add('local_command_effort_corner',np.tile(local,N),'rad / Nm / BW fraction','command reserve .03 rad; configured effort limits; whole body weight fraction')
 add('dependent_planning_bounds',np.tile(np.r_[[.1,.1],np.ones(12),[.1,.1],np.ones(12)],NP),'m / rad','same .1 m root and 1 rad joint nondimensionalization as357')
 add('phase_order',np.ones(208),'fraction','unit interval phase')
 for name in ['increasing_rate','decreasing_rate']:add(name,np.tile(rate*.02,N),'rad per segment','one 20 ms interval at actual reserved nominal rates')
 for name in ['upper_linear_headroom','lower_linear_headroom']:add(name,np.full(N*16,.03*.02),'rad s','existing .03 rad desired reserve times one 20 ms interval')
 for name,count in [('prehover_gap',98),('posthover_gap',5),('nonnegative_gap',104),('approach_gap_order',39)]:add(name,np.full(count,.012),'m','existing 12 mm hover height')
 add('normal_descent',np.full(5,.006*.02),'m','one 20 ms interval at existing 6 mm/s descent cap')
 add('total_time',[13.5],'s','existing total nominal transition budget')
 scale=np.concatenate(scales);assert len(scale)==19760 and np.all(scale>0)
 keep=np.ones(19760,dtype=bool);proof=[]
 # Only identically zero affine force rows are removed. No numerical Jacobian threshold.
 for i in range(N):
  for j in range(8):
   if np.all(p['null'][i,j]==0):
    w=float(p['wbase'][i,j])
    for offset,value,sign in [(64,w,'lower'),(72,1-w,'upper')]:
     index=i*80+offset+j
     assert value>=0,('negative structurally constant constraint',index,value)
     keep[index]=False;proof.append(dict(index=index,knot=i+1,corner_slot=j,bound=sign,value=value,null_row_exact_zero=True))
 assert len(proof)==392
 return scale,keep,groups,proof

def transformed(eq,iq,scale,keep,slack):
 return np.r_[iq[keep]/scale[keep]+slack,eq+slack,-eq+slack]
