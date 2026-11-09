"""Explicit original-unit row scaling and hard/epigraph partition for371 support geometry.
Only the optional exact COMxy prescription changes. No negative load/phase/hover slack.
"""
import numpy as np
from low_seat_feasibility_rows_365 import rows as old_rows

def rows(p):
 # Original row ordering and physical units are unchanged for the first19760 rows.
 scale,_,groups,_=old_rows(p)
 scale=np.r_[scale,np.full(len(p['support_h']),.005)]
 groups=groups+[dict(name='nominal_support_polygon',start=19760,stop=len(scale),raw_unit='m',scale_reason='original 5 mm support margin; scaling only')]
 keep=np.ones(len(scale),bool);proof=[]
 for i in range(114):
  for j in range(8):
   if np.all(p['null'][i,j]==0) and np.all(p['wcom'][i,j]==0):
    w=float(p['wbase'][i,j])
    for offset,value,sign in [(64,w,'lower'),(72,1-w,'upper')]:
     idx=i*80+offset+j;assert value>=0;keep[idx]=False;proof.append(dict(index=idx,knot=i+1,corner_slot=j,bound=sign,value=value,null_and_com_rows_exact_zero=True))
 hard=np.ones(len(scale),bool)
 for i in range(114):hard[i*80:i*80+32]=False
 for group in groups:
  if group['name'] in ['increasing_rate','decreasing_rate','upper_linear_headroom','lower_linear_headroom','total_time']:hard[group['start']:group['stop']]=False
 return scale,keep,hard,groups,proof

def transformed(eq,iq,scale,keep,hard,slack):
 v=iq[keep]/scale[keep];v=v+np.where(hard[keep],0.,slack)
 return eq.copy(),v
