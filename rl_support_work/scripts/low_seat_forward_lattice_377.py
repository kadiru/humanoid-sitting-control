"""Offline legal-float32 target construction, never a runtime sensor/velocity substitute."""
import numpy as np

def key(x):
 b=int(np.float32(x).view(np.uint32));return b^0x80000000 if b<0x80000000 else (~b)&0xffffffff

def value(i):
 b=i^0x80000000 if i>=0x80000000 else (~i)&0xffffffff
 return np.array(b,np.uint32).view(np.float32)[()]

def forward(previous,target,lo,hi):
 return np.float32(previous+np.clip(np.float32(np.float32(.1)*np.float32(np.clip(target,lo,hi)-previous)),np.float32(-.003),np.float32(.003)))

def target_for_goal(previous,goal,lo,hi):
 # Exact lattice bisection, then choose nearest of the two bracketing actual outputs.
 # Report any departure from the proposed rounded goal; NEVER relabel it exact.
 a=key(lo);b=key(hi)
 if goal<forward(previous,lo,lo,hi) or goal>forward(previous,hi,lo,hi):return None
 while a<b:
  mid=(a+b)//2
  if forward(previous,value(mid),lo,hi)<goal:a=mid+1
  else:b=mid
 choices=[a]+([a-1] if a>key(lo) else [])
 best=min(choices,key=lambda k:(abs(float(forward(previous,value(k),lo,hi))-float(goal)),k))
 target=value(best);out=forward(previous,target,lo,hi)
 return target,out
