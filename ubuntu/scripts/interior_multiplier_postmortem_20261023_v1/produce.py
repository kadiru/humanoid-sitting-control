"""Read stored IEEE754 words only; never load the rejected primal or matrix."""
from common import *
import ast,struct,sys,zipfile
def words(path,member,shape):
 with zipfile.ZipFile(path) as z:
  with z.open(member+'.npy') as f:
   assert f.read(6)==b'\x93NUMPY';version=f.read(2);assert version[0] in (1,2,3)
   length=struct.unpack('<H' if version[0]==1 else '<I',f.read(2 if version[0]==1 else 4))[0]
   header=ast.literal_eval(f.read(length).decode());assert header['shape']==shape and header['descr']=='<f8' and not header['fortran_order']
   payload=f.read()
 count=1
 for n in shape:count*=n
 assert len(payload)==8*count
 return [x[0] for x in struct.iter_unpack('<Q',payload)]
def value(b):return struct.unpack('<d',struct.pack('<Q',b))[0]
def association(family,i):
 if family!='inequality':return dict(kind='eta' if i==78790 else 'original_direction_coordinate',index=i)
 for start,end,name in ((0,331495,'kept_auxiliary'),(331495,399745,'positive_consistency'),(399745,467995,'negative_consistency'),(467995,560262,'closed_upper_interval'),(560262,652529,'closed_lower_interval'),(652529,744796,'reserved_upper_interval'),(744796,837063,'reserved_lower_interval')):
  if start<=i<end:return dict(kind=name,index=i-start)
 raise AssertionError(i)
def conditions(family,b,lo,hi):
 magnitude=b&((1<<63)-1);negative=bool(b>>63);assert magnitude<0x7ff0000000000000
 if not magnitude:return []
 bad=[]
 if family in ('inequality','upper') and not negative:bad.append('positive_'+family+'_marginal')
 if family=='lower' and negative:bad.append('negative_lower_marginal')
 if family=='lower' and lo==0xfff0000000000000:bad.append('nonzero_lower_at_negative_infinity')
 if family=='upper' and hi==0x7ff0000000000000:bad.append('nonzero_upper_at_positive_infinity')
 return bad
def main(origin):
 admit('producer',origin);bounds=words(OLD/'arrays.npz','bounds',(78791,2));counts={};worst={};records=0
 with (O/'violations.jsonl').open('x') as out:
  for family,n in (('inequality',837063),('lower',78791),('upper',78791)):
   data=words(OLD/'backend_marginals.npz',family,(n,))
   for i,b in enumerate(data):
    if i%4096==0:check(origin)
    bad=conditions(family,b,0 if family=='inequality' else bounds[2*i],0 if family=='inequality' else bounds[2*i+1])
    if not bad:continue
    row=dict(family=family,index=i,raw_hex=value(b).hex(),raw_value=value(b),conditions=bad,association=association(family,i))
    line=json.dumps(row,allow_nan=False);assert len(line)<512;out.write(line+'\n');records+=1
    for condition in bad:
     counts[condition]=counts.get(condition,0)+1
     if condition not in worst or (b&((1<<63)-1))>worst[condition]['magnitude_bits']:
      worst[condition]=dict(magnitude_bits=b&((1<<63)-1),family=family,index=i,raw_hex=value(b).hex(),raw_value=value(b))
 assert records>0
 write(O/'summary.json',dict(classification='Read-only attribution of saved multiplier rejection',records=records,counts=counts,worst=worst,violation_trace_sha256=sha(O/'violations.jsonl'),matrices_or_primal_loaded=False,KKT_replayed=False,solver_calls=0,candidate_accepted=False,physical_success=False))
 check(origin)
if __name__=='__main__':main(float(sys.argv[1]))
