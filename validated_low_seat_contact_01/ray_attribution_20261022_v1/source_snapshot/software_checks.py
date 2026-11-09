"""Synthetic arithmetic checks only. No project datasets or scientific launch."""
import ast
from fractions import Fraction as F
import json
import math
from pathlib import Path
from types import SimpleNamespace

here=Path(__file__).resolve().parent
def functions(name,selected,namespace):
    tree=ast.parse((here/name).read_text())
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in selected]
    assert len(nodes)==len(selected)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(here/name),'exec'),namespace)
    return namespace
p=functions('producer.py',{'dyad','add','ratio','row_values'},{'F':F})
a=functions('independent_audit.py',{'grid'},{'Q':1<<1074})
cases=[([.5,-.25],[.75,.125],.25),([1.,-1.],[1.,1.],0.),
       ([2.**-1022,2.**-1074],[2.**-52,1.],2.**-1074),
       ([2.**900,-2.**900],[2.**-900,2.**-900],1.),
       ([math.nextafter(1.,2.),-1.],[1.,1.],2.**-52)]
for coeff,values,rhs in cases:
    M=SimpleNamespace(indptr=[0,len(coeff)+1],indices=list(range(len(coeff)))+[78790],data=coeff+[-1.])
    mu=.125;v=[(0,0)]*78791
    for j,x in enumerate(values):v[j]=p['dyad'](x)
    v[-1]=p['dyad'](-mu/2)
    m,me,s,se=p['row_values'](M,0,v,p['dyad'](mu),rhs)
    expected=sum((F.from_float(c)*F.from_float(x) for c,x in zip(coeff,values)),F(0))+F.from_float(mu)/2
    assert F(s,1<<se)==expected and F(m,1<<me)==F.from_float(rhs)+F.from_float(mu)
    slope=sum(a['grid'](c)*a['grid'](x) for c,x in zip(coeff,values))+a['grid'](-1.)*a['grid'](-mu/2)
    assert F(slope,1<<2148)==expected
    if s>0:
        n,d=p['ratio'](m,me,s,se);assert F(n,d)==F(m,1<<me)/expected
for f in here.glob('*.py'):compile(f.read_text(),str(f),'exec')
print(json.dumps(dict(passed=True,synthetic_cases=len(cases),compiled_files=len(list(here.glob('*.py'))),scientific_data_read=False)))
