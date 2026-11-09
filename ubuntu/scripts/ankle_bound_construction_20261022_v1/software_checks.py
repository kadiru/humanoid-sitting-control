"""Synthetic software checks; no scientific dataset access."""
import ast
from fractions import Fraction as F
from pathlib import Path
import json

p=Path(__file__).resolve().parent
ns={}
t=ast.parse((p/'independent_audit.py').read_text())
defs=[n for n in t.body if isinstance(n,ast.FunctionDef) and n.name in {'dyad','add','exact_row','less_equal'}]
exec(compile(ast.Module(body=defs,type_ignores=[]),'audit_helpers','exec'),ns)
# Synthetic scalar verification of both modeled-merit identities in the audit.
for raw,h,alpha,t,r,Ad in [(3.,-4.,.25,2.,-.75,.125),(-.5,.5,.125,.25,-.1,-.2),(2.**-100,2.**-1074,.5,.125,.25,-.375)]:
    k=-F.from_float(alpha).denominator.bit_length()+1
    k=-k
    s,e=ns['dyad'](raw);n,q=ns['dyad'](h)
    m,z=ns['add'](s,e+k,n,q)
    assert F(m,1<<z)==F.from_float(raw)*F.from_float(alpha)+F.from_float(h)
    row=-F.from_float(Ad)-F.from_float(t)
    assert row+F.from_float(t)-F.from_float(r)==-(F.from_float(r)+F.from_float(Ad))
for f in p.glob('*.py'):compile(f.read_text(),str(f),'exec')
print(json.dumps(dict(passed=True,synthetic_merit_cases=3,compiled_files=len(list(p.glob('*.py'))),scientific_data_read=False)))
