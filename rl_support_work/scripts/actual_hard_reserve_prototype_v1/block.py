"""Source-level adapter for the existing X24/Y51/75-feature layout.

Inputs are explicit data supplied by a future separately frozen study loader.
This module never loads a study point, assembles a study LP, or evaluates FK.
"""
from fractions import Fraction as Q
from reserve import rational, canonical


def build(x, actual_y, jacobian, physical_basis, x_profile, y_profile, fixed=False):
    if not (len(x)==len(physical_basis)==len(x_profile)==24
            and len(actual_y)==len(jacobian)==len(y_profile)==51
            and all(len(row)==24 for row in jacobian)):
        raise ValueError('Wrong X24/Y51/51x24 layout')
    basis=[canonical(row.items()) for row in physical_basis]
    if fixed and any(basis):
        raise ValueError('Fixed state has nonzero physical direction')
    base={j:rational(v) for j,v in enumerate(list(x)+list(actual_y))}
    transform={j:row for j,row in enumerate(basis)}
    for output in range(51):
        coefficients={}
        for k in range(24):
            derivative=rational(jacobian[output][k])
            if derivative==0:
                continue
            for column,a in basis[k].items():
                coefficients[column]=coefficients.get(column,Q(0))+derivative*a
        transform[24+output]={c:a for c,a in sorted(coefficients.items()) if a!=0}
    profile={j:Q(0) if fixed else rational(v)
             for j,v in enumerate(list(x_profile)+list(y_profile))}
    if any(v<0 for v in profile.values()):
        raise ValueError('Negative profile')
    return dict(base_actual=base,transform=transform,profile=profile)
