"""Measured exact finite/nonlinear error reserve. No oracle or solver calls."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'reserve_components'))
from dyadic import D,ZERO
from assemble import canonical


def evaluate(raw,dx,M_row,keep,soft,constants,actual,eta,check):
    eta=D.of(float(eta));assert ZERO<eta<=D(1,-1)
    predicted=[];profile=[]
    for sid,(x,y,J,p,xp,yp,fixed) in enumerate(raw):
        if sid%32==0:check()
        dd=[D.of(v) for v in dx[sid]]
        predicted.append([D.of(x[k])+dd[k] for k in range(24)]+
            [D.of(y[j])+sum((D.of(J[j][k])*dd[k] for k in range(24)),ZERO) for j in range(51)])
        profile.append([ZERO]*75 if fixed else [D.of(v) for v in xp+yp])
    seen=set();bad_features=0;bad_margin=0;bad_actual=0;first_feature=None;first_margin=None
    minimum_lower=None;minimum_actual=None;count=0
    pair=lambda v:list(v.ratio())
    for i in range(len(keep)):
        if not keep[i] or soft[i]:continue
        if count%256==0:check()
        pred=D.of(constants[i]);real=pred;kappa=ZERO
        for f,a in canonical(M_row(i)).items():
            sid,j=divmod(f,75);value=D.of(actual[sid][j]);width=eta*profile[sid][j]
            if f not in seen:
                seen.add(f);error=value-predicted[sid][j]
                if abs(error)>width:
                    bad_features+=1
                    if first_feature is None:first_feature=dict(feature=f,error=pair(error),bound=pair(width))
            pred+=a*predicted[sid][j];real+=a*value;kappa+=abs(a)*profile[sid][j]
        lower=pred-kappa*eta
        if lower.n<0:
            bad_margin+=1
            if first_margin is None:first_margin=dict(row=i,raw_lower=pair(lower))
        if real.n<0:bad_actual+=1
        minimum_lower=lower if minimum_lower is None or lower<minimum_lower else minimum_lower
        minimum_actual=real if minimum_actual is None or real<minimum_actual else minimum_actual
        count+=1
    check()
    return dict(passed=bad_features==bad_margin==bad_actual==0,hard_rows=count,
        checked_features=len(seen),features_outside_reserve=bad_features,
        negative_tangent_reserves=bad_margin,negative_actual_rows=bad_actual,
        first_feature_error=first_feature,first_negative_reserve=first_margin,
        minimum_raw_lower=pair(minimum_lower),minimum_raw_actual=pair(minimum_actual),
        eta=pair(eta),physical_success=False)
