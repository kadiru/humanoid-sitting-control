"""Independent rational endpoint audit; no producer/assembly arithmetic import."""
from fractions import Fraction as F


def evaluate(raw,dx,M_row,keep,soft,constants,actual,eta,check):
    eta=F(float(eta));assert 0<eta<=F(1,2)
    predictions=[];widths=[]
    for sid,(x,y,J,p,xp,yp,fixed) in enumerate(raw):
        if sid%32==0:check()
        dd=[F(n,d) for n,d in dx[sid]]
        predictions.append([F(x[k])+dd[k] for k in range(24)]+
            [F(y[j])+sum((F(J[j][k])*dd[k] for k in range(24)),F(0)) for j in range(51)])
        widths.append([F(0)]*75 if fixed else [eta*F(v) for v in xp+yp])
    checked=set();bad_features=0;bad_margin=0;bad_actual=0;first_feature=None;first_margin=None
    minimum_lower=None;minimum_actual=None;count=0
    pair=lambda v:[v.numerator,v.denominator]
    for i in range(len(keep)):
        if not keep[i] or soft[i]:continue
        if count%256==0:check()
        terms={}
        for f,a in M_row(i):terms[f]=terms.get(f,F(0))+F(a)
        terms={f:a for f,a in sorted(terms.items()) if a}
        lower=F(constants[i]);real=lower
        for f,a in terms.items():
            sid,j=divmod(f,75);value=F(actual[sid][j]);width=widths[sid][j]
            if f not in checked:
                checked.add(f);error=value-predictions[sid][j]
                if not -width<=error<=width:
                    bad_features+=1
                    if first_feature is None:first_feature=dict(feature=f,error=pair(error),bound=pair(width))
            lower+=a*(predictions[sid][j]-width if a>=0 else predictions[sid][j]+width)
            real+=a*value
        if lower<0:
            bad_margin+=1
            if first_margin is None:first_margin=dict(row=i,raw_lower=pair(lower))
        if real<0:bad_actual+=1
        minimum_lower=lower if minimum_lower is None or lower<minimum_lower else minimum_lower
        minimum_actual=real if minimum_actual is None or real<minimum_actual else minimum_actual
        count+=1
    check()
    return dict(passed=bad_features==bad_margin==bad_actual==0,hard_rows=count,
        checked_features=len(checked),features_outside_reserve=bad_features,
        negative_tangent_reserves=bad_margin,negative_actual_rows=bad_actual,
        first_feature_error=first_feature,first_negative_reserve=first_margin,
        minimum_raw_lower=pair(minimum_lower),minimum_raw_actual=pair(minimum_actual),
        eta=pair(eta),physical_success=False)
