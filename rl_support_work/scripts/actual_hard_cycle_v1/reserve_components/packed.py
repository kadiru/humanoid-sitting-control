"""Bounded CSR buffers and exact NEW-row return gates; no solver or file IO."""
from array import array
import math
from dyadic import D, ZERO


def collect(records,nrows,ncols,max_nnz,check):
    """Consume an independently audited stream; commit only after exhaustion.

    A caller must keep these buffers private until this function returns.
    Check callbacks belong to the eventual shared deadline, not a fresh timer.
    """
    offsets=array('Q',[0]);columns=array('I');values=array('d');rhs=array('d');ids=array('I')
    if offsets.itemsize!=8 or columns.itemsize!=4 or values.itemsize!=8:
        raise ValueError('packed buffer sizes differ from resource contract')
    previous=-1
    for row in records:
        if len(rhs)%256==0:check()
        if len(rhs)>=nrows or row.row_id<=previous or row.row_id<0:
            raise ValueError('packed row count/order')
        if len(row.columns)!=len(row.values) or len(values)+len(row.values)>max_nnz:
            raise ValueError('packed nonzero budget')
        if row.columns!=tuple(sorted(set(row.columns))):
            raise ValueError('packed column order')
        if any(type(c) is not int or c<0 or c>=ncols for c in row.columns):
            raise ValueError('packed column dimension')
        if not all(math.isfinite(v) for v in row.values) or not math.isfinite(row.rhs):
            raise ValueError('nonfinite packed row')
        columns.extend(row.columns);values.extend(row.values);rhs.append(row.rhs)
        offsets.append(len(values));ids.append(row.row_id);previous=row.row_id
    check()
    if len(rhs)!=nrows:
        raise ValueError('incomplete packed matrix')
    return dict(indptr=offsets,indices=columns,data=values,rhs=rhs,original_ids=ids,
                shape=(nrows,ncols),nnz=len(values),complete=True)


def explicit_bounds(original,witnesses):
    """Install witness-derived outward boxes, preserving all pre-existing bounds."""
    bounds=[list(map(float,r)) for r in original]
    for col,w in witnesses.items():
        if col!=w.column or not 0<=col<len(bounds)-1:
            raise ValueError('bound witness coordinate')
        bounds[col][0]=max(bounds[col][0],w.lower)
        bounds[col][1]=min(bounds[col][1],w.upper)
        if bounds[col][0]>bounds[col][1]:
            raise ValueError('empty explicit bound intersection')
    return bounds


def exact_point(records,point,boxes,eta_column,expected_count,check):
    """Post-solver gate on the untouched NEW point, before finite encoding.

    Does not accept a numerical tolerance for boxes or these NEW exact rows.
    Original KKT, full-original primal, gap constraints and interval certificate
    remain separate mandatory gates in the future runner.
    """
    coords=[D.of(float(x)) for x in point]
    for c,(lo,hi) in boxes.items():
        if not lo<=coords[c]<=hi:
            raise ValueError('returned point outside exact encoding box')
    eta=ZERO if eta_column is None else coords[eta_column]
    if not ZERO<=eta<=D(1,-1):
        raise ValueError('returned eta outside box')
    count=0;previous=-1;minimum=None
    for row in records:
        if count%256==0:check()
        if row.row_id<=previous:
            raise ValueError('returned new-row order/duplicate')
        margin=row.intercept+sum((v*coords[c] for c,v in row.tangent.items()),ZERO)-row.kappa*eta
        if margin.n<0:
            raise ValueError('returned exact actual-hard tangent reserve is negative')
        minimum=margin if minimum is None or margin<minimum else minimum
        previous=row.row_id;count+=1
    check()
    if count!=expected_count:
        raise ValueError('returned exact actual-hard row count')
    return dict(passed=True,rows=count,minimum_raw_margin=minimum,eta=eta,
                point_modified=False,physical_success=False)
