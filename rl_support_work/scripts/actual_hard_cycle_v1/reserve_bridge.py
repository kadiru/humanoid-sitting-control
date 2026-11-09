"""Integration of new actual-hard guards with immutable original stage inputs.

No entry point. Callers must authorize and verify frozen sources/inputs first.
The original completed cycle's outputs and points are never loaded here.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'reserve_components'))


def same(a,b):
    assert a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()


def sparse_same(a,b):
    a=a.tocsr();b=b.tocsr();assert a.shape==b.shape
    for name in ('indptr','indices','data'):same(getattr(a,name),getattr(b,name))


def context(c,origin):
    import numpy as np
    from scipy import sparse
    import adapter
    reference=dict(np.load(c.BASE/'reference.npz',allow_pickle=False))
    state=dict(np.load(c.CURRENT/'state.npz',allow_pickle=False))
    f=np.load(c.ACCEPTED/'mapped.npz',allow_pickle=False)['features']
    y=dict(np.load(c.ACCEPTED/'actual.npz',allow_pickle=False))
    j=dict(np.load(c.ACCEPTED/'accepted_J.npz',allow_pickle=False))
    same(y['changed'],j['changed'])
    z=dict(reference)
    for key,accepted in [('features',f),('actualY',y['Y']),('J',j['J'])]:
        same(state[key],accepted);assert np.isfinite(accepted).all();z[key]=accepted
    mesh=dict(np.load(c.D/'clock_transcription_preparation_444/mesh.npz',allow_pickle=False))
    same(z['features'][mesh['fixed']],reference['features'][mesh['fixed']])
    rows=dict(np.load(c.D/'clock_row_preparation_v2_446/row_arrays.npz',allow_pickle=False))
    M=sparse.load_npz(c.D/'clock_row_preparation_v2_446/all_rows_matrix.npz').tocsr()
    P=sparse.load_npz(c.BASE/'feature_basis.npz').tocsr()
    H=sparse.load_npz(c.CURRENT/'H.npz').tocsr()
    im=c.read(c.CURRENT/'intervals.json')
    assert H.shape==(92267,78790) and H.nnz==125867 and im['trust']==1
    assert len(z['varying'])==80155 and len(z['control_feature_rows'])==11906
    expected=[['original_control',i] for i in range(11906)]
    expected += [['feature_trust',int(i)] for i in z['varying']]
    expected += [['drop',i] for i in range(206)]
    assert im['owners']==expected
    assert im['drops']==c.read(c.ACCEPTED/'map.json')['drop_state']
    raw,states=adapter.make_states(z,mesh['fixed'],P)
    lo,hi=adapter.interval_values(im,H)
    row=adapter.row_inputs(rows,M)
    c.check(origin)
    return dict(z=z,mesh=mesh,raw=raw,states=states,H=H,im=im,lo=lo,hi=hi,row=row)


def build_cache(c,origin):
    """A only: assemble/audit once, save reusable primary guards and ceil(kappa)."""
    import math
    import numpy as np
    from scipy import sparse
    from array import array
    import adapter,assemble,audit,packed
    from dyadic import directed
    d=context(c,origin);row=d['row']
    reference=[]
    for raw,state in zip(d['raw'],d['states']):
        reference.append(audit.state(raw,state));c.check(origin)
    hrow=lambda i:adapter.sparse_row(d['H'],i)
    box,witness=assemble.parameter_boxes(d['states'],d['z']['varying'],hrow,
        d['im']['owners'],d['lo'],d['hi'],78790)
    refbox=audit.boxes(reference,set(map(int,d['z']['varying'])),hrow,
        d['im']['owners'],d['lo'],d['hi'],witness,78790)
    weights=array('d')
    def audited():
        records=assemble.stream(**row,states=d['states'],boxes=box,expected_hard_rows=156647)
        expected=(i for i in range(len(row['keep'])) if row['keep'][i] and not row['soft'][i])
        for i in expected:
            exact,encoded=next(records)
            audit.row(i,row['M_row'](i),row['constants'][i],row['scales'][i],reference,
                      refbox,exact,encoded)
            up=directed(exact.kappa,exact.scale,upward=True)
            # Independent exact upper enclosure; eta>=0 makes this sufficient.
            target=audit.q(exact.kappa)/audit.q(exact.scale)
            assert audit.q(up)>=target and audit.q(math.nextafter(up,-math.inf))<target
            assert math.isfinite(up) and up>=0
            weights.append(up);yield encoded
        assert next(records,None) is None
    buffers=packed.collect(audited(),156647,78791,7205762,lambda:c.check(origin))
    G=sparse.csr_matrix((np.frombuffer(buffers['data'],dtype=np.float64),
         np.frombuffer(buffers['indices'],dtype=np.uint32),np.frombuffer(buffers['indptr'],dtype=np.uint64)),
         shape=(156647,78791))
    rhs=np.frombuffer(buffers['rhs'],dtype=np.float64);up=np.frombuffer(weights,dtype=np.float64)
    assert G[:,-1:].nnz==0 and up.shape==(156647,) and np.isfinite(up).all()
    sparse.save_npz(c.O/'hard_matrix.npz',G)
    np.savez(c.O/'hard_arrays.npz',rhs=rhs,kappa_upper=up,
             original_ids=np.frombuffer(buffers['original_ids'],dtype=np.uint32))
    c.write(c.O/'hard_boxes.json',{str(col):dict(column=w.column,feature=w.feature,
        interval=w.interval,lower=w.lower,upper=w.upper) for col,w in witness.items()})
    proof=dict(passed=True,hard_rows=156647,primary_nnz=int(G.nnz),physical_boxes=len(box),
       independent_composition_encoding_box_audit=True,upward_kappa_audited=True,
       actual_origin='finite_model_step_20261023_v1',old_arithmetic_replayed=False,
       files_sha256={n:c.sha(c.O/n) for n in ('hard_matrix.npz','hard_arrays.npz','hard_boxes.json')})
    c.write(c.O/'hard_input_audit.json',proof);c.check(origin)
    return G,rhs,witness


def cache(c,origin,interior=False):
    import numpy as np
    from scipy import sparse
    import assemble
    proof=c.read(c.PRIMARY/'hard_input_audit.json')
    assert proof['passed'] and proof['upward_kappa_audited'] and proof['hard_rows']==156647
    for name,h in proof['files_sha256'].items():assert c.sha(c.PRIMARY/name)==h
    G=sparse.load_npz(c.PRIMARY/'hard_matrix.npz').tocsr()
    a=dict(np.load(c.PRIMARY/'hard_arrays.npz',allow_pickle=False))
    assert G.shape==(156647,78791) and G.nnz<=7205762 and G[:,-1:].nnz==0
    assert a['rhs'].shape==a['kappa_upper'].shape==a['original_ids'].shape==(156647,)
    witness={int(k):assemble.BoxWitness(**v) for k,v in c.read(c.PRIMARY/'hard_boxes.json').items()}
    if interior:
        G=sparse.hstack([G[:,:78790],sparse.csr_matrix(a['kappa_upper'][:,None])],format='csr')
        assert G.nnz<=7362409
    c.check(origin);return G,a['rhs'],witness


def append(c,origin,M,arrays,interior=False):
    import numpy as np
    from scipy import sparse
    import packed
    G,rhs,witness=cache(c,origin,interior) if interior else build_cache(c,origin)
    a=dict(arrays);a['rhs']=np.r_[arrays['rhs'],rhs]
    a['bounds']=np.asarray(packed.explicit_bounds(arrays['bounds'],witness),dtype=np.float64)
    out=sparse.vstack([M,G],format='csr')
    assert out.shape==((993712 if interior else 809178),78791)
    assert out.nnz<=(10412995 if interior else 9820080)
    c.check(origin);return out,a


def audit_extension(c,origin,M,a,base_M,base_a,interior=False):
    import numpy as np
    G,rhs,witness=cache(c,origin,interior)
    n=base_M.shape[0]
    sparse_same(M[:n],base_M);sparse_same(M[n:],G)
    same(a['rhs'][:n],base_a['rhs']);same(a['rhs'][n:],rhs)
    same(a['cost'],base_a['cost']);same(a['equality_rhs'],base_a['equality_rhs'])
    # Independent bound intersection, no producer bound helper.
    expected=base_a['bounds'].copy()
    for col,w in witness.items():
        assert w.column==col and 0<=col<78790
        expected[col,0]=max(expected[col,0],w.lower)
        expected[col,1]=min(expected[col,1],w.upper)
    same(a['bounds'],expected);assert np.all(expected[:,0]<=expected[:,1])
    c.check(origin);return True


def point_certificate(c,origin,point,interior=False,independent=False):
    """Exact encoded NEW-row checks; inherit the proved inward implication."""
    from fractions import Fraction as F
    G,rhs,witness=cache(c,origin,interior)
    q=[F.from_float(float(x)) for x in point]
    box_ok=all(F(w.lower)<=q[col]<=F(w.upper) for col,w in witness.items())
    eta=q[-1] if interior else F(0)
    eta_ok=(0<eta<=F(1,2)) if interior else True
    minimum=None;limiting=None;negative=0
    pairs=[float(x).as_integer_ratio() for x in point] if independent else None
    for i in range(G.shape[0]):
        if i%256==0:c.check(origin)
        start,end=G.indptr[i:i+2]
        if independent:
            accumulator=0;power=0
            for k in range(start,end):
                an,ad=float(G.data[k]).as_integer_ratio();bn,bd=pairs[G.indices[k]]
                exponent=ad.bit_length()+bd.bit_length()-2
                if exponent>power:accumulator<<=exponent-power;power=exponent
                accumulator+=(an*bn)<<(power-exponent)
            value=F(accumulator,1<<power)
        else:
            value=sum((F.from_float(float(G.data[k]))*q[G.indices[k]] for k in range(start,end)),F(0))
        margin=F.from_float(float(rhs[i]))-value
        assert abs(margin.numerator).bit_length()<=4220 and margin.denominator.bit_length()<=2150
        if margin<0:negative+=1
        if minimum is None or margin<minimum:minimum=margin;limiting=i
    result=dict(passed=bool(box_ok and eta_ok and negative==0),exact_boxes=box_ok,
        eta_positive=eta_ok,eta=[eta.numerator,eta.denominator],negative_rows=negative,
        minimum_encoded_margin=[minimum.numerator,minimum.denominator],limiting_row=limiting,
        rows=156647,coefficient_products=int(G.nnz),point_modified=False,
        hard_input_audit_sha256=c.sha(c.PRIMARY/'hard_input_audit.json'))
    c.check(origin)
    if independent:assert result==c.read(c.O/'hard_point_certificate.json')
    else:c.write(c.O/'hard_point_certificate.json',result)
    return result


def posterior_check(c,d,mapped,ny,origin,independent=False):
    """Bind NEW B trace and actual C features without rebuilding J*P."""
    import json
    import numpy as np
    from scipy import sparse
    import adapter
    from dyadic import D,ZERO
    from fractions import Fraction as F
    certificate=c.read(c.INTERIOR/'hard_point_certificate.json')
    assert certificate['passed'] and certificate['eta_positive']
    point=np.load(c.INTERIOR/'witness.npy',allow_pickle=False)
    assert list(F(float(point[-1])).as_integer_ratio())==certificate['eta']
    assert c.sha(c.INTERIOR/'witness.npy')==c.read(c.INTERIOR/'audit.json')['unchanged_primal_sha256']
    traces={}
    with (c.INTERIOR/'exact_rows.jsonl').open() as source:
        for i,line in enumerate(source):
            row,n,den=json.loads(line);assert row==i
            assert abs(n).bit_length()<=4213 and den.bit_length()<=2149 and den>0 and den&(den-1)==0
            kind,index=d['intervals']['owners'][i]
            if kind=='feature_trust':assert index not in traces;traces[index]=(n,den)
    assert i+1==92267 and len(traces)==80155
    P=sparse.load_npz(c.BASE/'feature_basis.npz').tocsr()
    H=sparse.load_npz(c.CURRENT/'H.npz').tocsr()
    assert P.shape==(105750,78790) and H.shape==(92267,78790)
    positions={int(f):11906+k for k,f in enumerate(d['z']['varying'])}
    z=d['z'];raw=[];increments=[]
    for sid in range(1410):
        if sid%32==0:c.check(origin)
        basis=[adapter.sparse_row(P,75*sid+j) for j in range(24)]
        delta=[]
        for j,terms in enumerate(basis):
            feature=75*sid+j
            if terms:
                assert adapter.sparse_row(H,positions[feature])==terms
                pair=traces[feature]
            else:pair=(0,1)
            delta.append(pair if independent else D.pair(*pair))
        fixed=bool(d['mesh']['fixed'][sid])
        assert not fixed or not any(basis)
        raw.append((list(map(float,z['features'][sid,:24])),list(map(float,z['actualY'][sid])),
            [list(map(float,row)) for row in z['J'][sid]],basis,
            list(map(float,z['feature_scale'][sid,:24])),list(map(float,z['sigma'][sid])),fixed))
        increments.append(delta)
    row=adapter.row_inputs(d['row'],d['M'])
    actual=adapter.finite_features(mapped,ny)
    if independent:from posterior_audit import evaluate
    else:from posterior import evaluate
    result=evaluate(raw,increments,row['M_row'],row['keep'],row['soft'],row['constants'],
                    actual,float(point[-1]),lambda:c.check(origin))
    assert result['hard_rows']==156647
    result.update(new_B_primal_sha256=c.sha(c.INTERIOR/'witness.npy'),
                  new_B_trace_sha256=c.sha(c.INTERIOR/'exact_rows.jsonl'))
    if independent:assert result==c.read(c.O/'hard_posterior.json')
    else:c.write(c.O/'hard_posterior.json',result)
    c.check(origin);return result
