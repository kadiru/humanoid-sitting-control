"""Synthetic tests only: no study files, solver, mechanics or network."""
import copy
from dataclasses import replace
from fractions import Fraction as F
import itertools
import math
import random
import unittest

from dyadic import D, ZERO, ONE, directed, ratio_float
import assemble as a
import audit as b
import returned


def fixture():
    raw=[]
    for sid in range(3):
        fixed=sid==0
        p=[[] for _ in range(24)]
        if not fixed:
            offset=23*(sid-1)
            for j in range(23):
                p[j]=[(offset+j,1.0)]
            p[23]=[(offset+j,-1.0) for j in range(16,23)]
        x=[0.0]*16+[0.125]*8
        y=[(j-25)/128.0 for j in range(51)]
        J=[[(j%7-k%5)/16.0 for k in range(24)] for j in range(51)]
        raw.append((x,y,J,p,[0.125]*24,[0.25]*51,fixed))
    states=[a.state_block(*r) for r in raw]
    reference=[b.state(r,s) for r,s in zip(raw,states)]
    varying=[sid*75+j for sid in range(1,3) for j in range(24)]
    owners=[['feature_trust',f] for f in varying]
    h=[raw[f//75][3][f%75] for f in varying]
    lo,hi=[D(-1,-3)]*len(h),[D(1,-3)]*len(h)
    box,witness=a.parameter_boxes(states,varying,h.__getitem__,owners,lo,hi,46)
    refbox=b.boxes(reference,varying,h.__getitem__,owners,lo,hi,witness,46)
    return dict(raw=raw,states=states,reference=reference,varying=varying,
                owners=owners,h=h,lo=lo,hi=hi,box=box,witness=witness,refbox=refbox)


class ArithmeticTests(unittest.TestCase):
    def test_binary64_extremes_and_cancellation(self):
        rng=random.Random(1383)
        values=[0.0,-0.0,math.ulp(0.0),-math.ulp(0.0),1.0,math.nextafter(1,0),
                float.fromhex('0x1.fffffffffffffp+1023')]
        values += [math.ldexp(rng.uniform(-1,1),rng.randrange(-1000,1000)) for _ in range(150)]
        for x,y in zip(values,values[::-1]):
            dx,dy=D.of(x),D.of(y)
            self.assertEqual(b.q(dx+dy),F(x)+F(y))
            self.assertEqual(b.q(dx*dy),F(x)*F(y))
            self.assertEqual(dx-dx,ZERO)

    def test_ratio_rounding_and_directed_endpoints(self):
        rng=random.Random(821)
        for _ in range(200):
            x=D(rng.randrange(-10000,10000),rng.randrange(-900,900))
            scale=D(rng.randrange(1,100),rng.randrange(-100,100))
            exact=b.q(x)/b.q(scale)
            self.assertEqual(ratio_float(x,scale),float(exact))
            lo,hi=directed(x,scale),directed(x,scale,True)
            self.assertLessEqual(F(lo),exact)
            self.assertGreaterEqual(F(hi),exact)
            self.assertGreater(F(math.nextafter(lo,math.inf)),exact)
            self.assertLess(F(math.nextafter(hi,-math.inf)),exact)

    def test_invalid_numeric_inputs_and_caps(self):
        for x in [math.inf,math.nan,True,F(1,3)]:
            with self.assertRaises((ValueError,TypeError)):
                D.of(x)
        with self.assertRaises(ValueError):D.pair(1,3)
        with self.assertRaises(OverflowError):D(1,16385)
        with self.assertRaises(OverflowError):ratio_float(D(1,1024),ONE)


class AssemblyTests(unittest.TestCase):
    def setUp(self):self.f=fixture()

    def test_full_physical_layout_and_conserved_load_pivot(self):
        f=self.f
        self.assertEqual(len(f['box']),46)
        self.assertEqual(len(a.validate_state_locality(f['states'],46)),46)
        st=f['states'][1]
        for out in [31,47,48,50]:
            expected=F(f['raw'][1][2][out][16])-F(f['raw'][1][2][out][23])
            self.assertEqual(b.q(st.tangent[24+out].get(16,ZERO)),expected)
        self.assertTrue(all(v==ZERO for v in f['states'][0].profile))

    def test_aliases_cancel_before_reserve(self):
        r=a.compose(5,[(146,2.0),(146,-2.0)],1.0,0.012,self.f['states'])
        self.assertEqual(r.terms,{})
        self.assertEqual(r.kappa,ZERO)
        self.assertEqual(r.tangent,{})

    def test_all_nine_observed_failure_families(self):
        # IDs are SYNTHETIC. These terms exercise the actual feature locations.
        terms=[[(146,-1.0),(221,1.0)],[(147,-0.75),(148,0.5)],
               [(222,0.5),(223,-0.25)],[(130,1.0)],[(146,-1.0)],
               [(114,1.0)],[(221,1.0)],[(146,1.0),(221,-1.0)],
               [(146,-1.0),(221,1.0)]]
        eta=46
        box={**self.f['box'],eta:(ZERO,D(1,-1))}
        rb={**self.f['refbox'],eta:(F(0),F(1,2))}
        for i,t in enumerate(terms):
            r=a.compose(i,t,0.5,0.012,self.f['states'])
            for interior in (False,True):
                ec=eta if interior else None
                encoded=a.encode(r,box,ec)
                self.assertTrue(b.row(i,t,0.5,0.012,self.f['reference'],rb,r,encoded,ec))
                self.assertTrue(set(encoded.columns)-{eta} <= set(range(46)))

    def test_state_locality_and_shape_rejected(self):
        wrong=copy.deepcopy(self.f['raw'][2]);wrong[3][0]=[(0,1.0)]
        states=self.f['states'][:2]+[a.state_block(*wrong)]
        with self.assertRaises(ValueError):a.validate_state_locality(states,46)
        raw=self.f['raw'][1]
        with self.assertRaises(ValueError):a.state_block(raw[0][:-1],*raw[1:])
        wrong=copy.deepcopy(raw);wrong[3][0]=[(0,0.5)]
        with self.assertRaises(ValueError):a.state_block(*wrong)

    def test_fixed_state_nonzero_basis_rejected(self):
        raw=list(self.f['raw'][1]);raw[-1]=True
        with self.assertRaises(ValueError):a.state_block(*raw)

    def test_box_unit_witness_tampering(self):
        f=self.f;h=copy.deepcopy(f['h']);h[0]=[(0,2.0)]
        with self.assertRaises(ValueError):
            a.parameter_boxes(f['states'],f['varying'],h.__getitem__,f['owners'],f['lo'],f['hi'],46)
        records=f['witness'].copy();records[0]=replace(records[0],lower=-1.0)
        with self.assertRaises(ValueError):
            b.boxes(f['reference'],f['varying'],f['h'].__getitem__,f['owners'],
                    f['lo'],f['hi'],records,46)

    def test_missing_direct_witness_rejected(self):
        f=self.f
        with self.assertRaises(ValueError):
            a.parameter_boxes(f['states'],f['varying'][1:],f['h'].__getitem__,
                              f['owners'],f['lo'],f['hi'],46)

    def test_row_support_budget_and_negative_constants_fail(self):
        with self.assertRaises(ValueError):
            a.compose(0,[(75+j,1) for j in range(11)],0,1,self.f['states'])
        with self.assertRaises(ValueError):
            a.compose(0,[(0,1),(75,1),(150,1)],0,1,self.f['states'])
        with self.assertRaises(ValueError):
            list(a.stream([True],[False],lambda i:[] ,[-1],[1],self.f['states'],{},1))

    def test_constant_and_fixed_rows_are_audited(self):
        f=self.f;terms=[[],[(71,1.0)],[(146,1.0)],[(221,1.0)]]
        keep=[True,True,True,False];soft=[False,False,True,False]
        producer=a.stream(keep,soft,terms.__getitem__,[1]*4,[1]*4,f['states'],f['box'],2)
        encoded=list(b.stream(producer,keep,soft,terms.__getitem__,[1]*4,[1]*4,
                              f['reference'],f['refbox'],2))
        self.assertEqual([r.row_id for r in encoded],[0,1])
        self.assertTrue(all(not r.columns for r in encoded))

    def test_missing_duplicate_extra_stream_records_rejected(self):
        f=self.f;terms=[[(146,1.0)],[(221,-1.0)]]
        args=([True]*2,[False]*2,terms.__getitem__,[1]*2,[1]*2)
        records=list(a.stream(*args,f['states'],f['box'],2))
        for corrupted in [records[:1],[records[0],records[0]],records+[records[0]]]:
            with self.assertRaises(ValueError):
                list(b.stream(corrupted,*args,f['reference'],f['refbox'],2))

    def test_numeric_coefficient_and_rhs_tampering(self):
        f=self.f;t=[(146,1.0)];r=a.compose(0,t,0.1,0.012,f['states']);e=a.encode(r,f['box'])
        altered=replace(e,values=(e.values[0]+1.0,)+e.values[1:])
        for ee in [altered,replace(e,rhs=math.nextafter(e.rhs,math.inf))]:
            with self.assertRaises(ValueError):
                b.row(0,t,0.1,0.012,f['reference'],f['refbox'],r,ee)

    def test_inward_encoding_implication_over_rational_box_grid(self):
        # One physical direction gives an exhaustively checked finite box grid.
        st=a.State((D(0),)*75,({0:D(3,-3)},)+({},)*74,(D(1,-4),)*75)
        r=a.compose(0,[(0,0.1)],0.01,0.012,[st])
        box={0:(D(-1),D(1)),1:(ZERO,D(1,-1))}
        e=a.encode(r,box,1)
        accepted=0
        for x,eta in itertools.product([F(i,20) for i in range(-20,21)],
                                       [F(i,20) for i in range(11)]):
            v={0:x,1:eta}
            if sum(F(c)*v[j] for j,c in zip(e.columns,e.values))<=F(e.rhs):
                accepted+=1
                self.assertGreaterEqual(b.q(r.intercept)+b.q(r.tangent[0])*x-b.q(r.kappa)*eta,0)
        self.assertGreater(accepted,0)

    def test_subnormal_coefficient_and_even_tie(self):
        st=a.State((ZERO,)*75,({0:D(1,-1074)},)+({},)*74,(ZERO,)*75)
        r=a.compose(0,[(0,0.5)],1.0,1.0,[st]);e=a.encode(r,{0:(D(-1),ONE)})
        self.assertEqual(e.values,(-0.0,))
        b.nearest(e.values[0],-F(1,2**1075))
        self.assertLess(F(e.rhs),1)


class ReturnTests(unittest.TestCase):
    def setUp(self):self.f=fixture()

    def test_nonlinear_and_encoding_error_in_same_reserve(self):
        f=self.f;dx=[[0.0]*24 for _ in range(3)]
        dx[1][2]=2**-10
        pred=list(returned.predictions(f['states'],[r[2] for r in f['raw']],dx))
        actual=[[float(b.q(v)) for v in row] for row in pred]
        actual[1][71]+=2**-14
        eta=D(1,-8)
        err=returned.feature_errors({146},f['states'],pred,actual,eta)
        self.assertEqual(err[146],D(1,-14))
        rows=list(returned.rows([True],[False],lambda i:[(146,1.0)],[1.0],
                                f['states'],pred,actual,eta,1))
        self.assertEqual(rows[0][0],0)
        self.assertGreaterEqual(rows[0][2],rows[0][1])

    def test_outside_nonlinear_reserve_stops(self):
        f=self.f;pred=[s.base for s in f['states']]
        actual=[[float(b.q(v)) for v in row] for row in pred];actual[1][71]+=0.5
        with self.assertRaises(ValueError):
            list(returned.rows([True],[False],lambda i:[(146,1.0)],[1.0],
                               f['states'],pred,actual,D(1,-8),1))

    def test_fixed_state_one_ulp_error_fails(self):
        f=self.f;pred=[s.base for s in f['states']]
        actual=[[float(b.q(v)) for v in row] for row in pred]
        actual[0][71]=math.nextafter(actual[0][71],math.inf)
        with self.assertRaises(ValueError):
            returned.feature_errors({71},f['states'],pred,actual,D(1,-2))

    def test_prediction_from_H_trace_and_immutable_error(self):
        f=self.f;dx=[[0.0]*24 for _ in range(3)];dx[1][16]=0.125;dx[1][23]=-0.125
        pred=list(returned.predictions(f['states'],[r[2] for r in f['raw']],dx))
        expected=F(f['raw'][1][1][31])+F(0.125)*(F(f['raw'][1][2][31][16])-F(f['raw'][1][2][31][23]))
        self.assertEqual(b.q(pred[1][55]),expected)
        dx[0][0]=math.ulp(0.0)
        with self.assertRaises(ValueError):
            list(returned.predictions(f['states'],[r[2] for r in f['raw']],dx))


if __name__=='__main__':unittest.main()
