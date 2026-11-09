import unittest
from dataclasses import replace
from fractions import Fraction as F
from types import SimpleNamespace
import numpy as np
import adapter
import assemble
import audit
import returned
from dyadic import D, ZERO
from test_stream import fixture


def csr(rows,ncols):
    indices=[];values=[];ptr=[0]
    for row in rows:
        for j,a in row:indices.append(j);values.append(a)
        ptr.append(len(values))
    return SimpleNamespace(shape=(len(rows),ncols),indptr=np.array(ptr,dtype=np.int64),
                           indices=np.array(indices,dtype=np.int32),data=np.array(values,dtype=np.float64))


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.f=fixture()
        f=self.f
        self.layout=adapter.Layout(3,2,46,4,2,48,48)
        self.z=dict(features=np.array([r[0]+[999.0]*51 for r in f['raw']]),
                    actualY=np.array([r[1] for r in f['raw']]),J=np.array([r[2] for r in f['raw']]),
                    feature_scale=np.full((3,75),0.125),sigma=np.full((3,51),0.25),
                    active=np.array([1,2]))
        self.P=csr([row for r in f['raw'] for row in r[3]+[[]]*51],46)

    def test_actual_outputs_are_not_auxiliary_features(self):
        raw,states=adapter.make_states(self.z,[True,False,False],self.P,self.layout)
        self.assertEqual(states,self.f['states'])
        self.assertEqual(states[1].base[71],D.of(self.z['actualY'][1,47].item()))
        self.assertNotEqual(states[1].base[71],D(999))

    def test_wrong_active_set_precision_and_layout_rejected(self):
        for key,value in [('active',np.array([0,1])),('actualY',self.z['actualY'].astype(np.float32))]:
            z={**self.z,key:value}
            with self.assertRaises(ValueError):adapter.make_states(z,[True,False,False],self.P,self.layout)
        with self.assertRaises(ValueError):adapter.make_states(self.z,[True,False,False],self.P)

    def test_full_mask_preserves_constants(self):
        M=csr([[],[(71,1.0)],[(146,1.0)],[]],225)
        row=dict(constant=np.ones(4),scale=np.ones(4),keep=np.array([True,True,True,False]),
                 soft=np.array([False,False,True,False]))
        get=adapter.bind_rows(row,M,self.layout)
        self.assertEqual(get(0),[])
        row['keep'][1]=False
        with self.assertRaises(ValueError):adapter.bind_rows(row,M,self.layout)

    def test_numpy_scalars_are_adapted_without_value_change(self):
        M=csr([[],[(71,1.0)],[(146,1.0)],[]],225)
        row=dict(constant=np.ones(4),scale=np.full(4,0.012),keep=np.array([True,True,True,False]),
                 soft=np.array([False,False,True,False]))
        r=adapter.row_inputs(row,M,self.layout)
        records=list(assemble.stream(**r,states=self.f['states'],boxes=self.f['box'],expected_hard_rows=2))
        self.assertEqual(len(records),2)
        self.assertEqual(audit.q(records[0][0].scale),F(float(row['scale'][0])))
        actual=adapter.finite_features(self.z['features'],self.z['actualY'],self.layout)
        self.assertEqual(actual[1][71],float(self.z['actualY'][1,47]))
        self.assertTrue(all(type(v) is float for state in actual for v in state))

    def test_exact_interval_adapter_rejects_changed_trust(self):
        f=self.f;H=csr(f['h'],46)
        im=dict(owners=f['owners'],closed_lower=[[-1,8]]*48,closed_upper=[[1,8]]*48,trust=1)
        lo,hi=adapter.interval_values(im,H,self.layout)
        self.assertEqual(lo,[D(-1,-3)]*48)
        self.assertEqual(hi,[D(1,-3)]*48)
        im['trust']=2
        with self.assertRaises(ValueError):adapter.interval_values(im,H,self.layout)

    def test_physical_trace_binding_and_pivot(self):
        f=self.f;trace=[[0,1] for _ in range(48)];trace[16]=[1,1024];trace[23]=[-1,1024]
        dx=adapter.physical_increments(f['states'],f['owners'],f['h'].__getitem__,trace)
        self.assertEqual(dx[1][16],D(1,-10));self.assertEqual(dx[1][23],D(-1,-10))
        self.assertEqual(dx[0],(ZERO,)*24)
        bad=f['h'].copy();bad[23]=[(16,1.0)]
        with self.assertRaises(ValueError):
            adapter.physical_increments(f['states'],f['owners'],bad.__getitem__,trace)

    def test_independent_posterior_audit_and_tampering(self):
        f=self.f;trace=[[0,1] for _ in range(48)];trace[2]=[1,1024]
        dx=adapter.physical_increments(f['states'],f['owners'],f['h'].__getitem__,trace)
        pred=list(returned.predictions(f['states'],[r[2] for r in f['raw']],dx))
        actual=[[float(audit.q(v)) for v in row] for row in pred]
        actual[1][71]+=2**-14
        terms=[[(146,1.0)],[(221,-1.0)],[]]
        eta=D(1,-8)
        rec=list(returned.rows([True]*3,[False]*3,terms.__getitem__,[1.0]*3,
                               f['states'],pred,actual,eta,3))
        args=(f['raw'],f['owners'],f['h'].__getitem__,trace,[True]*3,[False]*3,
              terms.__getitem__,[1.0]*3,actual,eta,3)
        self.assertTrue(audit.returned(rec,*args))
        bad=rec.copy();bad[0]=(0,rec[0][1]+D(1,-30),rec[0][2])
        with self.assertRaises(ValueError):audit.returned(bad,*args)
        with self.assertRaises(ValueError):audit.returned(rec[:-1],*args)
        actual[1][71]+=0.5
        with self.assertRaises(ValueError):audit.returned(rec,*args)


if __name__=='__main__':unittest.main()
