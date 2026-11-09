import math
import unittest
import assemble
import audit
import packed
from dyadic import D, ZERO
from test_stream import fixture


class PackedTests(unittest.TestCase):
    def setUp(self):
        self.f=fixture();self.terms=[[],[(146,1.0)],[(221,-1.0)]]
        self.rows=[assemble.compose(i,t,1.0,0.012,self.f['states']) for i,t in enumerate(self.terms)]

    def test_stream_buffer_is_bounded_and_complete(self):
        encoded=[assemble.encode(r,self.f['box']) for r in self.rows]
        checks=[]
        result=packed.collect(iter(encoded),3,47,3*46,lambda:checks.append(1))
        self.assertTrue(result['complete'])
        self.assertEqual(list(result['original_ids']),[0,1,2])
        self.assertEqual(result['indptr'][1],0)
        self.assertEqual(result['indptr'][-1],len(result['data']))
        self.assertGreaterEqual(len(checks),2)
        with self.assertRaises(ValueError):packed.collect(iter(encoded),3,47,0,lambda:None)
        with self.assertRaises(ValueError):packed.collect(iter(encoded[:-1]),3,47,138,lambda:None)

    def test_late_audit_error_prevents_complete_buffer(self):
        encoded=[assemble.encode(r,self.f['box']) for r in self.rows]
        def late_failure():
            yield from encoded
            raise ValueError('end-of-stream coverage audit failed')
        with self.assertRaises(ValueError):packed.collect(late_failure(),3,47,138,lambda:None)

    def test_shared_deadline_propagates(self):
        def deadline():raise TimeoutError('stage cap')
        with self.assertRaises(TimeoutError):packed.collect([],3,47,138,deadline)

    def test_explicit_boxes_preserve_other_variable_bounds(self):
        original=[[-math.inf,math.inf] for _ in range(46)]+[[0,0.5]]
        new=packed.explicit_bounds(original,self.f['witness'])
        self.assertEqual(new[-1],[0,0.5]);self.assertEqual(original[0],[-math.inf,math.inf])
        self.assertEqual(new[0],[-0.125,0.125])

    def test_exact_point_margin_and_independent_audit(self):
        f=self.f;point=[0.0]*46+[0.125]
        boxes={**f['box'],46:(ZERO,D(1,-1))}
        result=packed.exact_point(iter(self.rows),point,boxes,46,3,lambda:None)
        rb={**f['refbox'],46:(audit.F(0),audit.F(1,2))}
        self.assertTrue(audit.point([True]*3,[False]*3,self.terms.__getitem__,[1.0]*3,
                                    f['reference'],rb,point,46,result,3))
        point[0]=math.nextafter(0.125,math.inf)
        with self.assertRaises(ValueError):packed.exact_point(iter(self.rows),point,boxes,46,3,lambda:None)

    def test_tiny_negative_exact_margin_rejected(self):
        r=assemble.compose(0,[],0.0,1.0,self.f['states'])
        r=assemble.ExactRow(r.row_id,r.terms,D(-1,-1074),r.tangent,r.kappa,r.scale)
        with self.assertRaises(ValueError):packed.exact_point([r],[0.0]*47,{},None,1,lambda:None)


if __name__=='__main__':unittest.main()
