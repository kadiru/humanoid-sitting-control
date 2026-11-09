import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from recording import write

class RecordingTests(unittest.TestCase):
    def test_nested_numpy_audit_and_false_verdict(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'audit.json'
            write(p,dict(passed=np.bool_(False),tests={'ok':np.bool_(True)},x=np.array([1.,2.]),i=np.int64(3)))
            r=json.loads(p.read_text());self.assertIs(r['passed'],False);self.assertIs(r['tests']['ok'],True)
            self.assertEqual(r['x'],[1.,2.]);self.assertEqual(r['i'],3)
            with self.assertRaises(FileExistsError):write(p,{'passed':True})
    def test_bad_value_cannot_leave_partial_audit(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'audit.json'
            with self.assertRaises(ValueError):write(p,dict(x=np.float64('nan')))
            self.assertFalse(p.exists())
            with self.assertRaises(TypeError):write(p,dict(x=object()))
            self.assertFalse(p.exists())

if __name__=='__main__':unittest.main(verbosity=2)
