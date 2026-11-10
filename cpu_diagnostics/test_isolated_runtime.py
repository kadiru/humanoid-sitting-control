"""GPU isolation must fail before any simulator/driver import."""
import unittest
from unittest.mock import patch
from isolated_runtime import require_gpu_isolation


class IsolationTests(unittest.TestCase):
    def check(self, *, container=True, nodes=(), env=None):
        if env is None:
            env={'CUDA_VISIBLE_DEVICES':'','NVIDIA_VISIBLE_DEVICES':'void'}
        with patch('isolated_runtime.Path.exists',return_value=container), \
             patch('isolated_runtime.glob.glob',return_value=list(nodes)), \
             patch.dict('os.environ',env,clear=True):
            require_gpu_isolation()

    def test_only_device_isolated_masked_container_is_allowed(self):
        self.check()

    def test_environment_mask_alone_is_insufficient(self):
        with self.assertRaises(RuntimeError):self.check(container=False)

    def test_each_gpu_device_family_is_rejected(self):
        for node in ['/dev/nvidia0','/dev/nvidiactl','/dev/dri/renderD128']:
            with self.subTest(node=node),self.assertRaises(RuntimeError):
                self.check(nodes=[node])

    def test_missing_or_visible_device_environment_is_rejected(self):
        for env in [{},{'CUDA_VISIBLE_DEVICES':'0','NVIDIA_VISIBLE_DEVICES':'void'},
                    {'CUDA_VISIBLE_DEVICES':'','NVIDIA_VISIBLE_DEVICES':'all'}]:
            with self.subTest(env=env),self.assertRaises(RuntimeError):self.check(env=env)


if __name__=='__main__':unittest.main()
