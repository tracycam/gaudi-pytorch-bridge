"""CPU semantic checks for the model-independent input storage API."""
import importlib.util
from pathlib import Path
import unittest

import torch

# Also runnable before building the wheel; no HPU or framework model required.
path = Path(__file__).resolve().parents[1] / "python_packages/habana_frameworks/torch/hpu/input_buffers.py"
spec = importlib.util.spec_from_file_location("input_buffers_under_test", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
InputBufferCache = module.InputBufferCache


class InputBufferCacheTest(unittest.TestCase):
    @torch.inference_mode()
    def test_reuses_storage_and_separates_names_shapes_and_dtypes(self):
        cache = InputBufferCache("cpu")
        first = cache.copy("x", torch.arange(6).reshape(2, 3))
        self.assertIs(first, cache.copy("x", torch.full((2, 3), 7)))
        self.assertTrue(torch.equal(first, torch.full((2, 3), 7)))
        self.assertIsNot(first, cache.copy("residual", torch.ones((2, 3), dtype=first.dtype)))
        self.assertIsNot(first, cache.copy("x", torch.ones((3, 2), dtype=first.dtype)))
        self.assertIsNot(first, cache.copy("x", torch.ones((2, 3), dtype=torch.float32)))

    @torch.no_grad()
    def test_noncontiguous_and_alias_input_use_ordinary_copy(self):
        cache = InputBufferCache("cpu")
        source = torch.arange(12).reshape(3, 4).t()
        actual = cache.copy("x", source)
        self.assertTrue(actual.is_contiguous())
        self.assertTrue(torch.equal(actual, source))
        version = actual._version
        self.assertIs(actual, cache.copy("x", actual))
        self.assertEqual(actual._version, version + 1)

    @torch.inference_mode()
    def test_eviction_and_clear_do_not_destroy_external_owners(self):
        cache = InputBufferCache("cpu", max_entries=2)
        first = cache.copy("x", torch.ones(3))
        cache.copy("y", torch.ones(3))
        cache.copy("x", torch.full((3,), 2.))
        cache.copy("z", torch.ones(3))  # y is least recently used.
        self.assertIs(first, cache.copy("x", torch.ones(3)))
        cache.clear()
        self.assertEqual(len(cache), 0)
        self.assertTrue(torch.equal(first, torch.ones(3)))
        self.assertIsNot(first, cache.copy("x", torch.ones(3)))

    def test_rejection_occurs_before_copy_or_allocation(self):
        for value in (0, -1, True, "2"):
            with self.assertRaises(ValueError):
                InputBufferCache("cpu", max_entries=value)
        cache = InputBufferCache("cpu")
        with self.assertRaises(RuntimeError):
            cache.copy("x", torch.ones(3, requires_grad=True))
        with torch.no_grad():
            for name in (None, "", 0):
                with self.assertRaises(ValueError):
                    cache.copy(name, torch.ones(3))
            with self.assertRaises(TypeError):
                cache.copy("x", torch.nn.Parameter(torch.ones(3)))
            with self.assertRaises(TypeError):
                cache.copy("x", torch.ones(3), non_blocking=1)
        self.assertEqual(len(cache), 0)


if __name__ == "__main__":
    unittest.main()
