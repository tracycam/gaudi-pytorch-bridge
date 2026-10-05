"""CPU-only semantic checks, run with the built bridge wheel installed."""
import collections
from dataclasses import dataclass, fields, is_dataclass
import unittest
from unittest.mock import patch

import torch
from habana_frameworks.torch.hpu import graphs


@dataclass
class Metadata:
    positions: object
    page: int


def legacy_hash(value, autocast):
    """Independent, frozen pre-optimization Python cache-key contract."""
    if isinstance(value, dict):
        return legacy_hash(tuple(value.items()), autocast)
    if isinstance(value, list) or (isinstance(value, tuple) and not isinstance(value, torch.Size)):
        return hash((tuple(legacy_hash(item, autocast) for item in value), autocast))
    if torch.is_tensor(value):
        return hash((value.dtype, value.shape, graphs._hpu_C.get_view_hash(value), autocast))
    if isinstance(value, collections.UserDict):
        return hash(tuple((key, tuple(legacy_hash(item, autocast) for item in items))
                          for key, items in value.items()))
    if is_dataclass(value):
        return hash(tuple((field.name, legacy_hash(getattr(value, field.name), autocast)) for field in fields(value)))
    return hash((value, autocast))


class InputTraversalTest(unittest.TestCase):
    def test_inference_keeps_identity_but_copies_distinct_aliases(self):
        with torch.inference_mode():
            persistent = torch.arange(8)
            destination = torch.zeros(4)
            source = torch.arange(4, dtype=torch.float32)
            # Distinct views are not an identity match, even on shared Storage.
            left, right = persistent[:4], persistent[4:]
            dst = ({'kv': [persistent] * 140}, Metadata(destination, 2), left)
            src = ({'kv': [persistent] * 140}, Metadata(source, 2), right)
            original = torch.Tensor.copy_
            calls = []
            def copy(d, s, **kwargs):
                calls.append((id(d), id(s)))
                return original(d, s, **kwargs)
            with patch.object(torch.Tensor, 'copy_', copy):
                graphs.copy_to(dst, src)
            self.assertEqual(len(calls), 2)
            torch.testing.assert_close(destination, source)
            torch.testing.assert_close(left, torch.arange(4, 8))

    def test_normal_mode_preserves_self_copy_version(self):
        tensor = torch.ones(4)
        version = tensor._version
        graphs.copy_to(tensor, tensor)
        self.assertEqual(tensor._version, version + 1)

    def test_keys_include_autocast_shape_view_and_dataclass_scalar(self):
        tensor = torch.ones(4)
        value = ({'kv': [tensor] * 140}, Metadata(tensor, 2), torch.Size([4]),
                 collections.UserDict({'items': [tensor, 7]}))
        with patch.object(graphs._hpu_C, 'get_view_hash', return_value=0):
            with patch.object(torch.hpu, 'is_autocast_hpu_enabled', return_value=False) as autocast:
                first = graphs.input_hash(value)
                self.assertEqual(autocast.call_count, 1)
            with patch.object(torch.hpu, 'is_autocast_hpu_enabled', return_value=True):
                self.assertNotEqual(first, graphs.input_hash(value))
            with patch.object(torch.hpu, 'is_autocast_hpu_enabled', return_value=False):
                self.assertNotEqual(first, graphs.input_hash((value[0], Metadata(tensor, 3), *value[2:])))
                self.assertNotEqual(graphs.input_hash(tensor), graphs.input_hash(tensor.reshape(2, 2)))
                self.assertNotEqual(graphs.input_hash(tensor), graphs.input_hash(tensor.double()))

    def test_native_hash_matches_python_key_protocol(self):
        tensors = [torch.zeros((n, 3), dtype=dtype) for n in (1, 4)
                   for dtype in (torch.float32, torch.bfloat16)]
        class CustomList(list):
            pass
        values = [tensors, tuple(tensors), {'a': tensors, 'b': (None, 17, True)},
                  Metadata(tensors[0], 3), collections.UserDict({'a': tensors}),
                  torch.Size([2, 3]), CustomList(tensors), torch.nn.Parameter(tensors[0])]
        for autocast in (False, True):
            for value in values:
                self.assertEqual(legacy_hash(value, autocast),
                    graphs._hpu_C.graph_input_hash(value, autocast, graphs._input_hash))

    def test_scalar_fast_keys_and_nested_metadata_match_frozen_protocol(self):
        scalar = [None, True, False, 0, 1, -1, 2**80, .5, -0., float('inf'), float('nan'), 'KV']
        values = scalar + [Metadata({'values': scalar}, 7), collections.UserDict({'values': scalar})]
        for autocast in (False, True):
            for value in values:
                self.assertEqual(legacy_hash(value, autocast),
                    graphs._hpu_C.graph_input_hash(value, autocast, graphs._input_hash))

    def test_recursive_input_raises_instead_of_exhausting_native_stack(self):
        value = []
        value.append(value)
        with self.assertRaises(RecursionError):
            graphs.input_hash(value)


if __name__ == '__main__':
    unittest.main()
