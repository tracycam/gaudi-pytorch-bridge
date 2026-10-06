# SPDX-License-Identifier: Apache-2.0
"""Reusable inference input storage using ordinary Torch allocation and copy."""

from collections import OrderedDict

import torch

__all__ = ["InputBufferCache"]


class InputBufferCache:
    """Reuse named input tensors across inference calls and shape buckets.

    This owns Tensor references, not a graph, allocator, stream or executor.
    The returned Tensor has contiguous storage and the source's shape/dtype.
    Copy ordering and in-flight storage lifetime remain Torch's responsibility.
    A repeated call with the same name and layout updates the same Tensor.

    Use one cache per independently scheduled input producer. Host sources of
    non-blocking copies must remain valid until their transfer has completed.
    Clear the associated graph cache when captured module state is replaced.
    Eviction releases this cache's reference; graphs can retain their own.

    Args:
        device: Destination device, normally ``"hpu"``.
        max_entries: Maximum retained name/shape/dtype buckets. At least one.
    """

    def __init__(self, device="hpu", *, max_entries=128):
        if type(max_entries) is not int or max_entries < 1:
            raise ValueError("max_entries must be a positive integer")
        self.device = torch.device(device)
        self.max_entries = max_entries
        self._buffers = OrderedDict()

    def copy(self, name, source, *, non_blocking=True):
        """Copy one input into reusable storage and return its destination.

        Different shapes/dtypes use distinct buckets; this is not broadcasting
        or quantization. Training and Tensor subclasses are intentionally not
        admitted by this inference utility. Unsupported copying itself follows
        ordinary ``Tensor.copy_`` behavior and propagates errors.
        """
        if type(name) is not str or not name:
            raise ValueError("input name must be a nonempty string")
        if type(source) is not torch.Tensor or source.layout != torch.strided:
            raise TypeError("input must be an ordinary strided Tensor")
        if torch.is_grad_enabled():
            raise RuntimeError("InputBufferCache requires no_grad or inference_mode")
        if type(non_blocking) is not bool:
            raise TypeError("non_blocking must be bool")
        key = (name, tuple(source.shape), source.dtype)
        destination = self._buffers.get(key)
        if destination is None:
            destination = torch.empty_like(
                source, device=self.device, memory_format=torch.contiguous_format
            )
            destination.copy_(source, non_blocking=non_blocking)
            # Publish a new bucket only after the ordinary copy was accepted.
            self._buffers[key] = destination
            if len(self._buffers) > self.max_entries:
                self._buffers.popitem(last=False)
        else:
            if (destination.shape != source.shape or destination.dtype != source.dtype
                    or destination.device.type != self.device.type
                    or (self.device.index is not None and destination.device.index != self.device.index)
                    or not destination.is_contiguous()):
                raise RuntimeError("cached input layout was modified; clear the cache before rebinding")
            destination.copy_(source, non_blocking=non_blocking)
            self._buffers.move_to_end(key)
        return destination

    def clear(self):
        """Release owned references; this does not synchronize device work."""
        self._buffers.clear()

    def __len__(self):
        return len(self._buffers)
