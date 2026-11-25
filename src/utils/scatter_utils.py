"""Centralized torch_scatter import with pure-PyTorch fallback when .so is missing or ABI-incompatible."""

import torch

try:
    from torch_scatter import scatter
except (ImportError, OSError):
    def scatter(src, index, dim=0, reduce="mean"):
        if reduce != "mean":
            raise NotImplementedError("Fallback only supports reduce='mean'")
        size = int(index.max().item()) + 1
        if src.dim() == 1:
            out = torch.zeros(size, device=src.device, dtype=src.dtype)
            out.scatter_add_(dim, index, src)
            cnt = torch.zeros(size, device=src.device, dtype=src.dtype)
            cnt.scatter_add_(dim, index, torch.ones_like(src))
            return out / cnt.clamp(min=1)
        out = torch.zeros(size, src.size(1), device=src.device, dtype=src.dtype)
        out.scatter_add_(dim, index.unsqueeze(1).expand_as(src), src)
        cnt = torch.zeros(size, src.size(1), device=src.device, dtype=src.dtype)
        cnt.scatter_add_(dim, index.unsqueeze(1).expand_as(src), torch.ones_like(src))
        return out / cnt.clamp(min=1)

try:
    from torch_scatter import segment_coo, segment_csr
except (ImportError, OSError):
    def segment_coo(src, index, dim_size=None, dim=0, reduce="sum"):
        if reduce != "sum":
            raise NotImplementedError("Fallback only supports reduce='sum'")
        if dim_size is None:
            dim_size = int(index.max().item()) + 1
        shape = [dim_size] + list(src.shape[1:])
        out = torch.zeros(shape, device=src.device, dtype=src.dtype)
        idx = index.view(-1, *(1,) * (src.dim() - 1)).expand_as(src)
        out.scatter_add_(dim, idx, src)
        return out

    def segment_csr(src, indptr, dim=0, reduce="sum"):
        if reduce != "sum":
            raise NotImplementedError("Fallback only supports reduce='sum'")
        num_segments = indptr.size(0) - 1
        if src.dim() == 1:
            out = torch.zeros(num_segments, device=src.device, dtype=src.dtype)
            for i in range(num_segments):
                out[i] = src[indptr[i] : indptr[i + 1]].sum()
            return out
        out = torch.zeros(num_segments, *src.shape[1:], device=src.device, dtype=src.dtype)
        for i in range(num_segments):
            out[i] = src[indptr[i] : indptr[i + 1]].sum(dim=0)
        return out
