import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["DCT2d", "IDCT2d"]


# Types

Tensor = torch.Tensor


# Checking

def _is_a_batched_tensor(x: Tensor) -> bool:
    return True if x.ndim == 4 else False


def _is_a_transformed_tensor(x: Tensor) -> bool:
    return True if x.ndim == 5 else False


def _is_a_resolution_divisible(x: Tensor, d: int) -> bool:
    return True if x.size(2) % d == 0 and x.size(3) % d == 0 else False


# Helper Functions

def _initialize_dct_kernel(d: int, k: int, device: str | torch.device = None) -> Tensor:
    return torch.tensor(scipy.fft.dct(np.eye(k), d, orthogonalize=True), dtype=torch.float32, device=device)


def _initialize_inverse_dct_kernel(d: int, k: int, device: str | torch.device = None) -> Tensor:
    return torch.tensor(scipy.fft.idct(np.eye(k), d, orthogonalize=True), dtype=torch.float32, device=device)


def _expand_kernel(kernel: Tensor, dims: int) -> Tensor:
    for _ in range(dims - 1):
        kernel = kernel.kron(kernel)
    return kernel


def _zigzag_permutation(k: int, device: str | torch.device = None) -> Tensor:
    idx = torch.arange(0, k ** 2, device=device, dtype=torch.int64).reshape(k, k).flipud()
    return torch.cat([idx.diagonal(i) if (i + k) % 2 == 1 else idx.diagonal(i).flip(0) for i in range(1 - k, k)])


# Modules

class _DCT(nn.Module):

    def __init__(self, dct: int, kernel_size: int, device: str | torch.device = None) -> None:
        super().__init__()
        self.dct = dct
        self.kernel_size = kernel_size
        self.register_buffer("kernel", _initialize_dct_kernel(dct, kernel_size, device).T)

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def extra_repr(self) -> str:
        return f"dct={self.dct}, kernel_size={self.kernel_size}"


class _IDCT(nn.Module):

    def __init__(self, dct: int, kernel_size: int, device: str | torch.device = None) -> None:
        super().__init__()
        self.dct = dct
        self.kernel_size = kernel_size
        self.register_buffer("kernel", _initialize_inverse_dct_kernel(dct, kernel_size, device))

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def extra_repr(self) -> str:
        return f"dct={self.dct}, kernel_size={self.kernel_size}"


class DCT2d(_DCT):

    def __init__(self, dct: int = 2, kernel_size: int = 8, device: str | torch.device = None) -> None:
        super().__init__(dct, kernel_size, device)
        k = _expand_kernel(self.kernel, 2).reshape(self.kernel_size ** 2, 1, self.kernel_size, self.kernel_size)
        k = k[_zigzag_permutation(self.kernel_size, k.device)]
        self.register_buffer("kernel", k)

    def forward(self, x: Tensor) -> Tensor:
        if not _is_a_batched_tensor(x):
            raise ValueError(f"{self.__class__.__name__} only supports batched image tensor. "
                             f"requires 4d tensor but {x.ndim}d tensor was given")

        if not _is_a_resolution_divisible(x, self.kernel_size):
            raise ValueError(f"the resolution must be divisible by the kernel size")

        b, c, h, w = x.shape

        x = x.reshape(b * c, 1, h, w)
        x = F.conv2d(x, self.kernel, None, self.kernel_size)
        x = x.reshape(b, c, -1, h // self.kernel_size, w // self.kernel_size)

        return x


class IDCT2d(_IDCT):

    def __init__(self, dct: int = 2, kernel_size: int = 8, device: str | torch.device = None) -> None:
        super().__init__(dct, kernel_size, device)
        k = _expand_kernel(self.kernel, 2).reshape(self.kernel_size ** 2, 1, self.kernel_size, self.kernel_size)
        k = k[_zigzag_permutation(self.kernel_size, k.device)]
        self.register_buffer("kernel", k)

    def forward(self, x: Tensor) -> Tensor:
        if not _is_a_transformed_tensor(x):
            raise ValueError(f"{self.__class__.__name__} only supports transformed image tensor by DCT2d. "
                             f"requires 5d tensor but {x.ndim}d tensor was given")

        b, c, _, h, w = x.shape

        x = x.reshape(b * c, -1, h, w)
        x = F.conv_transpose2d(x, self.kernel, None, self.kernel_size)
        x = x.reshape(b, c, h * self.kernel_size, w * self.kernel_size)

        return x
