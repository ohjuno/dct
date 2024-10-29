# Discrete Cosine Transform

Implementation of "[Discrete Cosine Transform](https://www.cse.iitd.ac.in/~pkalra/col783-2017/DCT-Paper.pdf)", in PyTorch.


## Installation

### Binaries

1. Download [latest release]().
2. ```$ pip install dct-[version]-py3-none-any.whl```

### Build from Source

1. Clone this repository ```$ git clone ...```
2. Build with [Flit](https://flit.pypa.io/en/stable/)  ```flit build --format wheel```
3. ```$ pip install dist/dct-[version]-py3-non-any.whl```


## How to Use

### DCT2d & IDCT2d

#### Example

```python
import torch
import dct

dct2d = dct.DCT2d(
    dct=2,
    kernel_size=8,  # K
    device="cpu",
)

idct2d = dct.IDCT2d(
    dct=2,
    kernel_size=8,  # K
    device="cpu",
)

image1 = torch.randint(0, 255, (1, 3, 224, 224), dtype=torch.float32)  # [B, C, H, W], Input must be a batched tensor in BCHW format
image2 = dct2d(image1)  # (1, 3, 64, 28, 28) [B, C, K * K, H // K, W // K]
image3 = idct2d(image2)  # (1, 3, 224, 224) [B, C, H, W]

print(image1.equal(image3.round()))  # True
```

#### Parameters

- **dct: _{1, 2, 3, 4}_**  
    Type of the DCT. See [here](https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.dct.html#scipy.fft.dct) for more details.

- **kernel_size: _int_**  
    Size of the DCT matrix. The size of image should be divisible by `kernel_size`.

- **device: _str | torch.device, optional_**  
    The device on which the DCT matrix is or will be allocated.   
    It is expected to be the same device as the input.