"""jpeg.py: JPEG decode / encode on the GPU where nvjpeg works, on the CPU (threads) where it does not.

On a vast.ai RTX 5090 box (torchvision 0.29.1+cu132) every GPU decode failed: "CUDAJpegDecoder ... Failed to initialize
nvjpeg with hardware backend: 6". There 64 page images decoded on the CPU in 16 threads at 1.7 ms each.
The first failure switches the process to the CPU for good (printed once)."""
from concurrent.futures import ThreadPoolExecutor

import torch
from torchvision.io import decode_jpeg, encode_jpeg

_GPU = {'decode': True, 'encode': True}
_POOL = []


def _pool():
    if not _POOL:
        _POOL.append(ThreadPoolExecutor(16))
    return _POOL[0]


def _nvjpeg_failed(e):
    s = str(e)
    return 'nvjpeg' in s.lower() or 'JpegDecoder' in s or 'JpegEncoder' in s


def decode(data, device):
    """data: a uint8 byte tensor or a list of them -> C x H x W uint8 (or a list) on device. Callers wait for the GPU
    first (gen._sync: nvjpeg runs on its own stream)"""
    dev = torch.device(device)
    single = isinstance(data, torch.Tensor)
    lst = [data] if single else list(data)
    if dev.type == 'cuda' and _GPU['decode']:
        try:
            out = decode_jpeg(lst, device=dev)
            return out[0] if single else out
        except RuntimeError as e:
            if not _nvjpeg_failed(e):
                raise
            _GPU['decode'] = False
            print(f'GPU JPEG decode unavailable ({str(e)[-60:]}): decoding on the CPU from now on', flush=True)
    out = [t.to(dev) for t in _pool().map(decode_jpeg, lst)]
    return out[0] if single else out


def encode(u8, quality):
    """C x H x W uint8 (any device) -> the JPEG bytes on the CPU"""
    if u8.is_cuda and _GPU['encode']:
        try:
            return encode_jpeg(u8, quality=quality).cpu()
        except RuntimeError as e:
            if not _nvjpeg_failed(e):
                raise
            _GPU['encode'] = False
            print(f'GPU JPEG encode unavailable ({str(e)[-60:]}): encoding on the CPU from now on', flush=True)
    return encode_jpeg(u8.cpu(), quality=quality)


def encode_many(u8s, qualities):
    if u8s and u8s[0].is_cuda and _GPU['encode']:
        return [encode(u, q) for u, q in zip(u8s, qualities)]
    cpu = [u.cpu() for u in u8s]
    return list(_pool().map(lambda a: encode_jpeg(a[0], quality=a[1]), zip(cpu, qualities)))
