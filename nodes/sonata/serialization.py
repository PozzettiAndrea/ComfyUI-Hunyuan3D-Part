"""
Serialization Encoding
Pointcept detached version — merged from z_order, hilbert, and default modules.

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
from typing import Optional, Union


# ---------------------------------------------------------------------------
# Z-order encoding (from z_order.py)
# ---------------------------------------------------------------------------

# @lint-ignore-every LICENSELINT
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang

class KeyLUT:
    def __init__(self):
        r256 = torch.arange(256, dtype=torch.int64)
        r512 = torch.arange(512, dtype=torch.int64)
        zero = torch.zeros(256, dtype=torch.int64)
        device = torch.device("cpu")

        self._encode = {
            device: (
                self._xyz2key(r256, zero, zero, 8),
                self._xyz2key(zero, r256, zero, 8),
                self._xyz2key(zero, zero, r256, 8),
            )
        }
        self._decode = {device: self._key2xyz(r512, 9)}

    def encode_lut(self, device=torch.device("cpu")):
        if device not in self._encode:
            cpu = torch.device("cpu")
            self._encode[device] = tuple(e.to(device) for e in self._encode[cpu])
        return self._encode[device]

    def decode_lut(self, device=torch.device("cpu")):
        if device not in self._decode:
            cpu = torch.device("cpu")
            self._decode[device] = tuple(e.to(device) for e in self._decode[cpu])
        return self._decode[device]

    def _xyz2key(self, x, y, z, depth):
        key = torch.zeros_like(x)
        for i in range(depth):
            mask = 1 << i
            key = (
                key
                | ((x & mask) << (2 * i + 2))
                | ((y & mask) << (2 * i + 1))
                | ((z & mask) << (2 * i + 0))
            )
        return key

    def _key2xyz(self, key, depth):
        x = torch.zeros_like(key)
        y = torch.zeros_like(key)
        z = torch.zeros_like(key)
        for i in range(depth):
            x = x | ((key & (1 << (3 * i + 2))) >> (2 * i + 2))
            y = y | ((key & (1 << (3 * i + 1))) >> (2 * i + 1))
            z = z | ((key & (1 << (3 * i + 0))) >> (2 * i + 0))
        return x, y, z


_key_lut = KeyLUT()


def xyz2key(
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    b: Optional[Union[torch.Tensor, int]] = None,
    depth: int = 16,
):
    """Encodes x, y, z coordinates to shuffled keys via pre-computed LUTs."""
    EX, EY, EZ = _key_lut.encode_lut(x.device)
    x, y, z = x.long(), y.long(), z.long()

    mask = 255 if depth > 8 else (1 << depth) - 1
    key = EX[x & mask] | EY[y & mask] | EZ[z & mask]
    if depth > 8:
        mask = (1 << (depth - 8)) - 1
        key16 = EX[(x >> 8) & mask] | EY[(y >> 8) & mask] | EZ[(z >> 8) & mask]
        key = key16 << 24 | key

    if b is not None:
        b = b.long()
        key = b << 48 | key

    return key


def key2xyz(key: torch.Tensor, depth: int = 16):
    """Decodes shuffled key to x, y, z coordinates and batch index via pre-computed LUTs."""
    DX, DY, DZ = _key_lut.decode_lut(key.device)
    x, y, z = torch.zeros_like(key), torch.zeros_like(key), torch.zeros_like(key)

    b = key >> 48
    key = key & ((1 << 48) - 1)

    n = (depth + 2) // 3
    for i in range(n):
        k = key >> (i * 9) & 511
        x = x | (DX[k] << (i * 3))
        y = y | (DY[k] << (i * 3))
        z = z | (DZ[k] << (i * 3))

    return x, y, z, b


# ---------------------------------------------------------------------------
# Hilbert encoding (from hilbert.py)
# Modified from https://github.com/PrincetonLIPS/numpy-hilbert-curve
# Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com), Kaixin Xu
# ---------------------------------------------------------------------------

def _right_shift(binary, k=1, axis=-1):
    """Right shift an array of binary values."""
    if binary.shape[axis] <= k:
        return torch.zeros_like(binary)

    slicing = [slice(None)] * len(binary.shape)
    slicing[axis] = slice(None, -k)
    shifted = torch.nn.functional.pad(
        binary[tuple(slicing)], (k, 0), mode="constant", value=0
    )

    return shifted


def _binary2gray(binary, axis=-1):
    """Convert an array of binary values into Gray codes."""
    shifted = _right_shift(binary, axis=axis)
    gray = torch.logical_xor(binary, shifted)
    return gray


def _gray2binary(gray, axis=-1):
    """Convert an array of Gray codes back into binary values."""
    shift = 2 ** (torch.Tensor([gray.shape[axis]]).log2().ceil().int() - 1)
    while shift > 0:
        gray = torch.logical_xor(gray, _right_shift(gray, shift))
        shift = torch.div(shift, 2, rounding_mode="floor")
    return gray


def hilbert_encode_(locs, num_dims, num_bits):
    """Encode locations in a hypercube into a Hilbert integer.

    Vectorized version of the Hilbert curve implementation by John Skilling.
    """
    orig_shape = locs.shape
    bitpack_mask = 1 << torch.arange(0, 8).to(locs.device)
    bitpack_mask_rev = bitpack_mask.flip(-1)

    if orig_shape[-1] != num_dims:
        raise ValueError(
            "The last dimension was of size %d, but num_dims=%d. These need to be equal."
            % (orig_shape[-1], num_dims)
        )

    if num_dims * num_bits > 63:
        raise ValueError(
            "num_dims=%d and num_bits=%d for %d bits total, which can't be encoded into a int64."
            % (num_dims, num_bits, num_dims * num_bits)
        )

    locs_uint8 = locs.long().view(torch.uint8).reshape((-1, num_dims, 8)).flip(-1)

    gray = (
        locs_uint8.unsqueeze(-1)
        .bitwise_and(bitpack_mask_rev)
        .ne(0)
        .byte()
        .flatten(-2, -1)[..., -num_bits:]
    )

    for bit in range(0, num_bits):
        for dim in range(0, num_dims):
            mask = gray[:, dim, bit]
            gray[:, 0, bit + 1 :] = torch.logical_xor(
                gray[:, 0, bit + 1 :], mask[:, None]
            )
            to_flip = torch.logical_and(
                torch.logical_not(mask[:, None]).repeat(1, gray.shape[2] - bit - 1),
                torch.logical_xor(gray[:, 0, bit + 1 :], gray[:, dim, bit + 1 :]),
            )
            gray[:, dim, bit + 1 :] = torch.logical_xor(
                gray[:, dim, bit + 1 :], to_flip
            )
            gray[:, 0, bit + 1 :] = torch.logical_xor(gray[:, 0, bit + 1 :], to_flip)

    gray = gray.swapaxes(1, 2).reshape((-1, num_bits * num_dims))

    hh_bin = _gray2binary(gray)

    extra_dims = 64 - num_bits * num_dims
    padded = torch.nn.functional.pad(hh_bin, (extra_dims, 0), "constant", 0)

    hh_uint8 = (
        (padded.flip(-1).reshape((-1, 8, 8)) * bitpack_mask)
        .sum(2)
        .squeeze()
        .type(torch.uint8)
    )

    hh_uint64 = hh_uint8.view(torch.int64).squeeze()

    return hh_uint64


def hilbert_decode_(hilberts, num_dims, num_bits):
    """Decode Hilbert integers into locations in a hypercube."""
    if num_dims * num_bits > 64:
        raise ValueError(
            "num_dims=%d and num_bits=%d for %d bits total, which can't be encoded into a uint64."
            % (num_dims, num_bits)
        )

    hilberts = torch.atleast_1d(hilberts)

    orig_shape = hilberts.shape
    bitpack_mask = 2 ** torch.arange(0, 8).to(hilberts.device)
    bitpack_mask_rev = bitpack_mask.flip(-1)

    hh_uint8 = (
        hilberts.ravel().type(torch.int64).view(torch.uint8).reshape((-1, 8)).flip(-1)
    )

    hh_bits = (
        hh_uint8.unsqueeze(-1)
        .bitwise_and(bitpack_mask_rev)
        .ne(0)
        .byte()
        .flatten(-2, -1)[:, -num_dims * num_bits :]
    )

    gray = _binary2gray(hh_bits)

    gray = gray.reshape((-1, num_bits, num_dims)).swapaxes(1, 2)

    for bit in range(num_bits - 1, -1, -1):
        for dim in range(num_dims - 1, -1, -1):
            mask = gray[:, dim, bit]
            gray[:, 0, bit + 1 :] = torch.logical_xor(
                gray[:, 0, bit + 1 :], mask[:, None]
            )
            to_flip = torch.logical_and(
                torch.logical_not(mask[:, None]),
                torch.logical_xor(gray[:, 0, bit + 1 :], gray[:, dim, bit + 1 :]),
            )
            gray[:, dim, bit + 1 :] = torch.logical_xor(
                gray[:, dim, bit + 1 :], to_flip
            )
            gray[:, 0, bit + 1 :] = torch.logical_xor(gray[:, 0, bit + 1 :], to_flip)

    extra_dims = 64 - num_bits
    padded = torch.nn.functional.pad(gray, (extra_dims, 0), "constant", 0)

    locs_chopped = padded.flip(-1).reshape((-1, num_dims, 8, 8))

    locs_uint8 = (locs_chopped * bitpack_mask).sum(3).squeeze().type(torch.uint8)

    flat_locs = locs_uint8.view(torch.int64)

    return flat_locs.reshape((*orig_shape, num_dims))


# ---------------------------------------------------------------------------
# Public API (from default.py)
# ---------------------------------------------------------------------------

@torch.inference_mode()
def encode(grid_coord, batch=None, depth=16, order="z"):
    assert order in {"z", "z-trans", "hilbert", "hilbert-trans"}
    if order == "z":
        code = z_order_encode(grid_coord, depth=depth)
    elif order == "z-trans":
        code = z_order_encode(grid_coord[:, [1, 0, 2]], depth=depth)
    elif order == "hilbert":
        code = hilbert_encode(grid_coord, depth=depth)
    elif order == "hilbert-trans":
        code = hilbert_encode(grid_coord[:, [1, 0, 2]], depth=depth)
    else:
        raise NotImplementedError
    if batch is not None:
        batch = batch.long()
        code = batch << depth * 3 | code
    return code


@torch.inference_mode()
def decode(code, depth=16, order="z"):
    assert order in {"z", "hilbert"}
    batch = code >> depth * 3
    code = code & ((1 << depth * 3) - 1)
    if order == "z":
        grid_coord = z_order_decode(code, depth=depth)
    elif order == "hilbert":
        grid_coord = hilbert_decode(code, depth=depth)
    else:
        raise NotImplementedError
    return grid_coord, batch


def z_order_encode(grid_coord: torch.Tensor, depth: int = 16):
    x, y, z = grid_coord[:, 0].long(), grid_coord[:, 1].long(), grid_coord[:, 2].long()
    code = xyz2key(x, y, z, b=None, depth=depth)
    return code


def z_order_decode(code: torch.Tensor, depth):
    x, y, z = key2xyz(code, depth=depth)
    grid_coord = torch.stack([x, y, z], dim=-1)  # (N, 3)
    return grid_coord


def hilbert_encode(grid_coord: torch.Tensor, depth: int = 16):
    return hilbert_encode_(grid_coord, num_dims=3, num_bits=depth)


def hilbert_decode(code: torch.Tensor, depth: int = 16):
    return hilbert_decode_(code, num_dims=3, num_bits=depth)
