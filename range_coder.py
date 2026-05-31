from __future__ import annotations

import bisect
from pathlib import Path
from typing import BinaryIO


class BitOutputStream:
    def __init__(self, out: BinaryIO) -> None:
        self.out = out
        self.current_byte = 0
        self.num_bits_filled = 0
        self.bytes_written = 0

    def write(self, bit: int) -> None:
        self.current_byte = (self.current_byte << 1) | (int(bit) & 1)
        self.num_bits_filled += 1
        if self.num_bits_filled == 8:
            self.out.write(bytes((self.current_byte,)))
            self.bytes_written += 1
            self.current_byte = 0
            self.num_bits_filled = 0

    def flush(self) -> None:
        if self.num_bits_filled:
            self.current_byte <<= 8 - self.num_bits_filled
            self.out.write(bytes((self.current_byte,)))
            self.bytes_written += 1
            self.current_byte = 0
            self.num_bits_filled = 0


class BitInputStream:
    def __init__(self, inp: BinaryIO) -> None:
        self.inp = inp
        self.current_byte = 0
        self.num_bits_remaining = 0

    def read(self) -> int:
        if self.num_bits_remaining == 0:
            data = self.inp.read(1)
            if len(data) == 0:
                return 0
            self.current_byte = data[0]
            self.num_bits_remaining = 8
        self.num_bits_remaining -= 1
        return (self.current_byte >> self.num_bits_remaining) & 1


class ArithmeticCoderBase:
    def __init__(self, num_state_bits: int = 32) -> None:
        if num_state_bits < 1:
            raise ValueError("state size must be positive")
        self.num_state_bits = num_state_bits
        self.full_range = 1 << num_state_bits
        self.half_range = self.full_range >> 1
        self.quarter_range = self.half_range >> 1
        self.minimum_range = self.quarter_range + 2
        self.maximum_total = self.minimum_range
        self.state_mask = self.full_range - 1
        self.low = 0
        self.high = self.state_mask

    def update(self, cumulative: list[int], symbol: int) -> None:
        total = cumulative[-1]
        if total > self.maximum_total:
            raise ValueError(f"frequency total {total} exceeds maximum {self.maximum_total}")
        sym_low = cumulative[symbol]
        sym_high = cumulative[symbol + 1]
        if sym_low == sym_high:
            raise ValueError("symbol has zero frequency")

        range_size = self.high - self.low + 1
        new_low = self.low + sym_low * range_size // total
        new_high = self.low + sym_high * range_size // total - 1
        self.low = new_low
        self.high = new_high

        while ((self.low ^ self.high) & self.half_range) == 0:
            self.shift()
            self.low = ((self.low << 1) & self.state_mask)
            self.high = ((self.high << 1) & self.state_mask) | 1
        while (self.low & ~self.high & self.quarter_range) != 0:
            self.underflow()
            self.low = ((self.low << 1) ^ self.half_range) & self.state_mask
            self.high = (((self.high ^ self.half_range) << 1) | self.half_range | 1) & self.state_mask

    def shift(self) -> None:
        raise NotImplementedError()

    def underflow(self) -> None:
        raise NotImplementedError()


class ArithmeticEncoder(ArithmeticCoderBase):
    def __init__(self, bitout: BitOutputStream, num_state_bits: int = 32) -> None:
        super().__init__(num_state_bits)
        self.output = bitout
        self.num_underflow = 0

    def write(self, cumulative: list[int], symbol: int) -> None:
        self.update(cumulative, symbol)

    def finish(self) -> None:
        self.num_underflow += 1
        if self.low < self.quarter_range:
            self._write_bit_with_pending(0)
        else:
            self._write_bit_with_pending(1)
        self.output.flush()

    def shift(self) -> None:
        bit = self.low >> (self.num_state_bits - 1)
        self._write_bit_with_pending(bit)

    def underflow(self) -> None:
        self.num_underflow += 1

    def _write_bit_with_pending(self, bit: int) -> None:
        self.output.write(bit)
        while self.num_underflow > 0:
            self.output.write(bit ^ 1)
            self.num_underflow -= 1


class ArithmeticDecoder(ArithmeticCoderBase):
    def __init__(self, bitin: BitInputStream, num_state_bits: int = 32) -> None:
        super().__init__(num_state_bits)
        self.input = bitin
        self.code = 0
        for _ in range(num_state_bits):
            self.code = (self.code << 1) | self.input.read()

    def read(self, cumulative: list[int]) -> int:
        total = cumulative[-1]
        range_size = self.high - self.low + 1
        offset = self.code - self.low
        value = ((offset + 1) * total - 1) // range_size
        symbol = bisect.bisect_right(cumulative, value) - 1
        self.update(cumulative, symbol)
        return symbol

    def shift(self) -> None:
        self.code = ((self.code << 1) & self.state_mask) | self.input.read()

    def underflow(self) -> None:
        self.code = ((self.code & self.half_range) | ((self.code << 1) & (self.state_mask >> 1)) | self.input.read())


def open_bit_output(path: str | Path) -> tuple[BinaryIO, BitOutputStream]:
    raw = Path(path).open("wb")
    return raw, BitOutputStream(raw)
