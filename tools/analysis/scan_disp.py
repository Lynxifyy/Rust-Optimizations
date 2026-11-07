#!/usr/bin/env python3
"""Sweep `.text` for load/store instructions at a given displacement.

Recovering a struct offset means finding the instruction that touches it, so this
scans every instruction in `.text` and reports the ones that load or store at one
of the requested byte displacements, together with the enclosing function.

The detection is a **raw word decoder**, not a disassembler. Two reasons:

- A disassembler's linear sweep stops at the first undecodable word, so over a
  213 MB `.text` it silently covers only the first few MB.
- Even a resuming sweep is slow and memory-hungry if it re-slices the tail on
  every resume (an O(n) copy per gap).

Decoding the handful of load/store encodings directly is a single pass, cannot
desync, and never copies. Capstone is used only to pretty-print the few hits.

Usage:
    python3 tools/analysis/scan_disp.py <so> 0x230 0x234 0x238 --limit 40
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import disasm  # noqa: E402

# Load/store with an unsigned 12-bit immediate, scaled by the access size.
#   size(2) 111 V=0 01 opc(2) imm12 Rn Rt
UNSIGNED_IMM = [
    (0b11, 0b01, 8, "ldr"),
    (0b11, 0b00, 8, "str"),
    (0b10, 0b01, 4, "ldr"),
    (0b10, 0b00, 4, "str"),
    (0b01, 0b01, 2, "ldrh"),
    (0b01, 0b00, 2, "strh"),
    (0b00, 0b01, 1, "ldrb"),
    (0b00, 0b00, 1, "strb"),
]

# Load/store unscaled (signed 9-bit immediate), LDUR/STUR.
UNSCALED_IMM = [
    (0b11, 0b01, "ldur"),
    (0b11, 0b00, "stur"),
    (0b10, 0b01, "ldur"),
    (0b10, 0b00, "stur"),
]


def _signed9(value: int) -> int:
    return value - 0x200 if value & 0x100 else value


def is_prologue(word: int) -> bool:
    """`stp x29, x30, [sp, #-imm]!` - the frame-setup prologue."""
    return (word & 0x0000FFFF) == 0x7BFD and (word & 0xFF000000) == 0xA9000000


def decode_access(word: int):
    """Returns (mnemonic, base_reg, byte_offset) for a load/store, else None."""
    for size, opc, mnem in UNSCALED_IMM:
        if (word >> 30) & 0b11 != size:
            continue
        if (word & 0x3B200000) != 0x38000000:
            continue
        if (word >> 22) & 0b11 != opc:
            continue
        if (word >> 10) & 0b11 != 0:
            continue
        imm9 = (word >> 12) & 0x1FF
        return mnem, (word >> 5) & 0x1F, _signed9(imm9)

    for size, opc, scale, mnem in UNSIGNED_IMM:
        if (word >> 30) & 0b11 != size:
            continue
        if (word & 0x3B000000) != 0x39000000:
            continue
        if (word >> 22) & 0b11 != opc:
            continue
        imm12 = (word >> 10) & 0xFFF
        return mnem, (word >> 5) & 0x1F, imm12 * scale

    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("so", type=Path)
    ap.add_argument("disps", nargs="+", type=lambda s: int(s, 0))
    ap.add_argument("--limit", type=int, default=40, help="max hits to print per disp")
    ap.add_argument("--context", type=int, default=6, help="instructions of context")
    args = ap.parse_args(argv)

    img = disasm.Image(str(args.so))
    text = img.sec(".text")
    base = int(text.virtual_address)
    blob = img.data(".text")
    n = len(blob) // 4
    wanted = set(args.disps)
    hits: dict[int, list[tuple[int, str, int, int]]] = {d: [] for d in wanted}

    current_fn = base
    for i in range(n):
        word = struct.unpack_from("<I", blob, i * 4)[0]
        addr = base + i * 4
        if is_prologue(word):
            current_fn = addr
        access = decode_access(word)
        if access is None:
            continue
        mnem, reg, disp = access
        if disp in wanted:
            hits[disp].append((addr, mnem, reg, current_fn))

    for disp in args.disps:
        found = hits[disp]
        print(f"\n=== disp {disp:#x} ({disp}): {len(found)} access(es) ===")
        for addr, mnem, reg, fn in found[: args.limit]:
            print(f"  {addr:#x} in fn {fn:#x}: {mnem} [x{reg}, #{disp:#x}]")
            start = addr - args.context * 4
            for ins in disasm.disasm(img, start, args.context * 2 + 1):
                mark = ">>" if ins.addr == addr else "  "
                print(f"      {mark} {ins.addr:#x}: {ins.mnemonic} {ins.op_str}")
        if len(found) > args.limit:
            print(f"  ... {len(found) - args.limit} more")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
