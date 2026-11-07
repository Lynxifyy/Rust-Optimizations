#!/usr/bin/env python3
"""Find functions that access a set of displacements off the same base register.

The signature of an object's field group (e.g. an Actor's position and rotation)
is one function touching several displacements on the *same* register. Stack
frames reuse the same displacements constantly (`[sp, #0x230]`), so this filters
out `sp`/`x29`-relative accesses and groups the rest by (function, base register).

It also reports whether each access is a SIMD/FP access (a float load), because a
position/rotation field is read as `ldr s0`/`ldr d0`, not `ldr x0`.

Usage:
    python3 tools/analysis/field_group.py <so> 0x230 0x238 --max-group 3
"""

from __future__ import annotations

import argparse
import struct
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import disasm  # noqa: E402
from scan_disp import decode_access, is_prologue  # noqa: E402

# Load/store: bit 26 is the V bit (1 = SIMD&FP). The scaled form is
# `size 111 V 01 opc imm12 Rn Rt`, so V is bit 26.
V_BIT = 1 << 26


def is_float_access(word: int) -> bool:
    return bool(word & V_BIT)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("so", type=Path)
    ap.add_argument("disps", nargs="+", type=lambda s: int(s, 0))
    ap.add_argument("--max-group", type=int, default=2,
                    help="only report groups touching at least this many distinct disps")
    ap.add_argument("--require-float", action="store_true",
                    help="only count SIMD/FP (float) accesses")
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--context", type=int, default=3)
    args = ap.parse_args(argv)

    img = disasm.Image(str(args.so))
    text = img.sec(".text")
    base = int(text.virtual_address)
    blob = img.data(".text")
    n = len(blob) // 4
    wanted = set(args.disps)

    # (function, base_reg) -> disp -> first (addr, mnemonic, float?)
    groups: dict[tuple[int, int], dict[int, tuple[int, str, bool]]] = defaultdict(dict)
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
        if disp not in wanted:
            continue
        if reg in (29, 31):  # fp / sp: stack-relative, never an object field
            continue
        is_float = is_float_access(word)
        if args.require_float and not is_float:
            continue
        slot = groups[(current_fn, reg)]
        if disp not in slot:
            slot[disp] = (addr, mnem, is_float)

    ranked = [
        (fn, reg, slot)
        for (fn, reg), slot in groups.items()
        if len(slot) >= args.max_group
    ]
    ranked.sort(key=lambda g: (-len(g[2]), g[0]))
    print(f"{len(ranked)} group(s) touching >= {args.max_group} of {[hex(d) for d in args.disps]}")

    for fn, reg, slot in ranked[: args.limit]:
        disps = sorted(slot)
        print(f"\n=== fn {fn:#x} base x{reg}: disps {[hex(d) for d in disps]} ===")
        for disp in disps:
            addr, mnem, is_float = slot[disp]
            kind = "float" if is_float else "int"
            print(f"  {addr:#x}: {mnem} [x{reg}, #{disp:#x}]  ({kind})")
        # Show the region around the first access for context.
        first = min(v[0] for v in slot.values())
        for ins in disasm.disasm(img, first - args.context * 4, args.context * 2 + 1):
            print(f"      {ins.addr:#x}: {ins.mnemonic} {ins.op_str}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
