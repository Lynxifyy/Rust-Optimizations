#!/usr/bin/env python3
"""Trace one register through a function: its writes, and stores of it to a base.

`x20` is reused for every temporary in a big constructor, so the only reliable
way to find "where did *this* allocation get stored" is to walk the function in
address order and pair each `mov xN, x0` (right after an `operator new`) with the
next `str xN, [base, #off]`.

Usage:
    python3 tools/analysis/trace_reg.py <so> <fn_start> <fn_len_bytes> x20 x19
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import disasm  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("so", type=Path)
    ap.add_argument("fn_start", type=lambda s: int(s, 0))
    ap.add_argument("fn_len", type=lambda s: int(s, 0))
    ap.add_argument("reg", help="register to trace, e.g. x20")
    ap.add_argument("base", help="base register of interest, e.g. x19")
    args = ap.parse_args(argv)

    img = disasm.Image(str(args.so))
    insns = disasm.disasm(img, args.fn_start, args.fn_len // 4)
    reg = args.reg
    base = args.base

    last_write = None
    for ins in insns:
        op = ins.op_str
        # Writes to the traced register: mov/sub/add/ldr with it as destination.
        dest_is_reg = op.startswith(reg + ",") or op.startswith(reg + " ")
        if ins.mnemonic in ("mov", "ldr", "adrp", "add", "sub", "orr", "ldp", "movz", "movk", "csel") and dest_is_reg:
            note = ""
            if ins.mnemonic == "mov" and op.split(",")[1].strip() == "x0":
                note = "  <- likely a new object (result of a call)"
            print(f"{ins.addr:#x}: {ins.mnemonic} {op}{note}")
            last_write = ins.addr
        # Stores of the traced register to the base register.
        if ins.mnemonic in ("str", "stp") and reg in op and base in op and "[" in op:
            print(f"{ins.addr:#x}: {ins.mnemonic} {op}   <== stores {reg} into {base}"
                  f" (last write to {reg}: {last_write:#x})" if last_write else "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
