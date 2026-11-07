#!/usr/bin/env python3
"""Disassembly helpers for recovering offsets from libminecraftpe.so.

This is the scratch tool used to recover the entity-seam offsets by hand; it is
not part of the shipped pipeline (`tools/extract_offsets.py` is). It loads a
`.so` once and offers small, readable primitives: disassemble a function, follow
a `bl`, resolve a string reference, and list the memory accesses in a function.

Recovering an offset means reading the accessor that touches it: find the
function that returns a position, then read the `ldr` that loads the float. This
module is the minimum needed to do that without re-parsing the ELF each time.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import capstone
import lief

CS = capstone.Cs(capstone.CS_ARCH_ARM64, capstone.CS_MODE_LITTLE_ENDIAN)
CS.detail = True


class Image:
    """A read-only ELF view with relocation-aware reads and string lookup."""

    def __init__(self, path: str):
        self.b = lief.parse(path)
        if self.b is None:
            raise SystemExit(f"lief could not parse {path}")
        self.sections = {s.name: s for s in self.b.sections}
        self._cache: dict[str, bytes] = {}
        self.relocs: dict[int, int] = {}
        for r in self.b.relocations:
            if getattr(getattr(r, "type", None), "name", "") == "AARCH64_RELATIVE":
                self.relocs[int(r.address)] = int(r.addend)

    def sec(self, name: str):
        return self.sections.get(name)

    def data(self, name: str) -> bytes:
        if name not in self._cache:
            s = self.sec(name)
            self._cache[name] = bytes(s.content) if s else b""
        return self._cache[name]

    def vaddr_of(self, name: str) -> int:
        s = self.sec(name)
        return int(s.virtual_address) if s else 0

    def is_exec(self, v: int) -> bool:
        t = self.sec(".text")
        return t is not None and int(t.virtual_address) <= v < int(t.virtual_address) + t.size

    def read_u64(self, v: int):
        if v in self.relocs:
            return self.relocs[v]
        for s in self.b.sections:
            base = int(s.virtual_address)
            if base and base <= v < base + s.size:
                off = v - base
                d = self.data(s.name)
                if off + 8 <= len(d):
                    return struct.unpack_from("<Q", d, off)[0]
        return None

    def str_at(self, v: int):
        """The NUL-terminated string at a vaddr, or None."""
        for s in self.b.sections:
            base = int(s.virtual_address)
            if base and base <= v < base + s.size:
                off = v - base
                d = self.data(s.name)
                end = d.find(b"\x00", off)
                if off < end < off + 200:
                    try:
                        return d[off:end].decode("utf-8")
                    except UnicodeDecodeError:
                        return None
        return None

    def find_str(self, needle: str, section: str = ".rodata") -> int:
        d = self.data(section)
        i = d.find(needle.encode() + b"\x00")
        return self.vaddr_of(section) + i if i >= 0 else 0

    def find_strs(self, needle: str, section: str = ".rodata") -> list[int]:
        out: list[int] = []
        d = self.data(section)
        n = needle.encode() + b"\x00"
        start = 0
        while True:
            i = d.find(n, start)
            if i < 0:
                break
            out.append(self.vaddr_of(section) + i)
            start = i + 1
        return out


@dataclass
class Insn:
    addr: int
    mnemonic: str
    op_str: str
    raw: "capstone.CsInsn"


def disasm(img: Image, start: int, count: int = 200) -> list[Insn]:
    t = img.sec(".text")
    off = start - int(t.virtual_address)
    blob = img.data(".text")[off : off + count * 4]
    out: list[Insn] = []
    for i, ins in enumerate(CS.disasm(blob, start)):
        out.append(Insn(ins.address, ins.mnemonic, ins.op_str, ins))
        if i + 1 >= count:
            break
    return out


def call_target(ins: Insn):
    """The absolute target of a `bl`/`b`, or None for an indirect call."""
    if ins.mnemonic not in ("bl", "b"):
        return None
    for op in ins.raw.operands:
        if op.type == capstone.arm64.ARM64_OP_IMM:
            return op.imm
    return None


def string_refs(img: Image, insns: list[Insn]):
    """Yield (addr, string) for `adrp`/`add` pairs that form a `.rodata` string."""
    regs: dict[int, int] = {}
    for ins in insns:
        if ins.mnemonic == "adrp" and len(ins.raw.operands) == 2:
            regs[ins.raw.operands[0].reg] = ins.raw.operands[1].imm
        elif ins.mnemonic == "add" and len(ins.raw.operands) == 3:
            rd = ins.raw.operands[0].reg
            rn = ins.raw.operands[1].reg
            imm = ins.raw.operands[2].imm
            if rn in regs:
                target = regs[rn] + imm
                s = img.str_at(target)
                if s:
                    yield ins.addr, s


def mem_accesses(insns: list[Insn]):
    """Yield (addr, mnemonic, base_reg, displacement) for `[reg, #imm]` accesses."""
    for ins in insns:
        if not ins.mnemonic.startswith(("ldr", "str", "ldur", "stur", "ldrb", "ldrh", "ldrsb")):
            continue
        for op in ins.raw.operands:
            if op.type == capstone.arm64.ARM64_OP_MEM:
                yield ins.addr, ins.mnemonic, op.mem.base, op.mem.disp
                break
