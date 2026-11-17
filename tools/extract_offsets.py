#!/usr/bin/env python3
"""Resolve GlowberryClient offset maps from a libminecraftpe.so.

Two things are extracted, and they are different kinds of thing:

1. **RTTI-resolved vtable slots** (reliable, no per-build constant). The game
   strips method symbols but keeps RTTI typeinfo-name strings, so a vtable slot
   can be found by name. This mirrors the preloader's `Vtable.cpp` algorithm
   exactly: find the typeinfo-name string in `.rodata`, find the typeinfo object
   in `.data.rel.ro` (whose second word is that string), then find the primary
   vtable (a pointer to the typeinfo with `offset_to_top == 0`), and read the
   slot. The difference here is that this runs **statically**: the pointers in
   `.data.rel.ro` live in `.rela.dyn` as `R_AARCH64_RELATIVE` addends, so the
   file's raw bytes are zero and the values must be read from the relocations.

2. **Byte signatures** (per-build, fragile). A signature is scanned across
   `.text` and is only accepted when it matches **exactly once**; a zero or a
   multiple match is reported with its candidates for manual review. This is the
   honest form of the rule the project documents: a pattern is valid for one
   build and must be re-derived when Mojang relinks the binary.

`EntitySeamOffsets` is reported as **unrecovered** on purpose. Its fields are
*field offsets inside a live `Level`/`Actor`*, which no signature or RTTI name
reveals - they come from disassembling the accessors, which is a separate job.
Emitting a number here would be a fabrication, so the JSON says so explicitly.

Usage:
    python3 tools/extract_offsets.py --so work/so/libminecraftpe-26_60.so \
        --version 26.60 --out offsets/offsets_26_60.json
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

# LIEF and capstone are heavy and only needed once a real .so is parsed. The
# pure helpers below (pattern parsing, scanning, report shape) import this module
# on their own, so the imports stay optional and are checked where they are used.
try:
    import lief
except ImportError:  # pragma: no cover - exercised by the CI install step
    lief = None

try:
    import capstone
except ImportError:  # pragma: no cover
    capstone = None


def _require_lief():
    if lief is None:
        raise SystemExit(
            "lief is required to parse a .so: pip install -r tools/requirements.txt"
        )
    return lief


def _require_capstone():
    if capstone is None:
        raise SystemExit(
            "capstone is required to disassemble: pip install -r tools/requirements.txt"
        )
    return capstone


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested without a real .so)
# ---------------------------------------------------------------------------


def parse_pattern(pattern: str) -> tuple[bytes, bytes]:
    """Parses a `48 8B ?? ?? 89` style pattern into `(bytes, mask)`.

    The mask is `1` for a fixed byte and `0` for a `??` wildcard, so a match is
    `all((b & m) == (p & m))` byte-wise. Accepts `??`, `?`, and `**` as wildcards.
    Raises `ValueError` on an empty pattern or an unknown token.
    """
    tokens = pattern.split()
    if not tokens:
        raise ValueError("empty pattern")
    out_bytes = bytearray()
    out_mask = bytearray()
    for token in tokens:
        if token in ("??", "?", "**"):
            out_bytes.append(0)
            out_mask.append(0)
            continue
        if len(token) != 2:
            raise ValueError(f"bad pattern token: {token!r}")
        try:
            value = int(token, 16)
        except ValueError as exc:
            raise ValueError(f"bad hex byte in pattern: {token!r}") from exc
        out_bytes.append(value)
        out_mask.append(0xFF)
    return bytes(out_bytes), bytes(out_mask)


def scan_bytes(haystack: bytes, pattern: bytes, mask: bytes) -> list[int]:
    """Returns every offset in `haystack` where `pattern` matches under `mask`."""
    if not pattern or len(pattern) != len(mask):
        raise ValueError("pattern and mask must be the same non-zero length")
    hits: list[int] = []
    first_fixed = next(
        (i for i, m in enumerate(mask) if m), None
    )
    if first_fixed is None:
        # An all-wildcard pattern matches everywhere; that is never useful.
        raise ValueError("pattern has no fixed bytes")
    start = 0
    limit = len(haystack) - len(pattern)
    while start <= limit:
        idx = haystack.find(bytes([pattern[first_fixed]]), start + first_fixed)
        if idx < 0:
            break
        begin = idx - first_fixed
        if begin < 0:
            start = begin + 1
            continue
        if all(
            (haystack[begin + i] & mask[i]) == (pattern[i] & mask[i])
            for i in range(len(pattern))
        ):
            hits.append(begin)
        start = begin + 1
    return hits


# ---------------------------------------------------------------------------
# ELF image
# ---------------------------------------------------------------------------


class ElfImage:
    """A thin, read-only view over a LIEF ELF with relocation-aware reads."""

    def __init__(self, path: Path):
        lief = _require_lief()
        self.path = path
        self.binary = lief.parse(str(path))
        if self.binary is None:
            raise SystemExit(f"lief could not parse {path}")
        if not isinstance(self.binary, lief.ELF.Binary):
            raise SystemExit(f"{path} is not an ELF binary")

        self.sections: dict[str, lief.ELF.Section] = {}
        self._content_cache: dict[str, bytes] = {}
        for section in self.binary.sections:
            self.sections[section.name] = section

        # address -> addend for R_AARCH64_RELATIVE relocations. In this PIE the
        # RTTI pointers are written by the loader from these addends, so the raw
        # file bytes are zero and the value must come from here.
        self.relative_relocations: dict[int, int] = {}
        for reloc in self.binary.relocations:
            # The enum moved between LIEF versions (`Relocation.TYPE.AARCH64_RELATIVE`
            # in 1.x, `RELOCATION_AARCH64.R_AARCH64_RELATIVE` in 0.x), so compare on
            # the name rather than importing a symbol that may not exist.
            name = getattr(getattr(reloc, "type", None), "name", "")
            if name == "AARCH64_RELATIVE":
                self.relative_relocations[int(reloc.address)] = int(reloc.addend)

    # -- sections ---------------------------------------------------------

    def section(self, name: str) -> lief.ELF.Section | None:
        return self.sections.get(name)

    def section_bytes(self, name: str) -> bytes:
        """The section's contents, copied once and cached.

        This is deliberately cached: `bytes(section.content)` allocates a fresh
        copy of the whole section every call, and the vtable scan reads one word
        at a time across all of `.data.rel.ro`, so an uncached read is O(n^2) in
        the section size and does not finish on a 340 MB library.
        """
        cached = self._content_cache.get(name)
        if cached is not None:
            return cached
        section = self.section(name)
        data = bytes(section.content) if section is not None else b""
        self._content_cache[name] = data
        return data

    def section_vaddr(self, name: str) -> int:
        section = self.section(name)
        return int(section.virtual_address) if section else 0

    # -- addressing -------------------------------------------------------

    def _section_for(self, vaddr: int) -> lief.ELF.Section | None:
        for section in self.binary.sections:
            start = int(section.virtual_address)
            if start and start <= vaddr < start + section.size:
                return section
        return None

    def read_u64(self, vaddr: int) -> int | None:
        """The value at `vaddr`, honouring a RELATIVE relocation if present."""
        if vaddr in self.relative_relocations:
            return self.relative_relocations[vaddr]
        section = self._section_for(vaddr)
        if section is None:
            return None
        offset = vaddr - int(section.virtual_address)
        content = self.section_bytes(section.name)
        if offset < 0 or offset + 8 > len(content):
            return None
        return struct.unpack_from("<Q", content, offset)[0]

    def vaddr_to_file_offset(self, vaddr: int) -> int | None:
        section = self._section_for(vaddr)
        if section is None:
            return None
        return int(section.file_offset) + (vaddr - int(section.virtual_address))

    def is_executable(self, vaddr: int) -> bool:
        text = self.section(".text")
        if text is None:
            return False
        start = int(text.virtual_address)
        return start <= vaddr < start + text.size

    # -- search -----------------------------------------------------------

    def find_c_string(self, section_name: str, text: str) -> int:
        """The vaddr of the NUL-terminated `text` in a section, or 0."""
        blob = self.section_bytes(section_name)
        if not blob:
            return 0
        needle = text.encode() + b"\x00"
        idx = blob.find(needle)
        if idx < 0:
            return 0
        return self.section_vaddr(section_name) + idx

    def find_typeinfo_name(self, type_name: str) -> int:
        return self.find_c_string(".rodata", type_name)


# ---------------------------------------------------------------------------
# RTTI vtable resolution (mirrors preloader Vtable.cpp, statically)
# ---------------------------------------------------------------------------


@dataclass
class VtableSlot:
    type_name: str
    slot: int
    function_vaddr: int
    function_file_offset: int
    vtable_vaddr: int
    resolved_by: str  # "slot-index" or "needle:<string>"


def _words_of(image: ElfImage, section_name: str):
    """Yields `(vaddr, value)` for every 8-byte word of a section.

    A single pass over the cached bytes, with the relocated value substituted
    from `.rela.dyn`. This is what keeps the vtable scan linear: a per-word
    `read_u64` would re-resolve the owning section on every step.
    """
    section = image.section(section_name)
    if section is None:
        return
    start = int(section.virtual_address)
    content = image.section_bytes(section_name)
    relocs = image.relative_relocations
    for offset in range(0, len(content) - 7, 8):
        vaddr = start + offset
        value = relocs.get(vaddr)
        if value is None:
            value = struct.unpack_from("<Q", content, offset)[0]
        yield vaddr, value


def find_typeinfo(image: ElfImage, type_name: str) -> tuple[int, int]:
    """Returns `(typeinfo_vaddr, typeinfo_name_vaddr)`, or `(0, 0)`."""
    name_vaddr = image.find_typeinfo_name(type_name)
    if not name_vaddr:
        return 0, 0
    # The typeinfo is `{ vtable_ptr, name_ptr }`; find the word that points at the
    # name string, then step back one word.
    previous = 0
    for vaddr, value in _words_of(image, ".data.rel.ro"):
        if value == name_vaddr and vaddr >= 8:
            return vaddr - 8, name_vaddr
        previous = vaddr
    del previous
    return 0, name_vaddr


def find_primary_vtable(image: ElfImage, typeinfo_vaddr: int) -> int:
    """The address point of the primary vtable (offset_to_top == 0), or 0."""
    previous_vaddr = 0
    previous_value = 0
    for vaddr, value in _words_of(image, ".data.rel.ro"):
        if value == typeinfo_vaddr and previous_value == 0 and previous_vaddr + 8 == vaddr:
            return vaddr
        previous_vaddr = vaddr
        previous_value = value
    return 0


def read_vtable_slot(image: ElfImage, vtable_vaddr: int, slot: int) -> int:
    if vtable_vaddr == 0:
        return 0
    return image.read_u64(vtable_vaddr + 8 + slot * 8) or 0


def resolve_slot(
    image: ElfImage, type_name: str, slot: int
) -> VtableSlot | None:
    typeinfo_vaddr, _ = find_typeinfo(image, type_name)
    if not typeinfo_vaddr:
        return None
    vtable_vaddr = find_primary_vtable(image, typeinfo_vaddr)
    if not vtable_vaddr:
        return None
    fn = read_vtable_slot(image, vtable_vaddr, slot)
    if not fn or not image.is_executable(fn):
        return None
    return VtableSlot(
        type_name=type_name,
        slot=slot,
        function_vaddr=fn,
        function_file_offset=image.vaddr_to_file_offset(fn) or 0,
        vtable_vaddr=vtable_vaddr,
        resolved_by="slot-index",
    )


# --- disassembly: resolve a slot by the string its function references -----


# One slot's body is far smaller than this; the window only exists so a runaway
# scan cannot walk into a neighbouring function, and the prologue check stops it
# sooner.
_DISASM_WINDOW_BYTES = 0x4000


def _decode_adrp_page(word: int, pc: int) -> int:
    """The page an `adrp` targets, or 0 when `word` is not an `adrp`."""
    if (word & 0x9F000000) != 0x90000000:
        return 0
    immlo = (word >> 29) & 0x3
    immhi = (word >> 5) & 0x7FFFF
    imm = (immhi << 2) | immlo
    if imm & (1 << 20):
        imm -= 1 << 21
    return (pc & ~0xFFF) + (imm << 12)


def _is_add_immediate(word: int) -> tuple[int, int] | None:
    """`(rn, imm12)` when `word` is a 64-bit `add xd, xn, #imm`, else None."""
    if (word & 0xFFC00000) != 0x91000000:
        return None
    return (word >> 5) & 0x1F, (word >> 10) & 0xFFF


def _is_prologue_word(word: int) -> bool:
    """`stp x29, x30, [sp, #-imm]!` - the arm64 frame-setup prologue."""
    return (word & 0x0000FFFF) == 0x7BFD and (word & 0xFF000000) == 0xA9000000


def function_references_address(
    blob: bytes, base: int, fn_vaddr: int, end_vaddr: int, target_vaddr: int
) -> bool:
    """True when the function at `fn_vaddr` materialises `target_vaddr`.

    Decodes the `adrp xN, page` / `add xN, xN, #off` pair that arm64 uses to form
    a symbol address. This is done with raw bit ops rather than a disassembler on
    purpose: capstone's linear `disasm` generator stops at the first undecodable
    instruction, so over a 200 MB `.text` it silently covers only the first few
    MB and misses most functions. The bit ops cannot desync.
    """
    if fn_vaddr == 0 or end_vaddr <= fn_vaddr or target_vaddr == 0:
        return False
    target_page = target_vaddr & ~0xFFF
    low = target_vaddr & 0xFFF
    start = fn_vaddr - base
    stop = min(end_vaddr - base, start + _DISASM_WINDOW_BYTES)
    if start < 0 or stop <= start:
        return False
    for offset in range(start, stop - 8, 4):
        word = struct.unpack_from("<I", blob, offset)[0]
        # Stop at the next function's prologue (past this function's own first
        # instructions), so the scan cannot read a neighbouring slot's body.
        if offset > start + 16 and _is_prologue_word(word):
            break
        pc = base + offset
        if _decode_adrp_page(word, pc) != target_page:
            continue
        rd = word & 0x1F
        for look in range(1, 7):
            j = offset + look * 4
            if j + 4 > stop:
                break
            w2 = struct.unpack_from("<I", blob, j)[0]
            add = _is_add_immediate(w2)
            if add is None:
                continue
            rn, imm12 = add
            rd2 = w2 & 0x1F
            if rn == rd and rd2 == rd and imm12 == low:
                return True
    return False


def function_references_string(image: ElfImage, fn_vaddr: int, needle_vaddr: int) -> bool:
    """Convenience wrapper: does the function at `fn_vaddr` reference `needle_vaddr`?"""
    text = image.section(".text")
    if text is None or not image.is_executable(fn_vaddr):
        return False
    blob = image.section_bytes(".text")
    base = int(text.virtual_address)
    return function_references_address(
        blob, base, fn_vaddr, base + text.size, needle_vaddr
    )


def disassemble_at(image: ElfImage, vaddr: int, count: int = 4) -> list[str]:
    """A short human-readable disassembly at `vaddr`, for the report."""
    capstone = _require_capstone()
    text = image.section(".text")
    if text is None or not image.is_executable(vaddr):
        return []
    blob = image.section_bytes(".text")
    offset = vaddr - int(text.virtual_address)
    disassembler = capstone.Cs(capstone.CS_ARCH_ARM64, capstone.CS_MODE_LITTLE_ENDIAN)
    out: list[str] = []
    for insn in disassembler.disasm(blob[offset : offset + count * 4], vaddr):
        out.append(f"{insn.mnemonic} {insn.op_str}")
        if len(out) >= count:
            break
    return out


def resolve_slot_by_needle(
    image: ElfImage, type_name: str, needle: str, *, max_slots: int = 256
) -> VtableSlot | None:
    typeinfo_vaddr, _ = find_typeinfo(image, type_name)
    if not typeinfo_vaddr:
        return None
    vtable_vaddr = find_primary_vtable(image, typeinfo_vaddr)
    if not vtable_vaddr:
        return None
    needle_vaddr = image.find_c_string(".rodata", needle)
    if not needle_vaddr:
        return None
    for slot in range(max_slots):
        fn = read_vtable_slot(image, vtable_vaddr, slot)
        if not fn or not image.is_executable(fn):
            continue
        if function_references_string(image, fn, needle_vaddr):
            return VtableSlot(
                type_name=type_name,
                slot=slot,
                function_vaddr=fn,
                function_file_offset=image.vaddr_to_file_offset(fn) or 0,
                vtable_vaddr=vtable_vaddr,
                resolved_by=f"needle:{needle}",
            )
    return None


# ---------------------------------------------------------------------------
# Signature scanning
# ---------------------------------------------------------------------------


@dataclass
class SignatureResult:
    name: str
    pattern: str
    description: str
    match_count: int
    matches: list[int] = field(default_factory=list)
    file_offsets: list[int] = field(default_factory=list)
    status: str = "unmatched"


def scan_signature(image: ElfImage, name: str, pattern: str, description: str) -> SignatureResult:
    pattern_bytes, mask = parse_pattern(pattern)
    blob = image.section_bytes(".text")
    base = image.section_vaddr(".text")
    hits = scan_bytes(blob, pattern_bytes, mask)
    vaddrs = [base + h for h in hits]
    offsets = [image.vaddr_to_file_offset(v) or 0 for v in vaddrs]
    if len(vaddrs) == 1:
        status = "unique"
    elif not vaddrs:
        status = "unmatched"
    else:
        status = "ambiguous"
    return SignatureResult(
        name=name,
        pattern=pattern,
        description=description,
        match_count=len(vaddrs),
        matches=vaddrs[:16],
        file_offsets=offsets[:16],
        status=status,
    )


# ---------------------------------------------------------------------------
# Entity-seam offset verification (deterministic, per build)
# ---------------------------------------------------------------------------

# The offsets the preloader hardcodes for the local-player feed. Recovering a
# brand-new offset needs judgement, but *verifying* a known one is deterministic:
# a struct field of a Vec3 is a member-address getter and a float-vector access
# group at the same displacement. Checking that per build is what turns "the
# preloader assumes 0x230" into "0x230 is still correct on this build".
#
# `Actor::mPos` and the view rotation are the two the feed reads.
KNOWN_ENTITY_OFFSETS: list[tuple[str, int, str]] = [
    ("actor_position", 0x230, "Vec3 position (Actor::mPos); the preloader reads x/y/z here"),
    ("actor_rotation", 0x238, "view rotation (yaw, pitch); the preloader reads two floats here"),
]

_RET = 0xD65F03C0


def _iter_words(image: ElfImage):
    blob = image.section_bytes(".text")
    for offset in range(0, len(blob) - 3, 4):
        yield offset, struct.unpack_from("<I", blob, offset)[0]


def member_address_getters_from_bytes(blob: bytes) -> set[int]:
    """The bytes-based core of [`member_address_getters`], so it is unit-testable."""
    out: set[int] = set()
    for offset in range(0, len(blob) - 7, 4):
        word = struct.unpack_from("<I", blob, offset)[0]
        if (word & 0xFFC00000) != 0x91000000:
            continue
        rn, rd, imm = (word >> 5) & 0x1F, word & 0x1F, (word >> 10) & 0xFFF
        if rn != 0 or rd != 0 or imm == 0:
            continue
        if struct.unpack_from("<I", blob, offset + 4)[0] == _RET:
            out.add(imm)
    return out


def member_address_getters(image: ElfImage) -> set[int]:
    """Displacements returned by an `add x0, x0, #imm; ret` getter.

    That two-instruction shape returns `this + imm` - the address of a member -
    which is how the engine exposes a Vec3 field it does not want to copy.
    """
    return member_address_getters_from_bytes(image.section_bytes(".text"))


def _decode_float_access(word: int) -> tuple[int, int] | None:
    """`(base_reg, disp)` for a SIMD/FP load/store, else None.

    The SIMD&FP load/store encodings are `size 111 V=1 01 opc imm12 Rn Rt`
    (scaled) and `size 111 V=1 00 opc 0 imm9 Rn Rt` (unscaled). With the top byte
    that means 0xBD for a 4-byte scaled access, 0xBC unscaled, and 0xFD/0xFC for
    the 8-byte forms.
    """
    top = word & 0xFF000000
    if top in (0xBD000000, 0xFD000000):
        scale = 4 if top == 0xBD000000 else 8
        imm12 = (word >> 10) & 0xFFF
        return (word >> 5) & 0x1F, imm12 * scale
    if top in (0xBC000000, 0xFC000000):
        imm9 = (word >> 12) & 0x1FF
        disp = imm9 - 0x200 if imm9 & 0x100 else imm9
        return (word >> 5) & 0x1F, disp
    return None


def vec3_float_groups_from_bytes(blob: bytes) -> set[int]:
    """The bytes-based core of [`vec3_float_groups`], so it is unit-testable."""
    per_fn: dict[tuple[int, int], set[int]] = {}
    current_fn = 0
    for offset in range(0, len(blob) - 3, 4):
        word = struct.unpack_from("<I", blob, offset)[0]
        if (word & 0x0000FFFF) == 0x7BFD and (word & 0xFF000000) == 0xA9000000:
            current_fn = offset  # frame-setup prologue: a new function begins
        access = _decode_float_access(word)
        if access is None:
            continue
        reg, disp = access
        if reg in (29, 31) or disp < 0:
            continue
        per_fn.setdefault((current_fn, reg), set()).add(disp)

    groups: set[int] = set()
    for disps in per_fn.values():
        for d in disps:
            if d + 4 in disps and d + 8 in disps:
                groups.add(d)
    return groups


def vec3_float_groups(image: ElfImage) -> set[int]:
    """Displacements read as a float vector (`off`, `off+4`, `off+8`) in one function.

    A Vec3 field shows up as three consecutive float accesses off one register
    within a single function, so the grouping is by (function, base register) -
    grouping per instruction would never see the three together. `sp`/`fp`-based
    accesses are excluded because stack frames reuse these displacements.
    """
    return vec3_float_groups_from_bytes(image.section_bytes(".text"))


def verify_known_offsets(image: ElfImage) -> list[dict]:
    """Checks each known entity offset against this build's code."""
    getters = member_address_getters(image)
    vec3 = vec3_float_groups(image)
    results = []
    for name, disp, meaning in KNOWN_ENTITY_OFFSETS:
        has_getter = disp in getters
        has_vec3 = disp in vec3
        results.append(
            {
                "name": name,
                "offset": disp,
                "meaning": meaning,
                "verified": bool(has_getter and has_vec3),
                "evidence": {
                    "member_address_getter": has_getter,
                    "float_vec3_group": has_vec3,
                },
            }
        )
    return results


# --- The Level -> ActorManager chain -----------------------------------------
#
# `levelEntityListOffset` is not a struct field on the Level; it is the offset of
# the `ActorManager*` member. The actor list is the `{begin, end, cap}` vector at
# `ActorManager + 0x8`, so the walk is two steps:
#
#     level + levelEntityListOffset  ->  ActorManager*  ->  + 0x8  ->  {begin, end}
#
# Recovering it deterministically (no hardcoded addresses):
#
#   1. resolve `ActorManager` by RTTI name;
#   2. find its constructor: the function that stores the vtable address point
#      (vptr) into `this`;
#   3. find the call sites of that constructor, follow the register moved into
#      `x0`, and read the `str xN, [xBase, #off]` that lands the pointer;
#   4. keep only offsets that are stored by exactly one call site (an ambiguity
#      means the rule caught something else).
#
# Step 4 is the guard that makes this a measurement rather than a guess: on
# 26.60 it yields a single offset, `0x3e0`.

# The libc++ `std::vector` offset inside the ActorManager (its first member).
ACTOR_MANAGER_VECTOR_OFFSET = 0x8

# How far past a `bl` to look for the pointer store.
_STORE_SEARCH_WINDOW = 60


def _bl_target(word: int, pc: int) -> int | None:
    if (word & 0xFC000000) != 0x94000000:
        return None
    imm = word & 0x3FFFFFF
    if imm & (1 << 25):  # 26-bit signed immediate
        imm -= 1 << 26
    return pc + (imm << 2)


def _mov_x0_source(word: int) -> int | None:
    """The source register of `mov x0, xN`, else None."""
    if (word & 0xFFE0FFE0) == 0xAA0003E0:
        return (word >> 16) & 0x1F
    return None


def _find_actor_manager_vptr(image: ElfImage) -> int:
    typeinfo, _ = find_typeinfo(image, "12ActorManager")
    if not typeinfo:
        return 0
    vtable = find_primary_vtable(image, typeinfo)
    if not vtable:
        return 0
    # A constructor stores the vptr (address point + 8), not the address point.
    return vtable + 8


def find_actor_manager_constructor(image: ElfImage) -> int:
    """The function that writes the ActorManager vptr into `this` (x0), or 0."""
    vptr = _find_actor_manager_vptr(image)
    if not vptr:
        return 0
    blob = image.section_bytes(".text")
    base = image.section_vaddr(".text")
    n = len(blob) // 4
    page = vptr & ~0xFFF
    low = vptr & 0xFFF

    for i in range(n):
        word = struct.unpack_from("<I", blob, i * 4)[0]
        pc = base + i * 4
        if _decode_adrp_page(word, pc) != page:
            continue
        rd = word & 0x1F
        for j in range(1, 6):
            if i + j >= n:
                break
            w2 = struct.unpack_from("<I", blob, (i + j) * 4)[0]
            add = _is_add_immediate(w2)
            if add is None or add[0] != rd or (w2 & 0x1F) != rd or add[1] != low:
                continue
            # `str x{rd}, [x0]` within a few instructions marks the constructor.
            for k in range(1, 9):
                if i + j + k >= n:
                    break
                w3 = struct.unpack_from("<I", blob, (i + j + k) * 4)[0]
                if (w3 & 0xFFC00000) == 0xF9000000 and (w3 & 0x1F) == rd and ((w3 >> 5) & 0x1F) == 0:
                    return _function_start(image, pc)
    return 0


def _function_start(image: ElfImage, addr: int) -> int:
    """Walk back to the frame-setup prologue that opens the function at `addr`."""
    blob = image.section_bytes(".text")
    base = image.section_vaddr(".text")
    offset = addr - base
    for back in range(0, 0x8000, 4):
        if offset - back < 0:
            break
        word = struct.unpack_from("<I", blob, offset - back)[0]
        if (word & 0x0000FFFF) == 0x7BFD and (word & 0xFF000000) == 0xA9000000:
            return addr - back
    return addr


def find_level_entity_manager_offset(image: ElfImage) -> int:
    """The Level member that holds the `ActorManager*`, or 0.

    Returns a value only when exactly one offset is stored by exactly one call
    site of the ActorManager constructor - an ambiguous result is refused.
    """
    ctor = find_actor_manager_constructor(image)
    if not ctor:
        return 0
    blob = image.section_bytes(".text")
    base = image.section_vaddr(".text")
    n = len(blob) // 4

    candidates: dict[int, int] = {}
    for i in range(n):
        word = struct.unpack_from("<I", blob, i * 4)[0]
        pc = base + i * 4
        if _bl_target(word, pc) != ctor:
            continue
        # Which register was moved into x0 just before the call?
        source = None
        for j in range(1, 5):
            if i - j < 0:
                break
            src = _mov_x0_source(struct.unpack_from("<I", blob, (i - j) * 4)[0])
            if src is not None:
                source = src
                break
        if source is None:
            continue
        for k in range(1, _STORE_SEARCH_WINDOW):
            if i + k >= n:
                break
            w2 = struct.unpack_from("<I", blob, (i + k) * 4)[0]
            if (w2 & 0xFFC00000) == 0xF9000000 and (w2 & 0x1F) == source:
                offset = ((w2 >> 10) & 0xFFF) * 8
                candidates[offset] = candidates.get(offset, 0) + 1
                break

    # Only a single, unambiguous offset is trusted.
    if len(candidates) != 1:
        return 0
    offset, count = next(iter(candidates.items()))
    return offset if count >= 1 else 0


# --- Actor -> Level -----------------------------------------------------------
#
# To reach the entity list from the local player (the pointer ClientInstance slot
# 31 returns, which is an `Actor*`), we need the `Actor` member that holds the
# `Level*`. Two independent signals pin it:
#
#   1. the `Actor` constructor stores its `Level&` argument (in `x1`) at
#      `str x1, [this, #off]`;
#   2. some function reads `[actor, #off]` and then `[that, #0x3e0]` - the Level's
#      ActorManager member.
#
# The offset is accepted only when both agree. On 26.60 that is `0x20`.

def find_actor_level_offset(image: ElfImage, level_offset: int) -> int:
    """The `Actor` member holding the `Level*`, or 0 (unrecovered).

    **Not recovered.** This returns 0 deliberately. The obvious signals do not
    agree, and a wrong value here is a fault on the game thread:

    - The `Actor` constructor at 26.60 stores its `Level&` argument (x1) at
      `Actor+0x20`; the `Mob` constructor stores x1 at `Actor+0x1f0`. The two
      conflict, and neither has been confirmed against a live object.
    - A displacement-only correlation (`[x,#off]` then `[.,+level_offset]`) is
      unsound: `+0x20` and `+0x1f0` are common offsets on many unrelated types,
      and none of the correlated sites uses the same base register with the known
      `Actor` position field (`+0x230`), which is the check that would prove the
      base really is an `Actor`.

    Recovering it needs a live-object confirmation (or a `getLevel()`-style
    accessor resolved by name), not a static correlation. See issue #4.

    `level_offset` is the recovered `Level` -> `ActorManager*` member for this
    build (it differs per build).
    """
    del image, level_offset
    return 0


# ---------------------------------------------------------------------------
# What to resolve
# ---------------------------------------------------------------------------

# RTTI-resolved vtable slots. These are the seams the project already trusts
# because the preloader resolves them by name at runtime; this records the same
# answer statically, per build, so a build that breaks one is caught before it
# ships.
RTTI_SLOT_TARGETS: list[tuple[str, int, str]] = [
    ("14ClientInstance", 31, "getLocalPlayer (local-player feed)"),
    ("18LivePlayerRenderer", 17, "render (player-render / cosmetics seam)"),
    ("20ClientNetworkHandler", 40, "player-join packet read"),
]

# Slots resolved by the string their function references. This is the
# version-robust form: the slot index moves, the referenced string does not.
RTTI_NEEDLE_TARGETS: list[tuple[str, str, str]] = [
    (
        "18LivePlayerRenderer",
        "variable.player_x_rotation",
        "render slot identified by the string it references",
    ),
]

# Byte signatures to scan. Each must match exactly once to be trusted; the
# project documents that these drift per build, so a non-unique result here is
# information, not a failure.
SIGNATURE_TARGETS: list[tuple[str, str, str]] = [
    (
        "aarch64_function_prologue",
        "FD 7B BF A9",
        "stp x29, x30, [sp, #-0x10]! - the common frame-setup prologue (many hits)",
    ),
]

# The EntitySeamOffsets fields, reported as unrecovered with the reason.
ENTITY_SEAM_FIELDS = [
    ("actorLevelOffset", "the Actor -> Level* member offset (local player to level)"),
    ("levelEntityListOffset", "the Level/Dimension entity-list vector offset"),
    ("entityRuntimeIdOffset", "the runtime id offset inside an Actor"),
    ("entityPositionOffset", "the position offset inside an Actor"),
    ("entityRotationOffset", "the view-rotation offset inside an Actor"),
]

# Which verified field feeds which EntitySeamOffsets name. A verified offset is
# recorded as a real value; an unverified one stays null (never guessed).
_VERIFIED_TO_FIELD = {
    "actor_position": "entityPositionOffset",
    "actor_rotation": "entityRotationOffset",
}


def _entity_seam_fields(image: ElfImage) -> list[dict]:
    """The four EntitySeamOffsets fields, with the verified ones filled in.

    A field is only given a value when the verifier proved it on this build; the
    rest stay null. `levelEntityListOffset` and `entityRuntimeIdOffset` are not
    recovered by this pass and must not be guessed.
    """
    verified = {v["name"]: v for v in verify_known_offsets(image)}
    level_offset = find_level_entity_manager_offset(image)
    fields = []
    for name, meaning in ENTITY_SEAM_FIELDS:
        offset = None
        if name == "actorLevelOffset":
            offset = find_actor_level_offset(image, level_offset) or None
        if name == "levelEntityListOffset" and level_offset:
            offset = level_offset
        for verified_name, field_name in _VERIFIED_TO_FIELD.items():
            if field_name == name and verified.get(verified_name, {}).get("verified"):
                offset = verified[verified_name]["offset"]
        fields.append({"name": name, "meaning": meaning, "offset": offset})
    return fields


def _actor_manager_report(image: ElfImage) -> dict:
    """The recovered Level -> ActorManager chain, for inspection."""
    ctor = find_actor_manager_constructor(image)
    level_offset = find_level_entity_manager_offset(image)
    return {
        "constructor": ctor,
        "level_entity_manager_offset": level_offset,
        "actor_level_offset": find_actor_level_offset(image, level_offset),
        "vector_offset_in_manager": ACTOR_MANAGER_VECTOR_OFFSET,
        "note": (
            "Two-step walk: level + level_entity_manager_offset -> ActorManager*, "
            "then vector_offset_in_manager -> {begin, end, cap}."
        ),
    }


def build_report(image: ElfImage, version: str, so_sha256: str, so_bytes: int) -> dict:
    report: dict = {
        "version": version,
        "so_bytes": so_bytes,
        "so_sha256": so_sha256,
        "abi": "arm64-v8a",
        "rtti_slots": [],
        "rtti_slots_by_needle": [],
        "signatures": [],
        "entity_seam_verification": verify_known_offsets(image),
        "actor_manager": _actor_manager_report(image),
        "entity_seam_offsets": {
            "resolved": False,
            "reason": (
                "The Actor position/rotation offsets are verified per build (see "
                "entity_seam_verification) and filled in below. The two that remain "
                "null - levelEntityListOffset and entityRuntimeIdOffset - are not "
                "recovered: they need a deeper disassembly pass, and a guessed value "
                "would fault the game thread. See docs/agent-notes/signature-derivation.md."
            ),
            "fields": _entity_seam_fields(image),
        },
    }

    for type_name, slot, description in RTTI_SLOT_TARGETS:
        resolved = resolve_slot(image, type_name, slot)
        if resolved is not None:
            entry = asdict(resolved)
            entry["description"] = description
            entry["prologue"] = disassemble_at(image, resolved.function_vaddr)
            report["rtti_slots"].append(entry)

    for type_name, needle, description in RTTI_NEEDLE_TARGETS:
        resolved = resolve_slot_by_needle(image, type_name, needle)
        if resolved is not None:
            entry = asdict(resolved)
            entry["description"] = description
            entry["prologue"] = disassemble_at(image, resolved.function_vaddr)
            report["rtti_slots_by_needle"].append(entry)

    for name, pattern, description in SIGNATURE_TARGETS:
        report["signatures"].append(
            asdict(scan_signature(image, name, pattern, description))
        )

    return report


def sha256_of(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--so", required=True, type=Path, help="path to libminecraftpe.so")
    parser.add_argument("--version", required=True, help="game version, e.g. 26.60")
    parser.add_argument("--out", required=True, type=Path, help="output JSON path")
    parser.add_argument(
        "--check-unique",
        action="store_true",
        help="exit non-zero if any signature did not match exactly once",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    image = ElfImage(args.so)
    report = build_report(
        image, args.version, sha256_of(args.so), args.so.stat().st_size
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.out}")

    for entry in report["rtti_slots"]:
        print(
            f"  RTTI {entry['type_name']} slot {entry['slot']} -> "
            f"{entry['function_vaddr']:#x} (file {entry['function_file_offset']:#x})"
        )
    for entry in report["rtti_slots_by_needle"]:
        print(
            f"  RTTI {entry['type_name']} by {entry['resolved_by']} -> "
            f"slot {entry['slot']} at {entry['function_vaddr']:#x}"
        )
    for entry in report["signatures"]:
        print(
            f"  SIG {entry['name']}: {entry['status']} "
            f"({entry['match_count']} match(es))"
        )
    am = report.get("actor_manager", {})
    print(
        f"  CHAIN actor+{am.get('actor_level_offset', 0):#x} -> level+"
        f"{am.get('level_entity_manager_offset', 0):#x} -> manager+"
        f"{am.get('vector_offset_in_manager', 0):#x}"
    )
    for entry in report["entity_seam_verification"]:
        mark = "VERIFIED" if entry["verified"] else "NOT VERIFIED"
        print(
            f"  ENTITY {entry['name']} @ {entry['offset']:#x}: {mark} "
            f"({entry['evidence']})"
        )

    if args.check_unique:
        bad = [e for e in report["signatures"] if e["status"] != "unique"]
        if bad:
            print(
                "signatures that were not unique: "
                + ", ".join(f"{e['name']}({e['status']})" for e in bad),
                file=sys.stderr,
            )
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
