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
    ("levelEntityListOffset", "the Level/Dimension entity-list vector offset"),
    ("entityRuntimeIdOffset", "the runtime id offset inside an Actor"),
    ("entityPositionOffset", "the position offset inside an Actor"),
    ("entityRotationOffset", "the view-rotation offset inside an Actor"),
]


def build_report(image: ElfImage, version: str, so_sha256: str, so_bytes: int) -> dict:
    report: dict = {
        "version": version,
        "so_bytes": so_bytes,
        "so_sha256": so_sha256,
        "abi": "arm64-v8a",
        "rtti_slots": [],
        "rtti_slots_by_needle": [],
        "signatures": [],
        "entity_seam_offsets": {
            "resolved": False,
            "reason": (
                "Field offsets inside a live Level/Actor are not revealed by any "
                "signature or RTTI name; they require disassembling the accessors. "
                "Not fabricated - see docs/agent-notes/signature-derivation.md."
            ),
            "fields": [
                {"name": name, "meaning": meaning, "offset": None}
                for name, meaning in ENTITY_SEAM_FIELDS
            ],
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
