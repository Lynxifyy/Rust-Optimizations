# Signature and offset derivation for the Bedrock client

How the per-build offset maps (`offsets/offsets_<version>.json`) are produced,
what they contain, what they deliberately do **not** contain, and how to refresh
them when Mojang ships a new build.

The tools are `tools/download_and_extract.py` and `tools/extract_offsets.py`.

## What the map contains

Two kinds of thing, with very different lifetimes:

| Field | Kind | Lifetime |
|---|---|---|
| `rtti_slots` | RTTI-resolved vtable slots | **Stable across builds** - the slot *index* is the same on 26.52/26.60/26.62; only the *address* moves |
| `rtti_slots_by_needle` | The same slots, resolved by the string their function references | Stable - survives a slot-index change |
| `signatures` | Byte patterns scanned in `.text` | **Per build** - a pattern valid for one build matches 0 or 2+ places on another |
| `entity_seam_offsets` | `Level`/`Actor` field offsets | **Not yet recovered** - see below |

### RTTI-resolved vtable slots (the reliable part)

The game strips method symbols but keeps RTTI typeinfo-name strings, so a vtable
slot can be found by *name*. `extract_offsets.py` mirrors the preloader's
`Vtable.cpp` algorithm, but statically:

1. Find the typeinfo-name string (e.g. `14ClientInstance`) in `.rodata`.
2. Find the typeinfo object in `.data.rel.ro` whose second word points at that
   string.
3. Find the primary vtable: a word pointing at the typeinfo with
   `offset_to_top == 0` immediately before it.
4. Read the slot: `vtable + 8 + slot * 8`.

The subtlety is step 2/3 on a static file: this is a PIE, so the pointers in
`.data.rel.ro` are written by the loader from `R_AARCH64_RELATIVE` addends in
`.rela.dyn`, and the raw file bytes are **zero**. Reading the file's bytes gives
nothing; the addends must be read from the relocations. That is the single most
important thing to get right, and it is why a naive byte scan "proves" RTTI is
absent.

Resolved on all three builds, cross-checked two ways (slot index and the
referenced string agree):

| Type | Slot | 26.52 | 26.60 | 26.62 |
|---|---|---|---|---|
| `14ClientInstance` | 31 | `0x9807fbc` | `0x9b7f610` | `0x9b7f6d0` |
| `18LivePlayerRenderer` | 17 | `0x967e1d8` | `0x9a2761c` | `0x9a276dc` |
| `20ClientNetworkHandler` | 40 | `0x9ed8050` | `0xa2e4d94` | `0xa2e51c4` |

Each address was verified to be a real function entry point: 4-byte aligned,
inside `.text`, beginning with a genuine prologue (e.g. `stp d15, d14, [sp,
#-0xa0]!` for the renderer). A wrong resolution would land mid-instruction.

### Byte signatures (the fragile part)

A signature is scanned across `.text` and is only trusted when it matches
**exactly once**. The scanner reports `unique` / `unmatched` / `ambiguous` with
the candidate offsets, so a pattern that drifts is visible rather than silently
wrong.

The one bundled pattern, `FD 7B BF A9` (`stp x29, x30, [sp, #-0x10]!`), is
**ambiguous** on every build (41k-44k matches) - which is the honest result for a
common prologue and is exactly why a generic pattern is not a usable anchor. A
real signature has to be derived from a specific build's `.text` around a known
function, not copied from a list.

## What the map does NOT contain, and why

`entity_seam_offsets` is emitted **unresolved** with the reason. Those four
fields (`levelEntityListOffset`, `entityRuntimeIdOffset`, `entityPositionOffset`,
`entityRotationOffset`) are byte offsets *inside a live `Level`/`Actor`*. Neither
a byte signature nor an RTTI name reveals them - they come from disassembling the
accessors that read those members. Emitting a guessed number there would be a
fault on the game thread, so the tool says "not recovered" and the mod keeps the
fallback. Recovering them is the next job, and it is a disassembly pass, not a
signature scan.

Likewise the `LevelChunk` and mesher functions named in the original plan are not
in the map: the game's chunk mesher is not reachable by any anchor (see
`docs/agents/native-entity-camera-feed-*` in the client repo). The map records
what is actually resolvable.

## About external signature databases

Pattern databases for **other** Bedrock clients (e.g. Horion, or LeviLamina's
signature API) are not a source for these maps. A byte pattern is valid for one
build of one binary; a pattern derived against a different binary, or a different
build, is expected to match 0 or many places - which is exactly what the scanner
reports when you try. The RTTI route is used instead precisely because it does
not need a per-build pattern.

## Refreshing for a new build

```sh
pip install -r tools/requirements.txt
python3 tools/download_and_extract.py --versions 26.70 --out work
python3 tools/extract_offsets.py --so work/so/libminecraftpe-26_70.so \
    --version 26.70 --out offsets/offsets_26_70.json --check-unique
```

`--check-unique` exits non-zero if any signature is not unique, so a CI job can
fail on a drifted pattern rather than shipping it. Then update the mod's
`game_version` and `minecraftpe_sha256` config to select and verify the new map.

## Cost

Each APK is ~380 MB and each extracted `.so` is ~330-370 MB. Three versions is
about 1.1 GB of APK downloads and 1 GB of `.so` files; `download_and_extract.py`
deletes each APK after extraction unless `--keep-apk` is passed. The extractor
itself takes under a minute per `.so` (the RTTI scan is linear over
`.data.rel.ro`, and the string-reference check is bounded to a 16 KB window per
slot).
