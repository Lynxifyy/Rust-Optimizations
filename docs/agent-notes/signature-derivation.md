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

`entity_seam_offsets` is partially filled. Two of the four fields are **verified
per build** and carry a value; two remain `null`.

| Field | Status | How |
|---|---|---|
| `entityPositionOffset` | **verified = `0x230`** | member-address getter + float-vec3 group |
| `entityRotationOffset` | **verified = `0x238`** | member-address getter + float-vec3 group |
| `levelEntityListOffset` | **recovered** (`0x3e0` on 26.60/26.62, `0x3d8` on 26.52) | constructor store trace (see below) |
| `actorLevelOffset` | **null** | not recovered - the walk's missing link (see below) |
| `entityRuntimeIdOffset` | **null** | not recovered (needs a deeper pass) |

The entity-list walk needs one more step before `Level`: from the local player
(what `ClientInstance` slot 31 returns, an `Actor*`) to its `Level*`. That offset
is **not recovered**, and the tool returns 0 rather than guess:

- the `Actor` constructor stores its `Level&` argument (`x1`) at `Actor+0x20`,
  while the `Mob` constructor stores `x1` at `Actor+0x1f0` - they conflict;
- a displacement-only correlation (`[x,#off]` then `[.,+level_offset]`) is
  unsound: `+0x20`/`+0x1f0` are common offsets on many types, and none of the
  correlated sites uses the same base register with the known `Actor` position
  field (`+0x230`), which is the check that would prove the base is an `Actor`.

Recovering it needs a live-object confirmation or a `getLevel()`-style accessor
resolved by name, not a static correlation. Until then the walker refuses (it
requires `actorLevelOffset`) and the mod keeps the fallback.

`levelEntityListOffset` is **not** a struct field on the `Level`; it is the offset
of the `ActorManager*` member. The actor list is the `{begin, end, cap}` vector at
`ActorManager + 0x8`, so the walk is two steps:

```text
level + level_entity_manager_offset  ->  ActorManager*  ->  +0x8  ->  {begin, end}
```

It is recovered deterministically, with no hardcoded address:

1. resolve `ActorManager` by RTTI name (`12ActorManager`);
2. find its constructor: the function that stores the vtable address point
   (`vptr`) into `this`;
3. find the call sites of that constructor, follow the register moved into `x0`,
   and read the `str xN, [xBase, #off]` that lands the pointer;
4. trust the result only when exactly one offset is stored by the call sites.

On 26.60 the trace runs through the `Level` constructor at `0x10afe58c` (which
stores the Level vptr `0x13e0a0f8`), and the store is `str x20, [x19, #0x3e0]` -
one call site, one offset. Cross-checked: a second, independent scan for
`[x, #0x3e0] -> read +8 and +0x10` (the vector walk) finds the same chain in
non-stack sites, one of them inside the Level itself. The `ActorManager` ctor also
zero-initializes its vector at `+8/+0x10/+0x18`, confirming the libc++ layout
`EntityListWalk` expects.

### How the two are verified

Recovering a *new* offset needs judgement, but **verifying a known one is
deterministic**, so the tool checks two independent signals per build:

1. **A member-address getter**: the two-instruction shape `add x0, x0, #0x230;
   ret`, which returns `this + 0x230`. The engine uses these to hand out a `Vec3`
   field's address instead of copying it. The scan found a getter for both
   `0x230` and `0x238`.
2. **A float-vec3 access group**: a function that reads `off`, `off+4`, `off+8`
   as SIMD/FP floats off the same register - the shape of a `Vec3` read. `sp`/
   `fp`-relative accesses are excluded because stack frames reuse these
   displacements.

Both signals hold for both offsets on **26.52, 26.60 and 26.62**. That turns "the
preloader assumes `Actor+0x230`" into "`0x230` is still correct on this build" -
which is exactly what a per-build map is for, and it matches the offsets the
preloader hardcodes in `GameLocalPlayer.cpp`.

### What is still not recovered

`entityRuntimeIdOffset` is **null**, not guessed. Notes for the next pass (also in
issue #4):

- `12ActorManager` and `5Level` **are** RTTI-resolvable (`resolveVtableFunction`
  finds them), contrary to an earlier assumption that the entity manager had no
  RTTI name. `ActorManager`'s vtable address point is `0x13d6eda8` on 26.60, and
  its slots 0/1/2 are member-address getters returning `this + 0x90 / 0xe8 /
  0x140`; slot 10 iterates a `{begin, end}` vector at `+8`/`+0x10`.
- `entityRuntimeIdOffset` is a field of `Actor`; the Actor vtable's slots resolve
  to pure-virtual interface bases, so the offset has to come from a function that
  reads the runtime id (e.g. the packet handlers), not from a vtable getter.

The `LevelChunk` and chunk-mesher functions named in the original plan remain
absent: the game's chunk mesher is not reachable by any anchor (the client
documents this).

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
