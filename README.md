# Rust-Optimizations

A native acceleration core for [GlowberryClient](https://github.com/ChimeraAnt-DEV/GlowberryClient),
written in Rust and shipped as a preloader mod, `libglowberry_rust.so`.

The goal is to move the heavy per-frame and per-chunk work off the game's own
threads and onto SIMD-friendly, lock-free Rust: SIMD 3-D math for the physics and
cull paths, binary greedy meshing for voxel data, and a bounded lock-free ring
buffer that hands finished meshes to the render thread without a lock.

## What this is, and what it is not

GlowberryClient is a Minecraft **Bedrock** launcher. A mod for it is not an APK
component - it is a shared library the client's **preloader** `dlopen`s into the
game process, where it can reach the game's memory. This repository builds that
library.

Two boundaries are stated up front, because the project it plugs into is explicit
about them:

- **The game's own chunk mesher is not reachable.** `libminecraftpe.so` is
  stripped of game-class symbols and the chunk mesh path exports nothing, so a
  mod cannot hook "the" mesher by name - see GlowberryClient's
  `docs/agents/native-entity-camera-feed-what-is-and-is-not-reachable.md`. This
  core therefore accelerates meshing of voxel data **handed to it**; it is a
  library the client's own voxel work (world reconstruction, preview) calls. The
  mod here installs no hook, because a hook without a verified address is a
  crash, not an optimization.
- **Allocator interposition is not done here.** Routing `malloc`/`free` through
  mimalloc was tried in the preloader and removed for concrete reasons (hidden
  visibility meant the definitions never reached the game's allocations;
  `thread_local` recurses under the emulated TLS of minSdk 28; a libc block freed
  by the interposer faults). Reintroducing it would re-introduce the crash. See
  the preloader's `AGENTS.md`. Instead the core allocates nothing per frame and
  reuses caller-owned buffers.

What it *does* provide is real and testable: the modules below, all
host-tested, plus an in-session work loop the mod runs inside the game process.

## The in-session loop (Phase 2)

`mod/GlowberryRustMod.cpp` does not just load the core; on `load` it starts the
pipeline and runs a real workload through it:

1. It takes the client's own world model - a footprint of **block layers**, the
   shape `FlatWorldGenerator` builds (bedrock / dirt / grass) - and packs it into
   a voxel volume.
2. It splits that volume into chunks and **submits every chunk** to the pipeline,
   which hands each to a Rust worker thread.
3. Each worker greedy-meshes its chunk and pushes the finished mesh onto a
   **lock-free ring**.
4. The mod **drains** the ring, exactly as a render thread would once per frame,
   and frees each mesh.

It then logs the chunk count, the total quads, the wall-clock time and the
pipeline counters (`submitted`/`meshed`/`drained`/`dropped`/`rejected`). The
assertion is that **drained == submitted and dropped == 0**: the pipeline neither
lost nor duplicated work. That log line is the proof the core does real work
in-session, and it is exercised end to end on the host by
`cpp/ffi_smoke_test.cpp`.

The same loop is reachable from any native code: the pipeline handle is kept live
for the session, and the whole `gbr_pipeline_*` surface is exported from the
`.so`, so the client's own voxel work can submit chunks to the same workers.

### What the loop is not

It is not wired to the running game's chunk data. Bedrock's world is only
reachable through the BerryAPI seam (`all_entities`, `get_block`), and those
slots are still null because the offsets that would fill them have not been
recovered - that is the disassembly job the next PR is for. Until then the loop
meshes the client's own world model, which is the one place the client itself
owns voxel-shaped data.

## Per-build offset maps (Phase 3)

`tools/` turns a Minecraft Bedrock APK into a per-build offset map:

- `tools/download_and_extract.py` resolves the mcpedl download chain, downloads
  the APK and extracts `lib/arm64-v8a/libminecraftpe.so`.
- `tools/extract_offsets.py` resolves RTTI vtable slots by name (the reliable
  part - the slot index is stable across builds) and scans byte signatures
  (the fragile part - it reports `unique`/`unmatched`/`ambiguous`).

Maps for **26.52, 26.60 and 26.62** are committed under `offsets/`. Each
resolves `ClientInstance` slot 31, `LivePlayerRenderer` slot 17 and
`ClientNetworkHandler` slot 40, cross-checked two independent ways, with every
address verified to be a real function entry point. See
`docs/agent-notes/signature-derivation.md` for the method and its limits.

At startup the mod loads `offsets_<game_version>.json` beside itself, **verifies
its `so_sha256` against the running `libminecraftpe.so`**, and uses it only on a
match. A missing, unparseable or mismatched map falls back to
`FlatWorldGenerator` meshing - the fail-closed default. `src/offsets.rs` owns
that logic and is host-tested.

`entity_seam_offsets` is reported **unresolved on purpose**: those are field
offsets inside a live `Level`/`Actor`, which need a disassembly pass, not a
signature scan. They are not guessed.

## Layout

```
src/
  lib.rs        crate root and the packed version
  math.rs       SIMD 3-D math: AABB, ray casts, swept collision, frustum culling
  mesher.rs     binary greedy meshing with baked ambient occlusion
  parallel.rs   a bounded scoped worker pool for off-thread meshing
  ring.rs       a bounded lock-free MPMC ring buffer
  pipeline.rs   the work loop: submit -> worker -> ring -> drain
  world.rs      a layered world description turned into a meshed volume
  offsets.rs    per-build offset map loader (fail-closed)
  ffi.rs        the frozen C ABI (gbr_*)
cpp/
  glowberry_rust_abi.h    the ABI header, single source of truth for both sides
  ffi_smoke_test.cpp      a C++ test that links the core and calls the ABI
  run_host_tests.sh       rustfmt + clippy + cargo test + the C++ smoke test
tools/
  download_and_extract.py APK download + libminecraftpe.so extraction
  extract_offsets.py      RTTI/signature offset extraction
  tests/                  unit tests for the pure logic
offsets/
  offsets_26_52.json      per-build offset maps
  offsets_26_60.json
  offsets_26_62.json
mod/
  GlowberryRustMod.cpp    the preloader mod: lifecycle, config, load self-test
  CMakeLists.txt          cross-compiles the mod for arm64-v8a
  manifest.json           the mod manifest the client reads
  glowberry_rust.map      the linker version script (exports PLGetModRegistration + gbr_*)
```

## The modules

- **`math`** - `Aabb` overlap, `ray_aabb`, `resolve_movement` (a swept,
  time-of-impact resolver that cannot tunnel through a thin wall at speed), and
  `Frustum::contains_aabb`. Built on `glam::Vec3A`, a 16-byte SIMD vector that
  lowers to NEON on aarch64 and SSE2 on x86_64.
- **`mesher`** - Mikola Lysenko's greedy mesher, binary variant. Coplanar faces
  with the same block, facing and baked AO merge into one quad, so a solid 16³
  chunk meshes to six quads instead of 1536. Per-vertex ambient occlusion is baked
  in and quads only merge when their AO signature matches.
- **`parallel`** - a *bounded* pool, deliberately not rayon: an explicit worker
  count, a scope per batch, and no threads that outlive the call. A global
  one-thread-per-core pool living in the game process would compete with the
  game's own render and audio threads on exactly the low-end devices this targets.
- **`ring`** - a bounded Vyukov MPMC queue. Producers (mesh workers) never block
  the consumer (render thread); a full ring returns the value so the worker can
  drop or defer, rather than growing without bound and turning a frame hitch into
  an OOM kill.
- **`pipeline`** - the loop the core exists for: a bounded pool of worker threads
  that pull chunk jobs off a ring, mesh them, and push the finished meshes onto a
  result ring the consumer drains per frame. A full result ring drops and counts
  rather than blocking.
- **`world`** - turns a footprint of block layers (the client's
  `FlatWorldGenerator` shape) into a voxel volume, meshes it whole, or splits it
  into chunks that feed the pipeline.

## Building

Requires a Rust toolchain and, for the mod, the Android NDK and a
`preloader-android` checkout.

```sh
# The core, and its tests:
cargo test
bash cpp/run_host_tests.sh

# The mod, for arm64-v8a:
cargo build --release --target aarch64-linux-android
cmake -B build/mod \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28 \
  -DPRELOADER_ANDROID_ROOT=/path/to/preloader-android mod
cmake --build build/mod
```

`arm64-v8a` is the only ABI. The client ships a single arm64 APK, and the
preloader refuses any other ABI, so a 32-bit build would be an artifact the client
could never load.

## Tests

`bash cpp/run_host_tests.sh` runs the Rust unit tests (SIMD math, mesher, ring,
FFI), `rustfmt` and `clippy -D warnings`, then compiles `cpp/ffi_smoke_test.cpp`
with `g++` against the built core and runs it. That last step is the one that
matters most: it is C++ calling the `extern "C"` symbols and reading the
`#[repr(C)]` structs, which is the only way to catch an ABI or layout mistake.

## CI

`.github/workflows/build.yml` runs the host tests, cross-compiles the mod for
arm64-v8a, verifies the `.so` (ELF shared object, AArch64, no glibc references,
and both `PLGetModRegistration` and `gbr_version` exported), and packages the mod
with its manifest. A `v*` tag publishes a Release.

## Licence

Apache-2.0.