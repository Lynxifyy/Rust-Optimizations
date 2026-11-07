// The frozen C ABI of the Rust core (crate `glowberry_rust`).
//
// This header is the contract between the Rust core and any C/C++ consumer. It
// is the single source of truth for the struct layouts and the symbol names, and
// it is what both `cpp/ffi_smoke_test.cpp` and the preloader mod wrapper include.
// Every layout here is mirrored by a `#[repr(C)]` type in `src/ffi.rs`; if the
// two ever disagree, the C++ smoke test fails and so does the mod.
//
// Ownership rules, repeated from `src/ffi.rs`:
//  - A function returning `GbrMesh` transfers ownership of the vertex and index
//    buffers to the caller. Call `gbr_mesh_free` exactly once per mesh.
//  - A ring is owned from `gbr_ring_new` until `gbr_ring_free`. A mesh popped
//    from a ring is owned by the caller afterwards.
//  - Every function tolerates a null pointer by returning a failure code.

#ifndef GLOWBERRY_RUST_ABI_H
#define GLOWBERRY_RUST_ABI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct GbrVec3 {
  float x;
  float y;
  float z;
};

struct GbrAabb {
  struct GbrVec3 min;
  struct GbrVec3 max;
};

// One mesh vertex. The GL attribute layout is position (3xf32), normal (3xi8),
// uv (2xf32), ao (u8), tint (u8); the struct is padded to 4-byte alignment.
struct GbrVertex {
  float pos[3];
  int8_t normal[3];
  float uv[2];
  uint8_t ao;
  uint8_t tint;
};

struct GbrMesh {
  struct GbrVertex *vertices;
  size_t vertex_count;
  uint32_t *indices;
  size_t index_count;
};

struct GbrRing; // opaque
struct GbrPipeline; // opaque

// One horizontal block layer, matching the client's FlatWorldGenerator model.
struct GbrBlockLayer {
  uint8_t block;
  uint32_t count;
};

// Pipeline counters. All monotonic.
struct GbrPipelineStats {
  uint64_t submitted;
  uint64_t meshed;
  uint64_t drained;
  uint64_t dropped;
  uint64_t rejected;
};

// The core version, packed as 0x00MMmmpp.
uint32_t gbr_version(void);

// Meshes one chunk of x * y * z blocks (x-fastest order). Free with gbr_mesh_free.
struct GbrMesh gbr_mesh_blocks(const uint8_t *blocks, uint32_t x, uint32_t y,
                               uint32_t z);

// Meshes `count` chunks across `workers` threads into `out[0..count]`. `dims`
// holds three uint32 per chunk (x, y, z). Returns 0 on success, -1 on a null
// argument. Each written mesh must be freed with gbr_mesh_free.
int gbr_mesh_batch(const uint8_t *const *blocks, const uint32_t *dims,
                   uint32_t count, uint32_t workers, struct GbrMesh *out);

// Frees a mesh produced by this core. Null is accepted.
void gbr_mesh_free(struct GbrMesh mesh);

// A bounded lock-free ring. Capacity rounds up to a power of two (minimum 2).
struct GbrRing *gbr_ring_new(uint32_t capacity);
void gbr_ring_free(struct GbrRing *ring);

// Meshes `blocks` and pushes the result. Returns 0 on success, 1 when the ring is
// full (the mesh is dropped, not leaked), -1 on a null argument.
int gbr_ring_push_mesh(struct GbrRing *ring, uint64_t chunk_id,
                       const uint8_t *blocks, uint32_t x, uint32_t y, uint32_t z);

// Pops one mesh. Returns 0 on success (free the mesh), 1 when empty, -1 on a null
// argument. On success `*out_chunk_id` receives the chunk id.
int gbr_ring_pop_mesh(struct GbrRing *ring, struct GbrMesh *out,
                      uint64_t *out_chunk_id);

// The best-effort number of meshes waiting in the ring.
size_t gbr_ring_len(const struct GbrRing *ring);

// Resolves a moving AABB against static solids. `flags` (optional) receives a
// bitmask: 1 = hit x, 2 = hit y, 4 = hit z, 8 = on ground. Returns 0 on success,
// -1 on a null required pointer.
int gbr_resolve_movement(const struct GbrAabb *start, const struct GbrVec3 *delta,
                         const struct GbrAabb *solids, uint32_t count,
                         struct GbrAabb *out, uint32_t *flags);

// Frustum-culls `count` boxes against a column-major 4x4 view-projection matrix
// (16 floats, `Mat4::from_cols_array` order). `out_visible[i]` is set to 1 or 0.
// Returns the number of visible boxes, or 0 when any pointer is null.
uint32_t gbr_frustum_cull(const float *mat, const struct GbrAabb *boxes,
                          uint8_t *out_visible, uint32_t count);

// --- The meshing pipeline (ring + worker threads) --------------------------

// Creates a pipeline with `workers` worker threads. `job_capacity` bounds
// not-yet-meshed jobs; `result_capacity` bounds finished meshes waiting to be
// drained. Returns null when `workers` or either capacity is zero.
struct GbrPipeline *gbr_pipeline_new(uint32_t workers, uint32_t job_capacity,
                                     uint32_t result_capacity);
void gbr_pipeline_free(struct GbrPipeline *pipeline);

// Submits a chunk to be meshed. Returns 0 on success, 1 when the pipeline is full
// or stopped (back-pressure), -1 on a null argument.
int gbr_pipeline_submit(struct GbrPipeline *pipeline, uint64_t chunk_id,
                        const uint8_t *blocks, uint32_t x, uint32_t y, uint32_t z);

// Drains up to `max` finished meshes. Returns the number written (0 when nothing
// is ready), or -1 on a null argument. Free each mesh with gbr_mesh_free.
int gbr_pipeline_drain(struct GbrPipeline *pipeline, struct GbrMesh *out_meshes,
                       uint64_t *out_ids, uint32_t max);

// Writes the pipeline counters to `*out`. Returns 0, or -1 on a null argument.
int gbr_pipeline_stats(const struct GbrPipeline *pipeline,
                       struct GbrPipelineStats *out);

// The best-effort number of finished meshes waiting to be drained.
size_t gbr_pipeline_pending(const struct GbrPipeline *pipeline);

// Meshes a layered world directly: a `width x depth` footprint of `layers`
// stacked bottom-up (the shape the client's FlatWorldGenerator produces). Returns
// an owned mesh; free it with gbr_mesh_free.
struct GbrMesh gbr_mesh_world_layers(const struct GbrBlockLayer *layers,
                                     uint32_t count, uint32_t width,
                                     uint32_t depth);

// --- Per-build offset maps -------------------------------------------------

struct GbrOffsetMap; // opaque

// Parses an offset map from JSON. Returns null on a parse error or bad UTF-8.
struct GbrOffsetMap *gbr_offsets_load(const uint8_t *json, size_t len);
void gbr_offsets_free(struct GbrOffsetMap *map);

// The function address of a resolved RTTI slot, or 0 when the map has none.
uint64_t gbr_offsets_slot(const struct GbrOffsetMap *map, const uint8_t *type_name,
                          size_t len, uint32_t slot);

// 1 when the map describes the given libminecraftpe.so build id (hex sha256),
// else 0. This is the fail-closed gate: a mismatched map must not be used.
uint32_t gbr_offsets_verify_build(const struct GbrOffsetMap *map,
                                  const uint8_t *sha256, size_t len);

// 1 when the entity seam is fully recovered, else 0.
uint32_t gbr_offsets_entity_seam_complete(const struct GbrOffsetMap *map);

// 1 when the map carries everything the entity-list walk needs, else 0.
uint32_t gbr_offsets_entity_walk_ready(const struct GbrOffsetMap *map);

// Walks the live entity list from a Level*. Writes up to `capacity` Actor*
// pointers into `out_actors` and returns the count (0 when refused or empty), or
// -1 on a null argument. Reads game memory: call on the game thread with a
// Level* the game returned.
int gbr_entity_walk(const struct GbrOffsetMap *map, const uint8_t *level,
                    const uint8_t **out_actors, uint32_t capacity);

// Reads one entity's position (Vec3) into out[0..3]. Returns 0 on success, 1
// when the read is refused (null actor, bad offset, non-finite), -1 on null.
int gbr_entity_position(const struct GbrOffsetMap *map, const uint8_t *actor,
                        float *out);

// Copies the map's version string into `out` (NUL-terminated, truncated to cap).
// Returns the length written, excluding the NUL.
size_t gbr_offsets_version(const struct GbrOffsetMap *map, uint8_t *out, size_t cap);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // GLOWBERRY_RUST_ABI_H
