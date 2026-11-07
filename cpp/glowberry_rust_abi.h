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

#ifdef __cplusplus
} // extern "C"
#endif

#endif // GLOWBERRY_RUST_ABI_H
