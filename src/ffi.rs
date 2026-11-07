//! The C ABI the preloader C++ shim links against.
//!
//! Rust and C++ do not share a stable ABI, so the C++ side never calls a Rust
//! symbol by its mangled name. It calls these `extern "C"` functions, whose
//! names and layouts are frozen. Everything here is `unsafe` at the boundary:
//! the caller must pass valid pointers and free every mesh it receives with
//! [`gbr_mesh_free`].
//!
//! Ownership rules, stated once:
//! - A function that returns a [`GbrMesh`] transfers ownership of its vertex and
//!   index buffers to the caller. The caller must call [`gbr_mesh_free`] exactly
//!   once per mesh.
//! - A ring is owned by the caller from [`gbr_ring_new`] until [`gbr_ring_free`].
//!   A mesh popped from a ring is owned by the caller afterwards.
//! - Every function tolerates a null pointer by returning a failure code rather
//!   than dereferencing it.

use std::os::raw::c_int;

use crate::math::{self, Aabb};
use crate::mesher::{self, ChunkDims, Mesh, Vertex};
use crate::parallel::{self, ChunkJob};
use crate::pipeline::{MeshJob, MeshPipeline};
use crate::ring::Ring;

/// A 3-vector, layout-compatible with the C++ `GbrVec3`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GbrVec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

/// An axis-aligned box, layout-compatible with the C++ `GbrAabb`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GbrAabb {
    pub min: GbrVec3,
    pub max: GbrVec3,
}

/// One mesh vertex. Layout matches the C++ `GbrVertex` and the GL attribute
/// layout: position (3xf32), normal (3xi8), uv (2xf32), ao (u8), tint (u8).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GbrVertex {
    pub pos: [f32; 3],
    pub normal: [i8; 3],
    pub uv: [f32; 2],
    pub ao: u8,
    pub tint: u8,
}

/// A meshed chunk handed to C++. Free with [`gbr_mesh_free`].
#[repr(C)]
#[derive(Debug)]
pub struct GbrMesh {
    pub vertices: *mut GbrVertex,
    pub vertex_count: usize,
    pub indices: *mut u32,
    pub index_count: usize,
}

impl GbrMesh {
    const fn empty() -> Self {
        Self {
            vertices: std::ptr::null_mut(),
            vertex_count: 0,
            indices: std::ptr::null_mut(),
            index_count: 0,
        }
    }
}

/// The opaque ring handle.
pub struct GbrRing {
    inner: Ring<MeshPayload>,
}

/// A mesh tagged with the chunk it came from.
pub struct MeshPayload {
    pub chunk_id: u64,
    pub mesh: Mesh,
}

/// Moves a `Vec<T>` into a raw pointer + length, keeping the exact allocation.
fn vec_into_raw<T>(v: Vec<T>) -> (*mut T, usize) {
    let boxed = v.into_boxed_slice();
    let len = boxed.len();
    (Box::into_raw(boxed) as *mut T, len)
}

/// Reclaims a pointer produced by [`vec_into_raw`].
///
/// # Safety
/// `ptr`/`len` must come from [`vec_into_raw`] and not have been freed already.
unsafe fn raw_into_vec<T>(ptr: *mut T, len: usize) -> Vec<T> {
    if ptr.is_null() {
        return Vec::new();
    }
    Box::from_raw(std::ptr::slice_from_raw_parts_mut(ptr, len)).into_vec()
}

/// Converts an owned [`Mesh`] into a [`GbrMesh`], transferring the buffers.
fn mesh_into_ffi(mesh: Mesh) -> GbrMesh {
    let Mesh { vertices, indices } = mesh;
    let (vp, vc) = vec_into_raw(vertices);
    let (ip, ic) = vec_into_raw(indices);
    GbrMesh {
        vertices: vp as *mut GbrVertex,
        vertex_count: vc,
        indices: ip,
        index_count: ic,
    }
}

/// One horizontal block layer, layout-compatible with the C++ `GbrBlockLayer`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GbrBlockLayer {
    pub block: u8,
    pub count: u32,
}

/// The pipeline counters, layout-compatible with the C++ `GbrPipelineStats`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GbrPipelineStats {
    pub submitted: u64,
    pub meshed: u64,
    pub drained: u64,
    pub dropped: u64,
    pub rejected: u64,
}

/// The opaque pipeline handle.
pub struct GbrPipeline {
    inner: MeshPipeline,
}

/// Creates a meshing pipeline with `workers` worker threads.
///
/// `job_capacity` bounds not-yet-meshed jobs; `result_capacity` bounds finished
/// meshes waiting to be drained. Returns null when `workers` or either capacity
/// is zero. Free with [`gbr_pipeline_free`].
#[no_mangle]
pub extern "C" fn gbr_pipeline_new(
    workers: u32,
    job_capacity: u32,
    result_capacity: u32,
) -> *mut GbrPipeline {
    if workers == 0 || job_capacity == 0 || result_capacity == 0 {
        return std::ptr::null_mut();
    }
    Box::into_raw(Box::new(GbrPipeline {
        inner: MeshPipeline::new(
            workers as usize,
            job_capacity as usize,
            result_capacity as usize,
        ),
    }))
}

/// Stops and frees a pipeline. Null is accepted.
///
/// # Safety
/// `pipeline` must come from [`gbr_pipeline_new`] and not be freed before.
#[no_mangle]
pub unsafe extern "C" fn gbr_pipeline_free(pipeline: *mut GbrPipeline) {
    if !pipeline.is_null() {
        drop(Box::from_raw(pipeline));
    }
}

/// Submits a chunk to be meshed. Returns 0 on success, 1 when the pipeline is
/// full or stopped (a back-pressure signal), -1 on a null argument.
///
/// # Safety
/// `pipeline` must be valid and `blocks` must point to `x * y * z` bytes.
#[no_mangle]
pub unsafe extern "C" fn gbr_pipeline_submit(
    pipeline: *mut GbrPipeline,
    chunk_id: u64,
    blocks: *const u8,
    x: u32,
    y: u32,
    z: u32,
) -> c_int {
    if pipeline.is_null() || blocks.is_null() || x == 0 || y == 0 || z == 0 {
        return -1;
    }
    let dims = ChunkDims::new(x as usize, y as usize, z as usize);
    let slice = std::slice::from_raw_parts(blocks, dims.volume());
    let job = MeshJob {
        chunk_id,
        blocks: slice.to_vec(),
        dims,
    };
    if (*pipeline).inner.submit(job) {
        0
    } else {
        1
    }
}

/// Drains up to `max` finished meshes into `out_meshes`/`out_ids`.
///
/// Returns the number written (0 when nothing is ready), or -1 on a null
/// argument. Each written mesh must be freed with [`gbr_mesh_free`].
///
/// # Safety
/// `out_meshes` and `out_ids` must have room for `max` items.
#[no_mangle]
pub unsafe extern "C" fn gbr_pipeline_drain(
    pipeline: *mut GbrPipeline,
    out_meshes: *mut GbrMesh,
    out_ids: *mut u64,
    max: u32,
) -> c_int {
    if pipeline.is_null() || out_meshes.is_null() || out_ids.is_null() {
        return -1;
    }
    if max == 0 {
        return 0;
    }
    let results = (*pipeline).inner.drain(max as usize);
    let meshes = std::slice::from_raw_parts_mut(out_meshes, max as usize);
    let ids = std::slice::from_raw_parts_mut(out_ids, max as usize);
    let n = results.len();
    for (i, result) in results.into_iter().enumerate() {
        meshes[i] = mesh_into_ffi(result.mesh);
        ids[i] = result.chunk_id;
    }
    n as c_int
}

/// Writes the pipeline counters to `*out`. Returns 0, or -1 on a null argument.
///
/// # Safety
/// `pipeline` and `out` must be valid.
#[no_mangle]
pub unsafe extern "C" fn gbr_pipeline_stats(
    pipeline: *const GbrPipeline,
    out: *mut GbrPipelineStats,
) -> c_int {
    if pipeline.is_null() || out.is_null() {
        return -1;
    }
    let s = (*pipeline).inner.stats();
    *out = GbrPipelineStats {
        submitted: s.submitted,
        meshed: s.meshed,
        drained: s.drained,
        dropped: s.dropped,
        rejected: s.rejected,
    };
    0
}

/// The best-effort number of finished meshes waiting to be drained.
///
/// # Safety
/// `pipeline` must be valid.
#[no_mangle]
pub unsafe extern "C" fn gbr_pipeline_pending(pipeline: *const GbrPipeline) -> usize {
    if pipeline.is_null() {
        return 0;
    }
    (*pipeline).inner.pending()
}

/// Meshes a layered world directly: a `width x depth` footprint of `layers`
/// stacked bottom-up (the shape the client's `FlatWorldGenerator` produces).
///
/// Returns an owned mesh; free it with [`gbr_mesh_free`]. Returns an empty mesh
/// on a null argument or an empty footprint.
///
/// # Safety
/// `layers` must point to `count` valid layer structs.
#[no_mangle]
pub unsafe extern "C" fn gbr_mesh_world_layers(
    layers: *const GbrBlockLayer,
    count: u32,
    width: u32,
    depth: u32,
) -> GbrMesh {
    if layers.is_null() || count == 0 || width == 0 || depth == 0 {
        return GbrMesh::empty();
    }
    let layer_slice = std::slice::from_raw_parts(layers, count as usize);
    let world = crate::world::LayeredWorld {
        width,
        depth,
        layers: layer_slice
            .iter()
            .map(|l| crate::world::BlockLayer {
                block: l.block,
                count: l.count,
            })
            .collect(),
    };
    mesh_into_ffi(world.to_mesh())
}

/// The opaque offset-map handle.
pub struct GbrOffsetMap {
    inner: crate::offsets::OffsetMap,
}

/// Parses an offset map from JSON. Returns null on a parse error.
///
/// # Safety
/// `json` must point to `len` readable bytes.
#[no_mangle]
pub unsafe extern "C" fn gbr_offsets_load(json: *const u8, len: usize) -> *mut GbrOffsetMap {
    if json.is_null() || len == 0 {
        return std::ptr::null_mut();
    }
    let bytes = std::slice::from_raw_parts(json, len);
    let Ok(text) = std::str::from_utf8(bytes) else {
        return std::ptr::null_mut();
    };
    match crate::offsets::OffsetMap::parse(text) {
        Ok(inner) => Box::into_raw(Box::new(GbrOffsetMap { inner })),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Frees an offset map. Null is accepted.
///
/// # Safety
/// `map` must come from [`gbr_offsets_load`] and not be freed before.
#[no_mangle]
pub unsafe extern "C" fn gbr_offsets_free(map: *mut GbrOffsetMap) {
    if !map.is_null() {
        drop(Box::from_raw(map));
    }
}

/// The function address of a resolved slot, or 0 when the map has none.
///
/// # Safety
/// `map` must be valid and `type_name` must point to `len` readable bytes.
#[no_mangle]
pub unsafe extern "C" fn gbr_offsets_slot(
    map: *const GbrOffsetMap,
    type_name: *const u8,
    len: usize,
    slot: u32,
) -> u64 {
    if map.is_null() || type_name.is_null() {
        return 0;
    }
    let bytes = std::slice::from_raw_parts(type_name, len);
    let Ok(name) = std::str::from_utf8(bytes) else {
        return 0;
    };
    (*map).inner.slot(name, slot).unwrap_or(0)
}

/// 1 when the map describes the given `libminecraftpe.so` build id, else 0.
///
/// # Safety
/// `map` must be valid and `sha256` must point to `len` readable bytes.
#[no_mangle]
pub unsafe extern "C" fn gbr_offsets_verify_build(
    map: *const GbrOffsetMap,
    sha256: *const u8,
    len: usize,
) -> u32 {
    if map.is_null() || sha256.is_null() {
        return 0;
    }
    let bytes = std::slice::from_raw_parts(sha256, len);
    let Ok(text) = std::str::from_utf8(bytes) else {
        return 0;
    };
    u32::from((*map).inner.verify_build(text).is_ok())
}

/// 1 when the entity seam is fully recovered, else 0.
///
/// # Safety
/// `map` must be valid.
#[no_mangle]
pub unsafe extern "C" fn gbr_offsets_entity_seam_complete(map: *const GbrOffsetMap) -> u32 {
    if map.is_null() {
        return 0;
    }
    u32::from((*map).inner.entity_seam_complete())
}

/// 1 when the map carries everything the entity-list walk needs, else 0.
///
/// # Safety
/// `map` must be valid.
#[no_mangle]
pub unsafe extern "C" fn gbr_offsets_entity_walk_ready(map: *const GbrOffsetMap) -> u32 {
    if map.is_null() {
        return 0;
    }
    u32::from(crate::entities::EntitySeam::from_map(&(*map).inner).is_some())
}

/// Walks the live entity list from a `Level*` and writes the `Actor*` pointers.
///
/// Returns the number of actors written (0 when the walk is refused or the list
/// is empty), or -1 on a null argument. `out_actors` receives up to `capacity`
/// pointers.
///
/// # Safety
/// `map` must be valid and carry the entity-seam offsets; `level` must be a live
/// `Level*` the game returned, read on the game thread. `out_actors` must have
/// room for `capacity` pointers.
#[no_mangle]
pub unsafe extern "C" fn gbr_entity_walk(
    map: *const GbrOffsetMap,
    level: *const u8,
    out_actors: *mut *const u8,
    capacity: u32,
) -> c_int {
    if map.is_null() || level.is_null() || out_actors.is_null() {
        return -1;
    }
    let Some(seam) = crate::entities::EntitySeam::from_map(&(*map).inner) else {
        return 0;
    };
    let Some(range) = crate::entities::read_entity_range(level, &seam) else {
        return 0;
    };
    let total = range.len().min(capacity as usize);
    let out = std::slice::from_raw_parts_mut(out_actors, capacity as usize);
    let mut written = 0usize;
    for i in 0..total {
        match range.actor_at(i) {
            Some(actor) => {
                out[written] = actor;
                written += 1;
            }
            None => continue,
        }
    }
    written as c_int
}

/// Reads one entity's position (`Vec3`) into `out[0..3]`.
///
/// Returns 0 on success, 1 when the read is refused (null actor, bad offset, or
/// a non-finite value), -1 on a null argument.
///
/// # Safety
/// `map` must be valid; `actor` must be a live `Actor*`; `out` must have room
/// for three floats.
#[no_mangle]
pub unsafe extern "C" fn gbr_entity_position(
    map: *const GbrOffsetMap,
    actor: *const u8,
    out: *mut f32,
) -> c_int {
    if map.is_null() || actor.is_null() || out.is_null() {
        return -1;
    }
    let Some(seam) = crate::entities::EntitySeam::from_map(&(*map).inner) else {
        return 1;
    };
    match crate::entities::read_entity_position(actor, seam.position_offset) {
        Some(position) => {
            let dst = std::slice::from_raw_parts_mut(out, 3);
            dst.copy_from_slice(&position);
            0
        }
        None => 1,
    }
}

/// Copies the map's version string into `out` (NUL-terminated, truncated to
/// `cap`). Returns the length written, excluding the NUL. 0 on a null argument.
///
/// # Safety
/// `out` must point to `cap` writable bytes.
#[no_mangle]
pub unsafe extern "C" fn gbr_offsets_version(
    map: *const GbrOffsetMap,
    out: *mut u8,
    cap: usize,
) -> usize {
    if map.is_null() || out.is_null() || cap == 0 {
        return 0;
    }
    let version = (*map).inner.version.as_bytes();
    let n = version.len().min(cap - 1);
    std::ptr::copy_nonoverlapping(version.as_ptr(), out, n);
    *out.add(n) = 0;
    n
}

/// The core version, packed as `0x00MMmmpp`.
#[no_mangle]
pub extern "C" fn gbr_version() -> u32 {
    crate::version()
}

/// Meshes one chunk of `x * y * z` blocks (`blocks` in x-fastest order).
///
/// Returns an owned mesh; free it with [`gbr_mesh_free`]. Returns an empty mesh
/// (null buffers) when `blocks` is null or a dimension is zero.
///
/// # Safety
/// `blocks` must point to at least `x * y * z` readable bytes, or be null.
#[no_mangle]
pub unsafe extern "C" fn gbr_mesh_blocks(blocks: *const u8, x: u32, y: u32, z: u32) -> GbrMesh {
    if blocks.is_null() || x == 0 || y == 0 || z == 0 {
        return GbrMesh::empty();
    }
    let dims = ChunkDims::new(x as usize, y as usize, z as usize);
    let slice = std::slice::from_raw_parts(blocks, dims.volume());
    mesh_into_ffi(mesher::mesh_chunk_blocks(slice, dims))
}

/// Frees a mesh produced by this module. Null is accepted and ignored.
///
/// # Safety
/// The mesh must have been produced here and not freed before.
#[no_mangle]
pub unsafe extern "C" fn gbr_mesh_free(mesh: GbrMesh) {
    drop(raw_into_vec(
        mesh.vertices as *mut Vertex,
        mesh.vertex_count,
    ));
    drop(raw_into_vec(mesh.indices, mesh.index_count));
}

/// Meshes `count` chunks across `workers` threads.
///
/// `blocks[i]` points at chunk `i`'s blocks; `dims` holds three `u32` per chunk
/// (`x, y, z`). `out` must point to `count` writable [`GbrMesh`] slots. Each
/// written mesh must be freed with [`gbr_mesh_free`]. Returns 0 on success, -1 on
/// a null argument.
///
/// # Safety
/// All pointers must be valid for the stated counts.
#[no_mangle]
pub unsafe extern "C" fn gbr_mesh_batch(
    blocks: *const *const u8,
    dims: *const u32,
    count: u32,
    workers: u32,
    out: *mut GbrMesh,
) -> c_int {
    if blocks.is_null() || dims.is_null() || out.is_null() {
        return -1;
    }
    if count == 0 {
        return 0;
    }
    let count = count as usize;
    let block_ptrs = std::slice::from_raw_parts(blocks, count);
    let dim_slice = std::slice::from_raw_parts(dims, count * 3);

    // Own the block buffers as slices for the duration of the call.
    let mut owned: Vec<&[u8]> = Vec::with_capacity(count);
    let mut chunk_dims = Vec::with_capacity(count);
    for i in 0..count {
        let d = ChunkDims::new(
            dim_slice[i * 3] as usize,
            dim_slice[i * 3 + 1] as usize,
            dim_slice[i * 3 + 2] as usize,
        );
        if block_ptrs[i].is_null() || d.volume() == 0 {
            return -1;
        }
        owned.push(std::slice::from_raw_parts(block_ptrs[i], d.volume()));
        chunk_dims.push(d);
    }

    let jobs: Vec<ChunkJob> = owned
        .iter()
        .zip(chunk_dims.iter())
        .map(|(b, d)| ChunkJob {
            blocks: b,
            dims: *d,
        })
        .collect();

    let meshes = parallel::mesh_chunks_parallel(&jobs, workers as usize);
    let out_slice = std::slice::from_raw_parts_mut(out, count);
    for (slot, mesh) in out_slice.iter_mut().zip(meshes) {
        *slot = mesh_into_ffi(mesh);
    }
    0
}

/// Creates a ring that holds up to `capacity` meshes (rounded up to a power of
/// two). Returns null when `capacity` is 0. Free with [`gbr_ring_free`].
#[no_mangle]
pub extern "C" fn gbr_ring_new(capacity: u32) -> *mut GbrRing {
    if capacity == 0 {
        return std::ptr::null_mut();
    }
    Box::into_raw(Box::new(GbrRing {
        inner: Ring::with_capacity(capacity as usize),
    }))
}

/// Frees a ring. Null is accepted. Any meshes still in the ring are dropped.
///
/// # Safety
/// `ring` must come from [`gbr_ring_new`] and not be freed before.
#[no_mangle]
pub unsafe extern "C" fn gbr_ring_free(ring: *mut GbrRing) {
    if !ring.is_null() {
        drop(Box::from_raw(ring));
    }
}

/// Meshes `blocks` and pushes the result onto the ring.
///
/// Returns 0 on success, 1 when the ring is full (the mesh is dropped, not
/// leaked), -1 on a null argument.
///
/// # Safety
/// `ring` must be valid and `blocks` must point to `x * y * z` bytes.
#[no_mangle]
pub unsafe extern "C" fn gbr_ring_push_mesh(
    ring: *mut GbrRing,
    chunk_id: u64,
    blocks: *const u8,
    x: u32,
    y: u32,
    z: u32,
) -> c_int {
    if ring.is_null() || blocks.is_null() || x == 0 || y == 0 || z == 0 {
        return -1;
    }
    let dims = ChunkDims::new(x as usize, y as usize, z as usize);
    let slice = std::slice::from_raw_parts(blocks, dims.volume());
    let mesh = mesher::mesh_chunk_blocks(slice, dims);
    let payload = MeshPayload { chunk_id, mesh };
    match (*ring).inner.push(payload) {
        Ok(()) => 0,
        Err(_) => 1,
    }
}

/// Pops one mesh from the ring into `*out` and reports the chunk id.
///
/// Returns 0 when a mesh was written (free it with [`gbr_mesh_free`]), 1 when the
/// ring is empty, -1 on a null argument. On success `*out_chunk_id` receives the
/// chunk id and `*out` receives the mesh.
///
/// # Safety
/// `ring`, `out`, and `out_chunk_id` must be valid.
#[no_mangle]
pub unsafe extern "C" fn gbr_ring_pop_mesh(
    ring: *mut GbrRing,
    out: *mut GbrMesh,
    out_chunk_id: *mut u64,
) -> c_int {
    if ring.is_null() || out.is_null() || out_chunk_id.is_null() {
        return -1;
    }
    match (*ring).inner.pop() {
        Some(payload) => {
            *out = mesh_into_ffi(payload.mesh);
            *out_chunk_id = payload.chunk_id;
            0
        }
        None => 1,
    }
}

/// The best-effort number of meshes waiting in the ring.
///
/// # Safety
/// `ring` must be valid.
#[no_mangle]
pub unsafe extern "C" fn gbr_ring_len(ring: *const GbrRing) -> usize {
    if ring.is_null() {
        return 0;
    }
    (*ring).inner.len()
}

fn to_aabb(b: &GbrAabb) -> Aabb {
    Aabb::new(
        math::Vec3::new(b.min.x, b.min.y, b.min.z),
        math::Vec3::new(b.max.x, b.max.y, b.max.z),
    )
}

/// Resolves a moving AABB against static solids, writing the final box to `*out`.
///
/// `flags` (optional) receives a bitmask: 1 = hit x, 2 = hit y, 4 = hit z,
/// 8 = on ground. Returns 0 on success, -1 on a null required pointer.
///
/// # Safety
/// `solids` must point to `count` valid boxes; `start`, `delta`, and `out` valid.
#[no_mangle]
pub unsafe extern "C" fn gbr_resolve_movement(
    start: *const GbrAabb,
    delta: *const GbrVec3,
    solids: *const GbrAabb,
    count: u32,
    out: *mut GbrAabb,
    flags: *mut u32,
) -> c_int {
    if start.is_null() || delta.is_null() || out.is_null() {
        return -1;
    }
    let start_aabb = to_aabb(&*start);
    let d = math::Vec3::new((*delta).x, (*delta).y, (*delta).z);
    let solids_v: Vec<Aabb> = if solids.is_null() || count == 0 {
        Vec::new()
    } else {
        std::slice::from_raw_parts(solids, count as usize)
            .iter()
            .map(to_aabb)
            .collect()
    };
    let r = math::resolve_movement(start_aabb, d, &solids_v);
    *out = GbrAabb {
        min: GbrVec3 {
            x: r.aabb.min.x,
            y: r.aabb.min.y,
            z: r.aabb.min.z,
        },
        max: GbrVec3 {
            x: r.aabb.max.x,
            y: r.aabb.max.y,
            z: r.aabb.max.z,
        },
    };
    if !flags.is_null() {
        let mut f = 0u32;
        if r.hit_x {
            f |= 1;
        }
        if r.hit_y {
            f |= 2;
        }
        if r.hit_z {
            f |= 4;
        }
        if r.on_ground {
            f |= 8;
        }
        *flags = f;
    }
    0
}

/// Frustum-culls `count` boxes against a column-major 4x4 view-projection matrix
/// (`mat` holds 16 floats in the order `Mat4::from_cols_array` produces).
///
/// `out_visible[i]` is set to 1 or 0. Returns the number of visible boxes, or 0
/// when any pointer is null.
///
/// # Safety
/// `mat` must point to 16 floats; `boxes` and `out_visible` to `count` items.
#[no_mangle]
pub unsafe extern "C" fn gbr_frustum_cull(
    mat: *const f32,
    boxes: *const GbrAabb,
    out_visible: *mut u8,
    count: u32,
) -> u32 {
    if mat.is_null() || boxes.is_null() || out_visible.is_null() || count == 0 {
        return 0;
    }
    let arr = *(mat as *const [f32; 16]);
    let m = glam::Mat4::from_cols_array(&arr);
    let frustum = math::Frustum::from_view_projection(m);
    let boxes = std::slice::from_raw_parts(boxes, count as usize);
    let out = std::slice::from_raw_parts_mut(out_visible, count as usize);
    let mut visible = 0u32;
    for (b, o) in boxes.iter().zip(out.iter_mut()) {
        let aabb = to_aabb(b);
        let v = frustum.contains_aabb(&aabb);
        *o = v as u8;
        if v {
            visible += 1;
        }
    }
    visible
}

#[cfg(test)]
mod tests {
    // Index expressions like `x + X * (y + Y * z)` spell out the x-fastest block
    // layout on purpose; clippy reads the constant factors as no-ops.
    #![allow(clippy::identity_op, clippy::erasing_op)]

    use super::*;

    #[test]
    fn version_matches_core() {
        assert_eq!(gbr_version(), crate::version());
    }

    #[test]
    fn mesh_blocks_round_trips_and_frees() {
        let dims = ChunkDims::new(4, 4, 4);
        let mut blocks = vec![0u8; dims.volume()];
        blocks[1 + 4 * (1 + 4 * 1)] = 1;
        unsafe {
            let mesh = gbr_mesh_blocks(blocks.as_ptr(), 4, 4, 4);
            assert_eq!(mesh.vertex_count, 24);
            assert_eq!(mesh.index_count, 36);
            assert!(!mesh.vertices.is_null());
            gbr_mesh_free(mesh);
        }
    }

    #[test]
    fn mesh_blocks_null_is_empty() {
        unsafe {
            let mesh = gbr_mesh_blocks(std::ptr::null(), 4, 4, 4);
            assert_eq!(mesh.vertex_count, 0);
            assert!(mesh.vertices.is_null());
            gbr_mesh_free(mesh);
        }
    }

    #[test]
    fn mesh_batch_parallel_and_free() {
        let dims = ChunkDims::cube_16();
        let buffers: Vec<Vec<u8>> = (0..4).map(|_| vec![1u8; dims.volume()]).collect();
        let ptrs: Vec<*const u8> = buffers.iter().map(|b| b.as_ptr()).collect();
        let dims_flat: Vec<u32> = vec![16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16];
        let mut out: Vec<GbrMesh> = (0..4).map(|_| GbrMesh::empty()).collect();
        unsafe {
            let rc = gbr_mesh_batch(ptrs.as_ptr(), dims_flat.as_ptr(), 4, 4, out.as_mut_ptr());
            assert_eq!(rc, 0);
            for m in out {
                assert_eq!(m.vertex_count, 24);
                gbr_mesh_free(m);
            }
        }
    }

    #[test]
    fn ring_push_pop_round_trip() {
        let dims = ChunkDims::new(4, 4, 4);
        let mut blocks = vec![0u8; dims.volume()];
        blocks[1 + 4 * (1 + 4 * 1)] = 1;
        unsafe {
            let ring = gbr_ring_new(8);
            assert!(!ring.is_null());
            assert_eq!(gbr_ring_push_mesh(ring, 42, blocks.as_ptr(), 4, 4, 4), 0);
            assert_eq!(gbr_ring_len(ring), 1);

            let mut mesh = GbrMesh::empty();
            let mut id = 0u64;
            assert_eq!(gbr_ring_pop_mesh(ring, &mut mesh, &mut id), 0);
            assert_eq!(id, 42);
            assert_eq!(mesh.vertex_count, 24);
            gbr_mesh_free(mesh);

            // Empty now: a second pop must report empty, not hand back a stale mesh.
            let mut empty = GbrMesh::empty();
            assert_eq!(gbr_ring_pop_mesh(ring, &mut empty, &mut id), 1);
            gbr_ring_free(ring);
        }
    }

    #[test]
    fn ring_full_returns_one_not_leak() {
        let dims = ChunkDims::new(2, 2, 2);
        let blocks = vec![1u8; dims.volume()];
        unsafe {
            // Capacity 1 rounds up to 2 (a single slot cannot work), so two fit.
            let ring = gbr_ring_new(1);
            assert_eq!(gbr_ring_push_mesh(ring, 1, blocks.as_ptr(), 2, 2, 2), 0);
            assert_eq!(gbr_ring_push_mesh(ring, 2, blocks.as_ptr(), 2, 2, 2), 0);
            assert_eq!(gbr_ring_push_mesh(ring, 3, blocks.as_ptr(), 2, 2, 2), 1);
            let mut mesh = GbrMesh::empty();
            let mut id = 0u64;
            assert_eq!(gbr_ring_pop_mesh(ring, &mut mesh, &mut id), 0);
            assert_eq!(id, 1);
            gbr_mesh_free(mesh);
            gbr_ring_free(ring);
        }
    }

    #[test]
    fn null_arguments_are_refused() {
        unsafe {
            assert!(gbr_ring_new(0).is_null());
            assert_eq!(gbr_ring_len(std::ptr::null()), 0);
            gbr_ring_free(std::ptr::null_mut());
            assert_eq!(
                gbr_ring_push_mesh(std::ptr::null_mut(), 0, std::ptr::null(), 1, 1, 1),
                -1
            );
        }
    }

    #[test]
    fn resolve_movement_ffi_reports_flags() {
        let start = GbrAabb {
            min: GbrVec3 {
                x: -0.5,
                y: 4.5,
                z: -0.5,
            },
            max: GbrVec3 {
                x: 0.5,
                y: 5.5,
                z: 0.5,
            },
        };
        let delta = GbrVec3 {
            x: 0.0,
            y: -10.0,
            z: 0.0,
        };
        let floor = GbrAabb {
            min: GbrVec3 {
                x: -10.0,
                y: -1.0,
                z: -10.0,
            },
            max: GbrVec3 {
                x: 10.0,
                y: 0.0,
                z: 10.0,
            },
        };
        let mut out = GbrAabb::default();
        let mut flags = 0u32;
        unsafe {
            let rc = gbr_resolve_movement(&start, &delta, &floor, 1, &mut out, &mut flags);
            assert_eq!(rc, 0);
            assert_eq!(flags & 8, 8, "on_ground should be set");
            assert!(
                out.min.y >= 0.0 && out.min.y < 1e-3,
                "min.y = {}",
                out.min.y
            );
        }
    }

    #[test]
    fn pipeline_submit_drain_round_trip() {
        let dims = ChunkDims::new(4, 4, 4);
        let blocks = vec![1u8; dims.volume()];
        unsafe {
            let p = gbr_pipeline_new(2, 16, 16);
            assert!(!p.is_null());
            assert_eq!(gbr_pipeline_submit(p, 11, blocks.as_ptr(), 4, 4, 4), 0);

            let mut meshes: Vec<GbrMesh> = (0..8).map(|_| GbrMesh::empty()).collect();
            let mut ids = vec![0u64; 8];
            // Wait for the worker to finish.
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
            let mut n = 0;
            while n == 0 && std::time::Instant::now() < deadline {
                n = gbr_pipeline_drain(p, meshes.as_mut_ptr(), ids.as_mut_ptr(), 8);
                if n == 0 {
                    std::thread::yield_now();
                }
            }
            assert_eq!(n, 1);
            assert_eq!(ids[0], 11);
            assert_eq!(meshes[0].vertex_count, 24);
            for m in meshes {
                gbr_mesh_free(m);
            }

            let mut stats = GbrPipelineStats::default();
            assert_eq!(gbr_pipeline_stats(p, &mut stats), 0);
            assert_eq!(stats.submitted, 1);
            assert_eq!(stats.meshed, 1);
            assert_eq!(stats.drained, 1);

            gbr_pipeline_free(p);
        }
    }

    #[test]
    fn pipeline_null_is_refused() {
        unsafe {
            assert!(gbr_pipeline_new(0, 8, 8).is_null());
            assert!(gbr_pipeline_new(1, 0, 8).is_null());
            gbr_pipeline_free(std::ptr::null_mut());
            assert_eq!(gbr_pipeline_pending(std::ptr::null()), 0);
            let mut stats = GbrPipelineStats::default();
            assert_eq!(gbr_pipeline_stats(std::ptr::null(), &mut stats), -1);
        }
    }

    #[test]
    fn world_layers_ffi_meshes_a_slab() {
        let layers = [
            GbrBlockLayer { block: 1, count: 1 },
            GbrBlockLayer { block: 2, count: 2 },
        ];
        unsafe {
            let mesh = gbr_mesh_world_layers(layers.as_ptr(), 2, 8, 8);
            // An 8x3x8 opaque box, but the side faces cannot merge across the
            // material boundary at y=1, so each of the four sides is two quads:
            // 4*2 + top + bottom = 10 quads = 40 vertices.
            assert_eq!(mesh.vertex_count, 40);
            gbr_mesh_free(mesh);
            // Null is safe.
            let empty = gbr_mesh_world_layers(std::ptr::null(), 0, 8, 8);
            assert_eq!(empty.vertex_count, 0);
            gbr_mesh_free(empty);
        }
    }

    #[test]
    fn frustum_cull_ffi() {
        let proj = glam::Mat4::perspective_rh_gl(90f32.to_radians(), 1.0, 0.1, 100.0);
        let view = glam::Mat4::look_at_rh(glam::Vec3::ZERO, -glam::Vec3::Z, glam::Vec3::Y);
        let m = (proj * view).to_cols_array();
        let boxes = [
            GbrAabb {
                min: GbrVec3 {
                    x: -0.5,
                    y: -0.5,
                    z: -10.5,
                },
                max: GbrVec3 {
                    x: 0.5,
                    y: 0.5,
                    z: -9.5,
                },
            },
            GbrAabb {
                min: GbrVec3 {
                    x: -0.5,
                    y: -0.5,
                    z: 9.5,
                },
                max: GbrVec3 {
                    x: 0.5,
                    y: 0.5,
                    z: 10.5,
                },
            },
        ];
        let mut vis = [0u8; 2];
        unsafe {
            let n = gbr_frustum_cull(m.as_ptr(), boxes.as_ptr(), vis.as_mut_ptr(), 2);
            assert_eq!(n, 1);
            assert_eq!(vis[0], 1);
            assert_eq!(vis[1], 0);
        }
    }
}
