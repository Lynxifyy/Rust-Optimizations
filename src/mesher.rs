//! Binary greedy voxel meshing with baked ambient occlusion.
//!
//! The classic greedy mesher (Mikola Lysenko, "Meshing in a Minecraft Game")
//! merges adjacent coplanar faces that share the same block, the same facing and
//! the same baked lighting into one large quad, so a flat 16x16 floor becomes a
//! single quad instead of 256. This is the "binary" variant: occupancy is a bit
//! test, and the three axes are processed independently with a 2-D mask.
//!
//! Scope, stated honestly: this accelerates meshing of voxel data handed to it.
//! It is not wired to Minecraft Bedrock's chunk renderer, which is not reachable
//! from a mod - `libminecraftpe.so` is stripped and the chunk mesh path exports
//! no symbol (see GlowberryClient `docs/agents/native-entity-camera-feed-*`).
//! The intended consumers are the client's own voxel work (preview and world
//! reconstruction).
//!
//! Ambient occlusion is baked per vertex (three neighbour samples, the classic
//! `0..3` occlusion value). Quads merge only when their AO signature matches, so
//! a merged quad never flattens an AO gradient.

use crate::math::Vec3;

/// A block id that is never solid.
pub const AIR: u8 = 0;

/// Integer dimensions of a chunk, in blocks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChunkDims {
    pub x: usize,
    pub y: usize,
    pub z: usize,
}

impl ChunkDims {
    #[inline]
    pub const fn new(x: usize, y: usize, z: usize) -> Self {
        Self { x, y, z }
    }

    #[inline]
    pub const fn volume(&self) -> usize {
        self.x * self.y * self.z
    }

    /// The 16x16x16 chunk the client's preview uses.
    #[inline]
    pub const fn cube_16() -> Self {
        Self::new(16, 16, 16)
    }
}

/// A single output vertex. `pos`/`normal` are fed to `glVertexAttribPointer`;
/// `ao` is `0..3` (0 fully occluded), `tint` is a per-face colour index.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vertex {
    pub pos: [f32; 3],
    pub normal: [i8; 3],
    pub uv: [f32; 2],
    pub ao: u8,
    pub tint: u8,
}

/// A meshed chunk: an indexed triangle list.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Mesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
}

impl Mesh {
    /// Number of emitted quads (four vertices each).
    #[inline]
    pub fn quad_count(&self) -> usize {
        self.vertices.len() / 4
    }
}

/// A read-only view over a chunk's block array, in `x`-fastest order.
pub struct ChunkView<'a> {
    blocks: &'a [u8],
    dims: ChunkDims,
}

impl<'a> ChunkView<'a> {
    /// Panics if `blocks.len()` does not match `dims.volume()`.
    pub fn new(blocks: &'a [u8], dims: ChunkDims) -> Self {
        assert_eq!(
            blocks.len(),
            dims.volume(),
            "block array length must equal dims.volume()"
        );
        Self { blocks, dims }
    }

    #[inline]
    fn at_signed(&self, c: [i32; 3]) -> u8 {
        if c[0] < 0
            || c[1] < 0
            || c[2] < 0
            || c[0] as usize >= self.dims.x
            || c[1] as usize >= self.dims.y
            || c[2] as usize >= self.dims.z
        {
            return AIR;
        }
        let (x, y, z) = (c[0] as usize, c[1] as usize, c[2] as usize);
        self.blocks[x + self.dims.x * (y + self.dims.y * z)]
    }
}

/// A mask cell: the block, facing and AO of one candidate quad.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FaceKey {
    block: u8,
    dir: i8,
    ao: [u8; 4],
}

/// Unit vector for an axis index.
#[inline]
fn unit(axis: usize) -> [i32; 3] {
    match axis {
        0 => [1, 0, 0],
        1 => [0, 1, 0],
        _ => [0, 0, 1],
    }
}

/// Bakes the four corner AO values of one face.
///
/// `vc` is the solid voxel, `dir` the facing sign, `(a, u, v)` the axis triple.
/// A corner is dark when both edge neighbours are solid, otherwise it loses one
/// step per solid sample.
fn face_ao(
    view: &ChunkView,
    vc: [i32; 3],
    a: usize,
    u: usize,
    v: usize,
    dir: i32,
    is_opaque: &impl Fn(u8) -> bool,
) -> [u8; 4] {
    let ea = unit(a);
    let eu = unit(u);
    let ev = unit(v);
    let base = [
        vc[0] + dir * ea[0],
        vc[1] + dir * ea[1],
        vc[2] + dir * ea[2],
    ];

    let mut out = [3u8; 4];
    // Corner order matches the emitted quad: (0,0), (1,0), (1,1), (0,1) in (u,v).
    let corners = [(-1i32, -1i32), (1, -1), (1, 1), (-1, 1)];
    for (i, (du, dv)) in corners.iter().enumerate() {
        let s1 = [
            base[0] + du * eu[0],
            base[1] + du * eu[1],
            base[2] + du * eu[2],
        ];
        let s2 = [
            base[0] + dv * ev[0],
            base[1] + dv * ev[1],
            base[2] + dv * ev[2],
        ];
        let cn = [s1[0] + dv * ev[0], s1[1] + dv * ev[1], s1[2] + dv * ev[2]];
        let o1 = is_opaque(view.at_signed(s1)) as u8;
        let o2 = is_opaque(view.at_signed(s2)) as u8;
        let oc = is_opaque(view.at_signed(cn)) as u8;
        out[i] = if o1 == 1 && o2 == 1 {
            0
        } else {
            3 - (o1 + o2 + oc)
        };
    }
    out
}

/// Meshes a chunk. `is_opaque` decides which blocks hide the face behind them.
pub fn mesh_chunk(view: &ChunkView, is_opaque: &impl Fn(u8) -> bool) -> Mesh {
    let dims = view.dims;
    let mut mesh = Mesh::default();

    for axis in 0..3usize {
        let u = (axis + 1) % 3;
        let v = (axis + 2) % 3;
        let na = match axis {
            0 => dims.x,
            1 => dims.y,
            _ => dims.z,
        };
        let nu = match u {
            0 => dims.x,
            1 => dims.y,
            _ => dims.z,
        };
        let nv = match v {
            0 => dims.x,
            1 => dims.y,
            _ => dims.z,
        };
        if na == 0 || nu == 0 || nv == 0 {
            continue;
        }

        // Slice -1..na so the face before voxel 0 and the face after voxel
        // na-1 are both produced. Mikola's original loop starts at -1 for this
        // reason; starting at 0 drops the near boundary of every axis.
        for slice in -1..(na as i32) {
            let mut mask = vec![None::<FaceKey>; nu * nv];

            for i in 0..nu {
                for j in 0..nv {
                    let mut c1 = [0i32; 3];
                    c1[axis] = slice;
                    c1[u] = i as i32;
                    c1[v] = j as i32;
                    let mut c2 = c1;
                    c2[axis] = slice + 1;

                    let b1 = view.at_signed(c1);
                    let b2 = view.at_signed(c2);
                    let o1 = is_opaque(b1);
                    let o2 = is_opaque(b2);

                    mask[i * nv + j] = if o1 && !o2 {
                        Some(FaceKey {
                            block: b1,
                            dir: 1,
                            ao: face_ao(view, c1, axis, u, v, 1, is_opaque),
                        })
                    } else if !o1 && o2 {
                        Some(FaceKey {
                            block: b2,
                            dir: -1,
                            ao: face_ao(view, c2, axis, u, v, -1, is_opaque),
                        })
                    } else {
                        None
                    };
                }
            }

            greedy_emit(&mask, nu, nv, axis, u, v, slice, &mut mesh);
        }
    }

    mesh
}

/// Merges equal mask cells into maximal quads and appends them.
#[allow(clippy::too_many_arguments)]
fn greedy_emit(
    mask: &[Option<FaceKey>],
    nu: usize,
    nv: usize,
    axis: usize,
    u: usize,
    v: usize,
    slice: i32,
    mesh: &mut Mesh,
) {
    let mut used = vec![false; nu * nv];

    for j in 0..nv {
        for i in 0..nu {
            let idx = i * nv + j;
            if used[idx] {
                continue;
            }
            let Some(key) = mask[idx] else {
                used[idx] = true;
                continue;
            };

            // Grow width along u.
            let mut w = 1;
            while i + w < nu && !used[(i + w) * nv + j] && mask[(i + w) * nv + j] == Some(key) {
                w += 1;
            }

            // Grow height along v, requiring the whole row to match.
            let mut h = 1;
            'grow: while j + h < nv {
                for k in 0..w {
                    let id2 = (i + k) * nv + (j + h);
                    if used[id2] || mask[id2] != Some(key) {
                        break 'grow;
                    }
                }
                h += 1;
            }

            for k in 0..w {
                for l in 0..h {
                    used[(i + k) * nv + (j + l)] = true;
                }
            }

            emit_quad(mesh, axis, u, v, slice, i, j, w, h, key);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn emit_quad(
    mesh: &mut Mesh,
    axis: usize,
    u: usize,
    v: usize,
    slice: i32,
    i: usize,
    j: usize,
    w: usize,
    h: usize,
    key: FaceKey,
) {
    // The face between voxel `slice` and voxel `slice + 1` sits at coordinate
    // `slice + 1`, whichever side is the solid one.
    let plane = (slice + 1) as f32;

    let corner = |off_u: usize, off_v: usize| -> [f32; 3] {
        let mut p = [0f32; 3];
        p[axis] = plane;
        p[u] = off_u as f32;
        p[v] = off_v as f32;
        p
    };

    let positions = [
        corner(i, j),
        corner(i + w, j),
        corner(i + w, j + h),
        corner(i, j + h),
    ];
    let uvs = [
        [0.0, 0.0],
        [w as f32, 0.0],
        [w as f32, h as f32],
        [0.0, h as f32],
    ];

    let mut normal = [0i8; 3];
    normal[axis] = key.dir;

    // (a, u, v) is a cyclic permutation, so e_u x e_v = e_a. For a +axis normal
    // the (u,v) order is counter-clockwise; a -axis normal reverses it.
    let order: [usize; 4] = if key.dir > 0 {
        [0, 1, 2, 3]
    } else {
        [0, 3, 2, 1]
    };

    let base = mesh.vertices.len() as u32;
    for &src in order.iter() {
        mesh.vertices.push(Vertex {
            pos: positions[src],
            normal,
            uv: uvs[src],
            ao: key.ao[src],
            tint: key.block,
        });
    }
    mesh.indices
        .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
}

/// Meshes a chunk where every non-[`AIR`] block is opaque. The common case.
pub fn mesh_chunk_blocks(blocks: &[u8], dims: ChunkDims) -> Mesh {
    let view = ChunkView::new(blocks, dims);
    mesh_chunk(&view, &|b| b != AIR)
}

/// Returns the world-space centre of a merged quad's bounding box. Used by the
/// caller to sort or cull the emitted geometry.
pub fn quad_center(mesh: &Mesh, quad: usize) -> Option<Vec3> {
    let base = quad * 4;
    if base + 3 >= mesh.vertices.len() {
        return None;
    }
    let mut acc = Vec3::ZERO;
    for k in 0..4 {
        let p = mesh.vertices[base + k].pos;
        acc += Vec3::new(p[0], p[1], p[2]);
    }
    Some(acc * 0.25)
}

#[cfg(test)]
mod tests {
    // Index expressions like `x + X * (y + Y * z)` spell out the x-fastest block
    // layout on purpose; clippy reads the constant factors as no-ops.
    #![allow(clippy::identity_op, clippy::erasing_op)]

    use super::*;

    fn solid_cube(blocks: &mut [u8], dims: ChunkDims, x0: usize, y0: usize, z0: usize, n: usize) {
        for x in x0..x0 + n {
            for y in y0..y0 + n {
                for z in z0..z0 + n {
                    blocks[x + dims.x * (y + dims.y * z)] = 1;
                }
            }
        }
    }

    #[test]
    fn empty_chunk_is_empty() {
        let dims = ChunkDims::cube_16();
        let blocks = vec![AIR; dims.volume()];
        let mesh = mesh_chunk_blocks(&blocks, dims);
        assert!(mesh.vertices.is_empty());
        assert!(mesh.indices.is_empty());
    }

    #[test]
    fn lone_block_emits_six_quads() {
        let dims = ChunkDims::new(4, 4, 4);
        let mut blocks = vec![AIR; dims.volume()];
        blocks[1 + 4 * (1 + 4 * 1)] = 1;
        let mesh = mesh_chunk_blocks(&blocks, dims);
        assert_eq!(mesh.quad_count(), 6);
        assert_eq!(mesh.vertices.len(), 24);
        assert_eq!(mesh.indices.len(), 36);
    }

    #[test]
    fn two_by_two_by_two_cube_merges_to_six_quads() {
        let dims = ChunkDims::new(4, 4, 4);
        let mut blocks = vec![AIR; dims.volume()];
        solid_cube(&mut blocks, dims, 0, 0, 0, 2);
        let mesh = mesh_chunk_blocks(&blocks, dims);
        // Vanilla per-block meshing would emit 8 * 6 = 48 quads.
        assert_eq!(mesh.quad_count(), 6, "outer faces should merge");
    }

    #[test]
    fn interior_faces_are_culled() {
        let dims = ChunkDims::new(4, 4, 4);
        let mut blocks = vec![AIR; dims.volume()];
        // Fill the whole chunk: only the six boundary planes are visible.
        for b in blocks.iter_mut() {
            *b = 1;
        }
        let mesh = mesh_chunk_blocks(&blocks, dims);
        assert_eq!(mesh.quad_count(), 6);
    }

    #[test]
    fn greedy_merges_flat_floor_to_one_quad() {
        let dims = ChunkDims::new(16, 16, 16);
        let mut blocks = vec![AIR; dims.volume()];
        for x in 0..16 {
            for z in 0..16 {
                blocks[x + 16 * (0 + 16 * z)] = 1;
            }
        }
        let mesh = mesh_chunk_blocks(&blocks, dims);
        // A 16x16 slab: top and bottom merge to one quad each, plus the four
        // sides (each 16x1, also one quad each) -> 6 quads total.
        assert_eq!(mesh.quad_count(), 6);
    }

    #[test]
    fn normals_are_unit_and_axis_aligned() {
        let dims = ChunkDims::new(4, 4, 4);
        let mut blocks = vec![AIR; dims.volume()];
        blocks[1 + 4 * (1 + 4 * 1)] = 1;
        let mesh = mesh_chunk_blocks(&blocks, dims);
        for v in &mesh.vertices {
            let n = v.normal;
            let mag = (n[0] as i32).abs() + (n[1] as i32).abs() + (n[2] as i32).abs();
            assert_eq!(mag, 1, "normal {n:?} must be a single unit axis");
        }
    }

    #[test]
    fn each_face_normal_appears_once_for_a_lone_block() {
        let dims = ChunkDims::new(4, 4, 4);
        let mut blocks = vec![AIR; dims.volume()];
        blocks[1 + 4 * (1 + 4 * 1)] = 1;
        let mesh = mesh_chunk_blocks(&blocks, dims);
        let mut normals: Vec<[i8; 3]> = mesh.vertices.iter().map(|v| v.normal).collect();
        normals.sort();
        normals.dedup();
        assert_eq!(normals.len(), 6);
    }

    #[test]
    fn occlusion_darkens_corners() {
        let dims = ChunkDims::new(4, 4, 4);
        let mut blocks = vec![AIR; dims.volume()];
        // A floor voxel at (1,1,1) and a solid neighbour above-left of its top
        // face, so some top-face corners are partially occluded.
        blocks[1 + 4 * (1 + 4 * 1)] = 1;
        blocks[0 + 4 * (2 + 4 * 1)] = 1; // above and -x of the top face
        let mesh = mesh_chunk_blocks(&blocks, dims);
        let min_ao = mesh.vertices.iter().map(|v| v.ao).min().unwrap();
        assert!(min_ao < 3, "some corner should be occluded, got {min_ao}");
    }

    #[test]
    fn opaque_neighbour_culls_the_shared_face() {
        let dims = ChunkDims::new(4, 4, 4);
        let opaque_only = |b: u8| b == 1;

        // Two opaque blocks at x=1 and x=2. The shared face is the plane x=2.
        let mut blocks = vec![AIR; dims.volume()];
        blocks[1 + 4 * (1 + 4 * 1)] = 1;
        blocks[2 + 4 * (1 + 4 * 1)] = 1;
        let mesh = mesh_chunk(&ChunkView::new(&blocks, dims), &opaque_only);

        let plus_x_at = |x: f32| {
            mesh.vertices
                .iter()
                .filter(|v| v.normal == [1, 0, 0] && (v.pos[0] - x).abs() < 1e-5)
                .count()
        };
        // No face is drawn between two opaque blocks.
        assert_eq!(plus_x_at(2.0), 0, "shared face must be culled");
        // The far side of the second block is still drawn.
        assert_eq!(plus_x_at(3.0), 4, "outer face must survive");
    }

    #[test]
    fn non_opaque_neighbour_does_not_cull_the_face() {
        let dims = ChunkDims::new(4, 4, 4);
        let opaque_only = |b: u8| b == 1;

        // Same layout, but the second block (id 2) is not opaque.
        let mut blocks = vec![AIR; dims.volume()];
        blocks[1 + 4 * (1 + 4 * 1)] = 1;
        blocks[2 + 4 * (1 + 4 * 1)] = 2;
        let mesh = mesh_chunk(&ChunkView::new(&blocks, dims), &opaque_only);

        let plus_x_at = |x: f32| {
            mesh.vertices
                .iter()
                .filter(|v| v.normal == [1, 0, 0] && (v.pos[0] - x).abs() < 1e-5)
                .count()
        };
        assert_eq!(plus_x_at(2.0), 4, "a non-opaque neighbour must not cull");
    }

    #[test]
    fn non_cubic_dimensions_mesh() {
        let dims = ChunkDims::new(32, 16, 32);
        let mut blocks = vec![AIR; dims.volume()];
        for x in 0..32 {
            for z in 0..32 {
                blocks[x + 32 * (0 + 16 * z)] = 1;
            }
        }
        let mesh = mesh_chunk_blocks(&blocks, dims);
        assert_eq!(mesh.quad_count(), 6);
    }

    #[test]
    fn meshing_is_deterministic() {
        let dims = ChunkDims::cube_16();
        let mut blocks = vec![AIR; dims.volume()];
        solid_cube(&mut blocks, dims, 3, 3, 3, 5);
        let a = mesh_chunk_blocks(&blocks, dims);
        let b = mesh_chunk_blocks(&blocks, dims);
        assert_eq!(a, b);
    }

    #[test]
    fn quad_center_is_inside_chunk() {
        let dims = ChunkDims::new(4, 4, 4);
        let mut blocks = vec![AIR; dims.volume()];
        blocks[1 + 4 * (1 + 4 * 1)] = 1;
        let mesh = mesh_chunk_blocks(&blocks, dims);
        let c = quad_center(&mesh, 0).unwrap();
        assert!(c.x >= 0.0 && c.x <= 4.0);
        assert!(c.y >= 0.0 && c.y <= 4.0);
        assert!(c.z >= 0.0 && c.z <= 4.0);
    }

    #[test]
    fn solid_chunk_face_count_matches_surface_area() {
        // A full 16^3 chunk exposes 6 faces of 16x16, each a single merged quad.
        let dims = ChunkDims::cube_16();
        let blocks = vec![1u8; dims.volume()];
        let mesh = mesh_chunk_blocks(&blocks, dims);
        assert_eq!(mesh.quad_count(), 6);
    }

    #[test]
    fn checkerboard_does_not_merge() {
        let dims = ChunkDims::new(8, 1, 8);
        let mut blocks = vec![AIR; dims.volume()];
        for x in 0..8 {
            for z in 0..8 {
                if (x + z) % 2 == 0 {
                    blocks[x + 8 * (0 + 1 * z)] = 1;
                }
            }
        }
        let mesh = mesh_chunk_blocks(&blocks, dims);
        // 32 isolated blocks, each 6 faces (bottom faces are culled only when a
        // neighbour exists; here they are isolated so all 6 show).
        assert_eq!(mesh.quad_count(), 32 * 6);
    }
}
