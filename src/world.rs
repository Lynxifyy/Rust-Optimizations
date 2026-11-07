//! Build a voxel volume from a layered world description and mesh it.
//!
//! GlowberryClient's `FlatWorldGenerator` builds worlds from a stack of block
//! layers - a list of `(block_name, count)` pairs, repeated across the whole
//! horizontal extent. That is the one place the client itself owns voxel-shaped
//! data, so it is the honest place for the mesher to do real work: the client
//! knows the layers, this core turns them into a greedy-meshed volume.
//!
//! Scope, stated honestly: this meshes a client-owned world description. It does
//! not read the running game's chunk data, which is not reachable from a mod
//! (`libminecraftpe.so` is stripped). See the repository README.

use crate::mesher::{mesh_chunk_blocks, ChunkDims, Mesh};

/// One horizontal layer of blocks, matching the client's `BlockLayer`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockLayer {
    /// A stable block id (the caller maps `minecraft:grass_block` etc. to this).
    pub block: u8,
    /// How many blocks tall this layer is.
    pub count: u32,
}

/// A layered world: a block stack repeated over an `width x depth` footprint.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LayeredWorld {
    pub width: u32,
    pub depth: u32,
    pub layers: Vec<BlockLayer>,
}

impl LayeredWorld {
    /// Total height: the sum of the layer counts.
    pub fn height(&self) -> u32 {
        self.layers.iter().map(|l| l.count).sum()
    }

    /// Packs the layers into an x-fastest block array of `width x height x depth`.
    ///
    /// The first layer in the list is the **bottom** (bedrock), matching how the
    /// client writes `block_layers` bottom-up. A layer with `count == 0` is
    /// skipped. A world with no layers (or a zero footprint) produces an empty
    /// array.
    pub fn to_blocks(&self) -> Vec<u8> {
        let h = self.height() as usize;
        let mut blocks = vec![0u8; self.width as usize * h * self.depth as usize];
        if blocks.is_empty() {
            return blocks;
        }
        let x = self.width as usize;
        let d = self.depth as usize;

        let mut y = 0usize;
        for layer in &self.layers {
            for _ in 0..layer.count {
                if y >= h {
                    break;
                }
                for xi in 0..x {
                    for zi in 0..d {
                        // x-fastest: index = x + width * (y + height * z)
                        blocks[xi + x * (y + h * zi)] = layer.block;
                    }
                }
                y += 1;
            }
        }
        blocks
    }

    /// Packs the layers and greedy-meshes the whole volume in one call.
    ///
    /// Convenience for a small footprint; a large world should go through the
    /// pipeline in chunk-sized pieces instead.
    pub fn to_mesh(&self) -> Mesh {
        let dims = ChunkDims::new(
            self.width as usize,
            self.height() as usize,
            self.depth as usize,
        );
        if dims.volume() == 0 {
            return Mesh::default();
        }
        mesh_chunk_blocks(&self.to_blocks(), dims)
    }
}

/// Splits a layered world into column chunks of `chunk_xz` by `chunk_y` blocks
/// and returns each chunk's `(origin_x, origin_y, origin_z, dims, blocks)`.
///
/// This is what feeds the pipeline: a caller walks the chunks and submits each
/// one, rather than meshing the whole world in a single call. A chunk is clamped
/// at the world edges, so a non-multiple footprint still meshes correctly.
pub fn split_layered_world(
    world: &LayeredWorld,
    chunk_xz: u32,
    chunk_y: u32,
) -> Vec<(u32, u32, u32, ChunkDims, Vec<u8>)> {
    let mut out = Vec::new();
    if chunk_xz == 0 || chunk_y == 0 || world.width == 0 || world.depth == 0 {
        return out;
    }
    let full = world.to_blocks();
    let h = world.height();
    if h == 0 {
        return out;
    }
    let wx = world.width;
    let wd = world.depth;

    let mut oy = 0u32;
    while oy < h {
        let cy = chunk_y.min(h - oy);
        let mut oz = 0u32;
        while oz < wd {
            let cz = chunk_xz.min(wd - oz);
            let mut ox = 0u32;
            while ox < wx {
                let cx = chunk_xz.min(wx - ox);

                let dims = ChunkDims::new(cx as usize, cy as usize, cz as usize);
                let mut blocks = vec![0u8; dims.volume()];
                for ly in 0..cy {
                    for lz in 0..cz {
                        for lx in 0..cx {
                            let src = (ox + lx) as usize
                                + wx as usize
                                    * ((oy + ly) as usize + h as usize * (oz + lz) as usize);
                            let dst = lx as usize
                                + cx as usize * (ly as usize + cy as usize * lz as usize);
                            blocks[dst] = full[src];
                        }
                    }
                }
                out.push((ox, oy, oz, dims, blocks));

                ox += cx;
            }
            oz += cz;
        }
        oy += cy;
    }
    out
}

#[cfg(test)]
mod tests {
    // Index expressions like `x + W * (y + H * z)` spell out the x-fastest block
    // layout on purpose; clippy reads the constant factors as no-ops.
    #![allow(clippy::identity_op, clippy::erasing_op)]

    use super::*;

    #[test]
    fn height_is_the_sum_of_layer_counts() {
        let world = LayeredWorld {
            width: 4,
            depth: 4,
            layers: vec![
                BlockLayer { block: 1, count: 1 }, // bedrock
                BlockLayer { block: 2, count: 2 }, // dirt
                BlockLayer { block: 3, count: 1 }, // grass
            ],
        };
        assert_eq!(world.height(), 4);
    }

    #[test]
    fn blocks_are_packed_bottom_up_x_fastest() {
        let world = LayeredWorld {
            width: 2,
            depth: 2,
            layers: vec![
                BlockLayer { block: 7, count: 1 }, // bottom
                BlockLayer { block: 9, count: 1 }, // top
            ],
        };
        let blocks = world.to_blocks();
        assert_eq!(blocks.len(), 2 * 2 * 2);
        // y=0 is layer 7, y=1 is layer 9.
        // index = x + 2 * (y + 2 * z)
        assert_eq!(blocks[0 + 2 * (0 + 2 * 0)], 7);
        assert_eq!(blocks[0 + 2 * (1 + 2 * 0)], 9);
    }

    #[test]
    fn zero_count_layer_is_skipped() {
        let world = LayeredWorld {
            width: 2,
            depth: 2,
            layers: vec![
                BlockLayer { block: 5, count: 0 },
                BlockLayer { block: 6, count: 1 },
            ],
        };
        assert_eq!(world.height(), 1);
        assert_eq!(world.to_blocks()[0], 6);
    }

    #[test]
    fn empty_world_meshes_to_nothing() {
        let world = LayeredWorld {
            width: 4,
            depth: 4,
            layers: vec![],
        };
        assert_eq!(world.to_mesh().quad_count(), 0);
        let world0 = LayeredWorld {
            width: 0,
            depth: 0,
            layers: vec![BlockLayer { block: 1, count: 1 }],
        };
        assert_eq!(world0.to_mesh().quad_count(), 0);
    }

    #[test]
    fn a_flat_slab_world_merges_to_six_quads() {
        // A 4x1x4 slab: top and bottom merge to one quad each, four sides each
        // one quad -> 6, regardless of the 16 blocks it contains.
        let world = LayeredWorld {
            width: 4,
            depth: 4,
            layers: vec![BlockLayer { block: 1, count: 1 }],
        };
        assert_eq!(world.to_mesh().quad_count(), 6);
    }

    #[test]
    fn a_two_layer_world_culls_the_internal_face() {
        // Two full layers of the *same* block: the whole 4x2x4 box is one
        // material, so every face merges and only six quads show.
        let world = LayeredWorld {
            width: 4,
            depth: 4,
            layers: vec![
                BlockLayer { block: 1, count: 1 },
                BlockLayer { block: 1, count: 1 },
            ],
        };
        assert_eq!(world.to_mesh().quad_count(), 6);
    }

    #[test]
    fn a_material_boundary_splits_the_side_faces() {
        // Two layers of *different* blocks. The internal seam is still culled
        // (both are opaque), but a side face cannot merge across the material
        // change, so each of the four sides becomes two quads: 4*2 + top + bottom.
        let world = LayeredWorld {
            width: 4,
            depth: 4,
            layers: vec![
                BlockLayer { block: 1, count: 1 },
                BlockLayer { block: 2, count: 1 },
            ],
        };
        assert_eq!(world.to_mesh().quad_count(), 10);
    }

    #[test]
    fn split_produces_tiling_chunks_that_cover_the_world() {
        let world = LayeredWorld {
            width: 4,
            depth: 4,
            layers: vec![BlockLayer { block: 1, count: 4 }],
        };
        let chunks = split_layered_world(&world, 2, 2);
        // 4x4x4 with 2x2x2 chunks -> 2 x 2 x 2 = 8 chunks.
        assert_eq!(chunks.len(), 8);
        let total: usize = chunks.iter().map(|c| c.4.len()).sum();
        assert_eq!(total, 4 * 4 * 4, "chunks must partition the world");
    }

    #[test]
    fn split_clamps_at_non_multiple_edges() {
        let world = LayeredWorld {
            width: 5,
            depth: 3,
            layers: vec![BlockLayer { block: 1, count: 5 }],
        };
        let chunks = split_layered_world(&world, 2, 2);
        // x: 5 -> 2,2,1 ; y: 5 -> 2,2,1 ; z: 3 -> 2,1  => 3 * 3 * 2 = 18
        assert_eq!(chunks.len(), 18);
        let total: usize = chunks.iter().map(|c| c.4.len()).sum();
        assert_eq!(total, 5 * 5 * 3);
        // Every chunk's dims must match its block count.
        for (_, _, _, dims, blocks) in &chunks {
            assert_eq!(blocks.len(), dims.volume());
        }
    }

    #[test]
    fn split_chunk_meshes_equal_a_whole_world_mesh_by_face_count() {
        // A world that fits one chunk must mesh identically either way.
        let world = LayeredWorld {
            width: 4,
            depth: 4,
            layers: vec![BlockLayer { block: 1, count: 4 }],
        };
        let whole = world.to_mesh().quad_count();
        let chunks = split_layered_world(&world, 8, 8);
        assert_eq!(chunks.len(), 1);
        let (_, _, _, dims, blocks) = &chunks[0];
        let meshed = mesh_chunk_blocks(blocks, *dims).quad_count();
        assert_eq!(meshed, whole);
    }

    #[test]
    fn split_of_empty_world_is_empty() {
        let world = LayeredWorld {
            width: 4,
            depth: 4,
            layers: vec![],
        };
        assert!(split_layered_world(&world, 2, 2).is_empty());
        assert!(split_layered_world(&world, 0, 2).is_empty());
    }
}
