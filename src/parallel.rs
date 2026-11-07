//! A bounded worker pool that meshes chunks off the render thread.
//!
//! Deliberately **not** rayon. Rayon's global pool spawns one worker per core,
//! has no affinity control, and lives for the whole process. This core is
//! injected into Minecraft Bedrock's process by the preloader, where a stray
//! unbounded pool competes with the game's own render and audio threads on a
//! low-end device - the exact device this work targets. Instead the caller
//! passes an explicit worker count (0 or 1 means "run inline"), the pool is
//! scoped to one batch, and no threads outlive the call.
//!
//! Chunk inputs are taken by shared reference, so meshing never needs a lock and
//! the caller keeps ownership - the `Arc<ChunkData>` pattern at the C++ boundary.

use std::thread;

use crate::mesher::{mesh_chunk_blocks, ChunkDims, Mesh};

/// One chunk to mesh: its dimensions and raw blocks.
pub struct ChunkJob<'a> {
    pub blocks: &'a [u8],
    pub dims: ChunkDims,
}

/// Meshes `jobs` across `workers` threads, preserving input order.
///
/// `workers == 0` or `workers == 1` runs inline with no thread spawn. Each worker
/// owns a contiguous range of jobs and a disjoint slice of the output, so there is
/// no shared mutable state and no atomic in the hot loop.
pub fn mesh_chunks_parallel(jobs: &[ChunkJob<'_>], workers: usize) -> Vec<Mesh> {
    if jobs.is_empty() {
        return Vec::new();
    }
    if workers <= 1 {
        return jobs
            .iter()
            .map(|j| mesh_chunk_blocks(j.blocks, j.dims))
            .collect();
    }

    let workers = workers.min(jobs.len());
    let chunk_size = jobs.len().div_ceil(workers);
    let mut out: Vec<Option<Mesh>> = (0..jobs.len()).map(|_| None).collect();

    thread::scope(|scope| {
        // Pair each job range with its matching, disjoint output slice.
        for (job_range, out_slice) in jobs.chunks(chunk_size).zip(out.chunks_mut(chunk_size)) {
            scope.spawn(move || {
                for (job, cell) in job_range.iter().zip(out_slice.iter_mut()) {
                    *cell = Some(mesh_chunk_blocks(job.blocks, job.dims));
                }
            });
        }
    });

    out.into_iter().map(|m| m.unwrap_or_default()).collect()
}

#[cfg(test)]
mod tests {
    // Index expressions like `x + X * (y + Y * z)` spell out the x-fastest block
    // layout on purpose; clippy reads the constant factors as no-ops.
    #![allow(clippy::identity_op, clippy::erasing_op)]

    use super::*;

    #[test]
    fn empty_jobs() {
        let out = mesh_chunks_parallel(&[], 4);
        assert!(out.is_empty());
    }

    #[test]
    fn inline_matches_sequential() {
        let dims = ChunkDims::cube_16();
        let a = vec![1u8; dims.volume()];
        let b = vec![0u8; dims.volume()];
        let jobs = vec![ChunkJob { blocks: &a, dims }, ChunkJob { blocks: &b, dims }];
        let out = mesh_chunks_parallel(&jobs, 0);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].quad_count(), 6);
        assert_eq!(out[1].quad_count(), 0);
    }

    #[test]
    fn parallel_result_matches_sequential() {
        let dims = ChunkDims::new(8, 8, 8);
        let buffers: Vec<Vec<u8>> = (0..8)
            .map(|i| {
                let mut b = vec![0u8; dims.volume()];
                b[i + 8 * (0 + 8 * 0)] = 1;
                b
            })
            .collect();
        let jobs: Vec<ChunkJob> = buffers
            .iter()
            .map(|b| ChunkJob { blocks: b, dims })
            .collect();

        let seq: Vec<Mesh> = jobs
            .iter()
            .map(|j| mesh_chunk_blocks(j.blocks, j.dims))
            .collect();
        let par = mesh_chunks_parallel(&jobs, 4);
        assert_eq!(seq, par, "parallel output must equal sequential output");
    }

    #[test]
    fn many_chunks_parallel_stable() {
        let dims = ChunkDims::cube_16();
        let buffers: Vec<Vec<u8>> = (0..32).map(|_| vec![1u8; dims.volume()]).collect();
        let jobs: Vec<ChunkJob> = buffers
            .iter()
            .map(|b| ChunkJob { blocks: b, dims })
            .collect();
        let par = mesh_chunks_parallel(&jobs, 8);
        assert_eq!(par.len(), 32);
        assert!(par.iter().all(|m| m.quad_count() == 6));
    }

    #[test]
    fn more_workers_than_jobs_is_fine() {
        let dims = ChunkDims::new(4, 4, 4);
        let b = vec![1u8; dims.volume()];
        let jobs = vec![ChunkJob { blocks: &b, dims }];
        let par = mesh_chunks_parallel(&jobs, 16);
        assert_eq!(par.len(), 1);
        assert_eq!(par[0].quad_count(), 6);
    }
}
