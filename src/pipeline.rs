//! The meshing pipeline: a real work loop that runs off the render thread.
//!
//! This is where the ring buffer and the mesher meet. A producer submits a chunk
//! mesh job; a bounded worker thread meshes it and pushes the result onto a
//! lock-free ring; the consumer (the render thread) drains the ring once per
//! frame. That is the loop the whole core exists for, and it is what the mod
//! exercises in-session.
//!
//! Why a *bounded* pipeline with a drop policy rather than an unbounded queue:
//! the worker can outrun the consumer (the render thread has a frame budget; the
//! worker does not). An unbounded queue turns that into growing memory and, on
//! the low-end devices this targets, an out-of-memory kill. So the result ring is
//! bounded and a full ring **drops** the finished mesh, counts it, and moves on -
//! a dropped mesh is a chunk the next frame can re-request, not a crash.
//!
//! The pipeline owns its worker threads for as long as it lives, so the caller
//! creates it once and drops it at teardown. The worker count is explicit: this
//! runs inside the game process, where a stray thread competes with the game's
//! own render and audio threads.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};

use crate::mesher::{mesh_chunk_blocks, ChunkDims, Mesh};
use crate::ring::Ring;

/// One unit of work: mesh a chunk of raw blocks.
pub struct MeshJob {
    /// A caller-chosen id, echoed back with the result so the consumer knows
    /// which chunk this mesh belongs to.
    pub chunk_id: u64,
    pub blocks: Vec<u8>,
    pub dims: ChunkDims,
}

/// A finished mesh, with the id of the job that produced it.
pub struct MeshResult {
    pub chunk_id: u64,
    pub mesh: Mesh,
}

/// Counters for diagnostics. All monotonic.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PipelineStats {
    /// Jobs accepted by [`MeshPipeline::submit`].
    pub submitted: u64,
    /// Jobs a worker actually meshed.
    pub meshed: u64,
    /// Finished meshes handed to the consumer.
    pub drained: u64,
    /// Finished meshes dropped because the result ring was full.
    pub dropped: u64,
    /// Jobs refused because the job queue was full (the consumer is too slow, or
    /// the workers are saturated).
    pub rejected: u64,
}

struct Shared {
    jobs: Ring<MeshJob>,
    results: Ring<MeshResult>,
    running: AtomicBool,
    submitted: AtomicU64,
    meshed: AtomicU64,
    drained: AtomicU64,
    dropped: AtomicU64,
    rejected: AtomicU64,
    // A job-available signal so a worker can sleep instead of spinning when the
    // queue is empty. This is a park/unpark-style wait, not a per-item lock.
    wake: (Mutex<bool>, Condvar),
}

impl Shared {
    fn new(job_capacity: usize, result_capacity: usize) -> Self {
        Self {
            jobs: Ring::with_capacity(job_capacity),
            results: Ring::with_capacity(result_capacity),
            running: AtomicBool::new(true),
            submitted: AtomicU64::new(0),
            meshed: AtomicU64::new(0),
            drained: AtomicU64::new(0),
            dropped: AtomicU64::new(0),
            rejected: AtomicU64::new(0),
            wake: (Mutex::new(false), Condvar::new()),
        }
    }

    /// Signals one waiting worker that a job is available.
    fn notify(&self) {
        let (lock, cvar) = &self.wake;
        if let Ok(mut ready) = lock.lock() {
            *ready = true;
            cvar.notify_one();
        }
    }

    /// Blocks until a job may be available or the pipeline is stopping.
    fn wait_for_job(&self) {
        let (lock, cvar) = &self.wake;
        if let Ok(mut ready) = lock.lock() {
            while !*ready && self.running.load(Ordering::Acquire) {
                let (guard, timeout) = cvar
                    .wait_timeout(ready, std::time::Duration::from_millis(20))
                    .expect("pipeline condvar poisoned");
                ready = guard;
                if timeout.timed_out() {
                    break;
                }
            }
            *ready = false;
        }
    }
}

/// A bounded, multi-worker chunk meshing pipeline.
pub struct MeshPipeline {
    shared: Arc<Shared>,
    workers: Vec<JoinHandle<()>>,
}

impl MeshPipeline {
    /// Starts `workers` worker threads (clamped to at least one).
    ///
    /// `job_capacity` bounds the not-yet-started jobs; `result_capacity` bounds
    /// the finished meshes waiting to be drained. Both round up to a power of two.
    pub fn new(workers: usize, job_capacity: usize, result_capacity: usize) -> Self {
        let workers = workers.max(1);
        let shared = Arc::new(Shared::new(job_capacity, result_capacity));
        let mut handles = Vec::with_capacity(workers);
        for _ in 0..workers {
            let shared = Arc::clone(&shared);
            handles.push(thread::spawn(move || worker_loop(shared)));
        }
        Self {
            shared,
            workers: handles,
        }
    }

    /// Submits a chunk to be meshed. Returns `false` when the job queue is full
    /// or the pipeline has been shut down.
    ///
    /// A `false` is a back-pressure signal, not an error: the caller should retry
    /// the chunk on a later frame rather than block the render thread.
    pub fn submit(&self, job: MeshJob) -> bool {
        if !self.shared.running.load(Ordering::Acquire) {
            self.shared.rejected.fetch_add(1, Ordering::Relaxed);
            return false;
        }
        match self.shared.jobs.push(job) {
            Ok(()) => {
                self.shared.submitted.fetch_add(1, Ordering::Relaxed);
                self.shared.notify();
                true
            }
            Err(_) => {
                self.shared.rejected.fetch_add(1, Ordering::Relaxed);
                false
            }
        }
    }

    /// Drains up to `max` finished meshes. Called once per frame by the consumer.
    pub fn drain(&self, max: usize) -> Vec<MeshResult> {
        let mut out = Vec::new();
        for _ in 0..max {
            match self.shared.results.pop() {
                Some(result) => {
                    self.shared.drained.fetch_add(1, Ordering::Relaxed);
                    out.push(result);
                }
                None => break,
            }
        }
        out
    }

    /// The best-effort number of finished meshes waiting to be drained.
    pub fn pending(&self) -> usize {
        self.shared.results.len()
    }

    /// The best-effort number of jobs not yet meshed.
    pub fn queued(&self) -> usize {
        self.shared.jobs.len()
    }

    /// A snapshot of the counters.
    pub fn stats(&self) -> PipelineStats {
        PipelineStats {
            submitted: self.shared.submitted.load(Ordering::Relaxed),
            meshed: self.shared.meshed.load(Ordering::Relaxed),
            drained: self.shared.drained.load(Ordering::Relaxed),
            dropped: self.shared.dropped.load(Ordering::Relaxed),
            rejected: self.shared.rejected.load(Ordering::Relaxed),
        }
    }

    /// Signals the workers to stop and joins them. Idempotent.
    pub fn shutdown(&mut self) {
        if !self.shared.running.swap(false, Ordering::AcqRel) {
            return;
        }
        // Wake every worker so none stays parked.
        for _ in 0..self.workers.len() {
            self.shared.notify();
        }
        for handle in self.workers.drain(..) {
            let _ = handle.join();
        }
    }
}

impl Drop for MeshPipeline {
    fn drop(&mut self) {
        self.shutdown();
    }
}

fn worker_loop(shared: Arc<Shared>) {
    while shared.running.load(Ordering::Acquire) {
        match shared.jobs.pop() {
            Some(job) => {
                let mesh = mesh_chunk_blocks(&job.blocks, job.dims);
                shared.meshed.fetch_add(1, Ordering::Relaxed);
                let result = MeshResult {
                    chunk_id: job.chunk_id,
                    mesh,
                };
                if shared.results.push(result).is_err() {
                    // The consumer is behind. Drop the finished mesh and count it
                    // rather than block or grow without bound; the chunk can be
                    // re-submitted next frame.
                    shared.dropped.fetch_add(1, Ordering::Relaxed);
                }
            }
            None => shared.wait_for_job(),
        }
    }
    // Drain whatever is left so a shutdown does not strand meshed chunks.
    while shared.jobs.pop().is_some() {}
}

#[cfg(test)]
mod tests {
    use super::*;

    fn job(id: u64, dims: ChunkDims, fill: u8) -> MeshJob {
        MeshJob {
            chunk_id: id,
            blocks: vec![fill; dims.volume()],
            dims,
        }
    }

    /// Drains until `want` results arrive or the deadline passes.
    fn drain_until(pipeline: &MeshPipeline, want: usize, deadline_ms: u64) -> Vec<MeshResult> {
        let start = std::time::Instant::now();
        let mut got = Vec::new();
        while got.len() < want && start.elapsed() < std::time::Duration::from_millis(deadline_ms) {
            got.extend(pipeline.drain(64));
            if got.len() < want {
                thread::yield_now();
            }
        }
        got
    }

    #[test]
    fn one_job_round_trips_through_a_worker() {
        let pipeline = MeshPipeline::new(1, 8, 8);
        let dims = ChunkDims::new(4, 4, 4);
        assert!(pipeline.submit(job(7, dims, 1)));
        let got = drain_until(&pipeline, 1, 2000);
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].chunk_id, 7);
        assert_eq!(got[0].mesh.quad_count(), 6);
        let s = pipeline.stats();
        assert_eq!(s.submitted, 1);
        assert_eq!(s.meshed, 1);
        assert_eq!(s.drained, 1);
        assert_eq!(s.dropped, 0);
    }

    #[test]
    fn many_jobs_all_arrive_with_multiple_workers() {
        // A generous queue so 100 jobs fit; submit is retried anyway, because
        // back-pressure is a normal outcome, not a failure.
        let pipeline = MeshPipeline::new(4, 256, 256);
        let dims = ChunkDims::cube_16();
        for id in 0..100 {
            while !pipeline.submit(job(id, dims, 1)) {
                thread::yield_now();
            }
        }
        let got = drain_until(&pipeline, 100, 5000);
        assert_eq!(got.len(), 100, "every submitted job should come back");
        let mut ids: Vec<u64> = got.iter().map(|r| r.chunk_id).collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), 100, "no duplicate or lost chunk ids");
        assert_eq!(pipeline.stats().meshed, 100);
    }

    #[test]
    fn empty_queue_reports_nothing() {
        let pipeline = MeshPipeline::new(2, 8, 8);
        assert!(pipeline.drain(16).is_empty());
        assert_eq!(pipeline.pending(), 0);
        assert_eq!(pipeline.queued(), 0);
        assert_eq!(pipeline.stats(), PipelineStats::default());
    }

    #[test]
    fn job_queue_full_reports_backpressure() {
        // One worker, and it is not started draining, so the job ring fills.
        let pipeline = MeshPipeline::new(1, 2, 2);
        let dims = ChunkDims::new(2, 2, 2);
        // Submit far more than the queue holds; at least one must be rejected.
        let mut accepted = 0;
        for id in 0..1000 {
            if pipeline.submit(job(id, dims, 1)) {
                accepted += 1;
            }
        }
        assert!(
            accepted < 1000,
            "a bounded queue must reject under pressure"
        );
        assert!(pipeline.stats().rejected > 0);
    }

    #[test]
    fn result_ring_full_drops_and_counts() {
        // A tiny result ring and a consumer that never drains: the worker must
        // drop finished meshes and count them rather than block or grow.
        let pipeline = MeshPipeline::new(1, 1024, 2);
        let dims = ChunkDims::new(2, 2, 2);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(3);
        while pipeline.stats().dropped == 0 && std::time::Instant::now() < deadline {
            // Keep feeding; the queue is large so submits mostly succeed.
            let _ = pipeline.submit(job(1, dims, 1));
            thread::yield_now();
        }
        let s = pipeline.stats();
        assert!(s.dropped > 0, "a full result ring must drop and count");
        // Nothing was lost silently: meshed == drained + dropped (at the end).
        assert_eq!(s.meshed, s.drained + s.dropped + pipeline.pending() as u64);
    }

    #[test]
    fn shutdown_joins_and_is_idempotent() {
        let mut pipeline = MeshPipeline::new(3, 8, 8);
        pipeline.submit(job(1, ChunkDims::new(2, 2, 2), 1));
        pipeline.shutdown();
        pipeline.shutdown(); // must not panic or hang
                             // A submit after shutdown is a no-op that reports back-pressure.
        assert!(!pipeline.submit(job(2, ChunkDims::new(2, 2, 2), 1)));
    }

    #[test]
    fn drop_shuts_down_without_panicking() {
        let pipeline = MeshPipeline::new(2, 8, 8);
        pipeline.submit(job(1, ChunkDims::cube_16(), 1));
        drop(pipeline); // Drop must join the workers cleanly
    }

    #[test]
    fn zero_workers_is_clamped_to_one() {
        let pipeline = MeshPipeline::new(0, 8, 8);
        assert!(pipeline.submit(job(1, ChunkDims::new(2, 2, 2), 1)));
        let got = drain_until(&pipeline, 1, 2000);
        assert_eq!(
            got.len(),
            1,
            "a zero-worker pipeline still meshes inline-ish"
        );
    }
}
