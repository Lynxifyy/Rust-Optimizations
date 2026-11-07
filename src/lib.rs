//! GlowberryClient native acceleration core.
//!
//! Pure Rust, no Android or preloader dependencies, so every module is
//! host-testable and the whole crate compiles and runs its tests on a normal
//! x86_64 Linux CI runner. The preloader (C++) integration lives in `cpp/`,
//! which links this crate through the C ABI in [`ffi`].
//!
//! Modules:
//! - [`math`]   - SIMD 3-D math: AABB collision, ray casts, frustum culling.
//! - [`mesher`] - binary greedy voxel meshing with baked ambient occlusion.
//! - [`offsets`] - the per-build offset map loader (fail-closed).
//! - [`parallel`] - a bounded scoped worker pool for off-thread meshing.
//! - [`pipeline`] - the ring + worker work loop (submit -> mesh -> drain).
//! - [`world`]    - a layered world description turned into a meshed volume.
//! - [`ring`]   - a bounded lock-free multi-producer/multi-consumer ring buffer.
//! - [`ffi`]    - the C ABI the C++ shim links against.

pub mod ffi;
pub mod math;
pub mod mesher;
pub mod offsets;
pub mod parallel;
pub mod pipeline;
pub mod ring;
pub mod world;

/// Semantic version of the core, as `major.minor.patch` packed into a `u32`
/// (`0x00MMmmpp`). The C++ shim logs it so a log line proves which core a
/// session loaded.
pub const fn version() -> u32 {
    let (major, minor, patch) = (0u32, 2u32, 0u32);
    (major << 16) | (minor << 8) | patch
}

#[cfg(test)]
mod version_tests {
    #[test]
    fn version_is_packed() {
        assert_eq!(super::version(), 0x0000_0200);
    }
}
