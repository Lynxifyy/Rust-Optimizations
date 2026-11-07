//! Walk the live entity list from a `Level*`, with the recovered offsets.
//!
//! This is the entity-list walk the `EntitySeamOffsets` map feeds. The recovered
//! chain is two steps, which is why a single `ReadVectorRange(level, offset)` is
//! not enough:
//!
//! ```text
//! level + level_entity_manager_offset  ->  ActorManager*
//! ActorManager + vector_offset         ->  std::vector<Actor*> {begin, end, cap}
//! ```
//!
//! Recovered values (verified per build; see `offsets/offsets_<version>.json`):
//! `level_entity_manager_offset` is `0x3e0` on 26.60/26.62 and `0x3d8` on 26.52;
//! `vector_offset` is `0x8` on all three.
//!
//! **Fail-closed, like the preloader's `EntityListWalk`.** Every step checks the
//! pointer, the offset, the alignment and the span before touching memory, and a
//! refusal returns `None` rather than reading a wrong address. That matters
//! because these offsets come from a disassembly: a wrong one must be a no-op,
//! never a fault on the game thread.
//!
//! **The reads are volatile.** These addresses are the game's memory, written by
//! the game's threads; a volatile read is what stops the compiler from folding or
//! reordering them. The caller must still invoke this on the game thread, with a
//! `Level*` the game just handed back - the same contract the preloader's
//! local-player hook follows.

use crate::offsets::OffsetMap;

/// The libc++ `std::vector` offset inside the `ActorManager` (its first member).
pub const VECTOR_OFFSET_IN_MANAGER: u64 = 0x8;

/// The largest list this walker will accept, so a wrong offset cannot make it
/// run away. Bedrock's runtime actor list is at most a few thousand entries.
pub const MAX_ENTITIES: usize = 65536;

/// The offsets needed to walk the entity list, all recovered per build.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EntitySeam {
    /// `Level` -> `ActorManager*` member offset.
    pub level_entity_manager_offset: u64,
    /// `ActorManager` -> `std::vector<Actor*>` member offset.
    pub vector_offset_in_manager: u64,
    /// `Actor` -> `Vec3` position offset (for reading each entity's position).
    pub position_offset: u64,
}

impl EntitySeam {
    /// The seam from an offset map, or `None` when any piece is missing.
    ///
    /// `levelEntityListOffset` and `entityPositionOffset` come from the map;
    /// `vector_offset_in_manager` is the recovered constant.
    pub fn from_map(map: &OffsetMap) -> Option<Self> {
        let level = *map.entity_seam.get("levelEntityListOffset")?;
        let position = *map.entity_seam.get("entityPositionOffset")?;
        if level == 0 || position == 0 {
            return None;
        }
        Some(Self {
            level_entity_manager_offset: level,
            vector_offset_in_manager: VECTOR_OFFSET_IN_MANAGER,
            position_offset: position,
        })
    }
}

/// A borrowed `{begin, end}` range over the actor vector.
#[derive(Clone, Copy, Debug)]
pub struct EntityRange {
    pub begin: *const u8,
    pub end: *const u8,
}

impl EntityRange {
    /// The number of `Actor*` entries, or 0 when the span is unusable.
    pub fn len(&self) -> usize {
        if self.begin.is_null() || self.end.is_null() {
            return 0;
        }
        let start = self.begin as usize;
        let stop = self.end as usize;
        if stop < start {
            return 0;
        }
        let bytes = stop - start;
        if bytes % 8 != 0 {
            return 0;
        }
        let count = bytes / 8;
        if count > MAX_ENTITIES {
            return 0;
        }
        count
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The `Actor*` at `index`, or `None` when out of range or null.
    ///
    /// # Safety
    /// The range must point at live game memory (game thread).
    pub unsafe fn actor_at(&self, index: usize) -> Option<*const u8> {
        if index >= self.len() {
            return None;
        }
        let slot = self.begin.add(index * 8) as *const usize;
        let value = core::ptr::read_volatile(slot);
        if value == 0 {
            None
        } else {
            Some(value as *const u8)
        }
    }
}

/// Walks `level + level_entity_manager_offset -> ActorManager + vector_offset`
/// and returns the `{begin, end}` range, or `None` when any step is refused.
///
/// # Safety
/// `level` must be a live `Level*` the game returned (game thread). This reads
/// game memory; it is the caller's job to guarantee the pointer is current.
pub unsafe fn read_entity_range(level: *const u8, seam: &EntitySeam) -> Option<EntityRange> {
    if level.is_null() || seam.level_entity_manager_offset == 0 {
        return None;
    }
    // The member offset must be 8-byte aligned to hold a pointer.
    if seam.level_entity_manager_offset % 8 != 0 {
        return None;
    }

    let manager_slot = level.add(seam.level_entity_manager_offset as usize) as *const usize;
    let manager = core::ptr::read_volatile(manager_slot) as *const u8;
    if manager.is_null() {
        return None;
    }
    if seam.vector_offset_in_manager % 8 != 0 {
        return None;
    }

    let vector = manager.add(seam.vector_offset_in_manager as usize) as *const usize;
    let begin = core::ptr::read_volatile(vector) as *const u8;
    let end = core::ptr::read_volatile(vector.add(1)) as *const u8;
    let range = EntityRange { begin, end };
    if range.is_empty() {
        None
    } else {
        Some(range)
    }
}

/// Reads one entity's position (`Vec3`) at `position_offset`.
///
/// Returns `None` for a null actor, an unaligned offset, or a non-finite or
/// absurd value - a garbage read must read as "no data", never as a position.
///
/// # Safety
/// `actor` must be a live `Actor*` from a range this module produced.
pub unsafe fn read_entity_position(actor: *const u8, position_offset: u64) -> Option<[f32; 3]> {
    if actor.is_null() || position_offset % 4 != 0 {
        return None;
    }
    let base = actor.add(position_offset as usize) as *const f32;
    let x = core::ptr::read_volatile(base);
    let y = core::ptr::read_volatile(base.add(1));
    let z = core::ptr::read_volatile(base.add(2));
    if !x.is_finite() || !y.is_finite() || !z.is_finite() {
        return None;
    }
    // A real Bedrock position is well inside the world border.
    const LIMIT: f32 = 3.0e7;
    if x.abs() > LIMIT || y.abs() > LIMIT || z.abs() > LIMIT {
        return None;
    }
    Some([x, y, z])
}

#[cfg(test)]
mod tests {
    use super::*;

    const LEVEL_MANAGER_OFFSET: usize = 0x3e0;
    const MANAGER_VECTOR_OFFSET: usize = 0x8;
    const POSITION_OFFSET: usize = 0x230;
    const ACTOR_SIZE: usize = POSITION_OFFSET + 12;

    /// A synthetic Level -> ActorManager -> vector<Actor*> graph.
    ///
    /// The actors are separately allocated (as the engine allocates them), and
    /// the vector holds their addresses - 8 bytes per element. The level and
    /// manager live in one buffer whose internal pointers are computed from its
    /// own base, so the graph stays valid while the `Fixture` is alive.
    struct Fixture {
        /// Owns the actor allocations. Kept alive because the pointer array in
        /// `buf` holds addresses into these; never read directly.
        #[allow(dead_code)]
        actors: Vec<Vec<u8>>,
        buf: Vec<u8>,
        count: usize,
    }

    impl Fixture {
        fn new(positions: &[[f32; 3]]) -> Self {
            let count = positions.len();

            // Each actor is its own allocation, with the position field filled.
            let actors: Vec<Vec<u8>> = positions
                .iter()
                .map(|p| {
                    let mut a = vec![0u8; ACTOR_SIZE];
                    a[POSITION_OFFSET..POSITION_OFFSET + 4].copy_from_slice(&p[0].to_le_bytes());
                    a[POSITION_OFFSET + 4..POSITION_OFFSET + 8]
                        .copy_from_slice(&p[1].to_le_bytes());
                    a[POSITION_OFFSET + 8..POSITION_OFFSET + 12]
                        .copy_from_slice(&p[2].to_le_bytes());
                    a
                })
                .collect();

            let level_off = 0usize;
            let manager_off = LEVEL_MANAGER_OFFSET + 8;
            let manager_size = MANAGER_VECTOR_OFFSET + 24;
            let vector_off = manager_off + MANAGER_VECTOR_OFFSET;
            let pointers_off = manager_off + manager_size;
            let total = pointers_off + count * 8;
            let mut buf = vec![0u8; total];
            let base = buf.as_ptr() as usize;

            // The vector's {begin, end, cap}: a contiguous array of Actor*.
            let begin = base + pointers_off;
            let end = begin + count * 8;
            buf[vector_off..vector_off + 8].copy_from_slice(&begin.to_le_bytes());
            buf[vector_off + 8..vector_off + 16].copy_from_slice(&end.to_le_bytes());
            buf[vector_off + 16..vector_off + 24].copy_from_slice(&end.to_le_bytes());

            // The pointer array: one Actor* per element.
            for (i, actor) in actors.iter().enumerate() {
                let slot = pointers_off + i * 8;
                buf[slot..slot + 8].copy_from_slice(&(actor.as_ptr() as usize).to_le_bytes());
            }

            // The manager pointer, at level + LEVEL_MANAGER_OFFSET.
            let m = base + manager_off;
            buf[level_off + LEVEL_MANAGER_OFFSET..level_off + LEVEL_MANAGER_OFFSET + 8]
                .copy_from_slice(&m.to_le_bytes());

            Self { actors, buf, count }
        }

        fn level(&self) -> *const u8 {
            self.buf.as_ptr()
        }

        fn seam(&self) -> EntitySeam {
            EntitySeam {
                level_entity_manager_offset: LEVEL_MANAGER_OFFSET as u64,
                vector_offset_in_manager: MANAGER_VECTOR_OFFSET as u64,
                position_offset: POSITION_OFFSET as u64,
            }
        }
    }

    #[test]
    fn walks_a_synthetic_entity_list() {
        let fx = Fixture::new(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let seam = fx.seam();
        unsafe {
            let range = read_entity_range(fx.level(), &seam).expect("range");
            assert_eq!(range.len(), 2);
            let a0 = range.actor_at(0).expect("actor 0");
            assert_eq!(
                read_entity_position(a0, seam.position_offset),
                Some([1.0, 2.0, 3.0])
            );
            let a1 = range.actor_at(1).expect("actor 1");
            assert_eq!(
                read_entity_position(a1, seam.position_offset),
                Some([4.0, 5.0, 6.0])
            );
        }
    }

    #[test]
    fn fixture_keeps_its_count() {
        let fx = Fixture::new(&[[0.0, 0.0, 0.0]]);
        assert_eq!(fx.count, 1);
    }

    #[test]
    fn null_level_is_refused() {
        let seam = EntitySeam {
            level_entity_manager_offset: LEVEL_MANAGER_OFFSET as u64,
            vector_offset_in_manager: MANAGER_VECTOR_OFFSET as u64,
            position_offset: POSITION_OFFSET as u64,
        };
        unsafe {
            assert!(read_entity_range(std::ptr::null(), &seam).is_none());
        }
    }

    #[test]
    fn zero_offset_is_refused() {
        let fx = Fixture::new(&[[0.0, 0.0, 0.0]]);
        let mut seam = fx.seam();
        seam.level_entity_manager_offset = 0;
        unsafe {
            assert!(read_entity_range(fx.level(), &seam).is_none());
        }
    }

    #[test]
    fn misaligned_offset_is_refused() {
        let fx = Fixture::new(&[[0.0, 0.0, 0.0]]);
        let mut seam = fx.seam();
        seam.level_entity_manager_offset = 0x3e1;
        unsafe {
            assert!(read_entity_range(fx.level(), &seam).is_none());
        }
    }

    #[test]
    fn null_manager_is_refused() {
        let level = vec![0u8; LEVEL_MANAGER_OFFSET + 8];
        let seam = EntitySeam {
            level_entity_manager_offset: LEVEL_MANAGER_OFFSET as u64,
            vector_offset_in_manager: MANAGER_VECTOR_OFFSET as u64,
            position_offset: POSITION_OFFSET as u64,
        };
        unsafe {
            assert!(read_entity_range(level.as_ptr(), &seam).is_none());
        }
    }

    #[test]
    fn empty_range_reports_zero() {
        let a = [0u8; 8];
        let r = EntityRange {
            begin: a.as_ptr(),
            end: a.as_ptr(),
        };
        assert_eq!(r.len(), 0);
        assert!(r.is_empty());
    }

    #[test]
    fn end_before_begin_is_refused() {
        let a = [0u8; 16];
        let r = EntityRange {
            begin: unsafe { a.as_ptr().add(8) },
            end: a.as_ptr(),
        };
        assert_eq!(r.len(), 0);
    }

    #[test]
    fn non_finite_position_reads_as_no_data() {
        let mut actor = vec![0u8; POSITION_OFFSET + 12];
        actor[POSITION_OFFSET..POSITION_OFFSET + 4].copy_from_slice(&f32::NAN.to_le_bytes());
        unsafe {
            assert!(read_entity_position(actor.as_ptr(), POSITION_OFFSET as u64).is_none());
        }
    }

    #[test]
    fn absurd_position_reads_as_no_data() {
        let mut actor = vec![0u8; POSITION_OFFSET + 12];
        actor[POSITION_OFFSET..POSITION_OFFSET + 4].copy_from_slice(&1.0e30f32.to_le_bytes());
        unsafe {
            assert!(read_entity_position(actor.as_ptr(), POSITION_OFFSET as u64).is_none());
        }
    }

    #[test]
    fn actor_at_out_of_range_is_none() {
        let fx = Fixture::new(&[[0.0, 0.0, 0.0]]);
        let seam = fx.seam();
        unsafe {
            let range = read_entity_range(fx.level(), &seam).unwrap();
            assert!(range.actor_at(range.len()).is_none());
        }
    }

    #[test]
    fn seam_from_map_requires_both_offsets() {
        let json = r#"{"version":"x","so_sha256":"a","entity_seam_offsets":{"fields":[
            {"name":"levelEntityListOffset","offset":992},
            {"name":"entityPositionOffset","offset":560}
        ]}}"#;
        let map = OffsetMap::parse(json).unwrap();
        let seam = EntitySeam::from_map(&map).expect("seam");
        assert_eq!(seam.level_entity_manager_offset, 992);
        assert_eq!(seam.position_offset, 560);
        assert_eq!(seam.vector_offset_in_manager, 8);

        let missing = OffsetMap::parse(
            r#"{"version":"x","so_sha256":"a","entity_seam_offsets":{"fields":[
                {"name":"levelEntityListOffset","offset":992}
            ]}}"#,
        )
        .unwrap();
        assert!(EntitySeam::from_map(&missing).is_none());
    }
}
