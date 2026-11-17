//! Fault-safe memory probing, for deciding an offset against a live object.
//!
//! Static analysis could not settle `Actor -> Level`: two candidate offsets
//! (`0x20`, `0x1f0`) both looked plausible and conflicted. The only honest way to
//! decide is to read the live object and see which candidate behaves like a
//! `Level`. That means reading a pointer that might be garbage.
//!
//! **How the read is made safe.** Not with a `SIGSEGV` handler. In a live game
//! process a signal handler fires for *any* thread's fault, so a genuine crash on
//! the render thread would be swallowed and `longjmp`'d out of, corrupting game
//! state; `longjmp` from a signal handler is undefined behaviour across frames
//! that own resources; and it collides with ART/bionic's own handlers. Instead
//! every read goes through `process_vm_readv`, which the kernel answers with
//! `EFAULT` for an unmapped or unreadable address instead of delivering a signal.
//! That is the same primitive the preloader uses to scan regions that may unmap
//! mid-scan, so it is a technique this project already trusts.
//!
//! The **validation** is pure and host-tested: given the two words a candidate
//! produced, does the pair look like a `{begin, end}` vector? The reader is the
//! only platform-specific part.

use core::ffi::c_void;

/// A read that could not be completed (unmapped, unreadable, or short).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReadError {
    /// `process_vm_readv` reported the address is not readable (EFAULT/EPERM).
    Unreadable,
    /// The read succeeded but returned fewer bytes than asked for.
    Short,
}

/// Reads `size` bytes from `address` in this process, fault-safely.
///
/// Returns `Err(Unreadable)` for an address the kernel will not read, rather than
/// faulting. This is the one unsafe primitive the probe is built on.
pub fn read_bytes(address: usize, out: &mut [u8]) -> Result<(), ReadError> {
    if address == 0 {
        return Err(ReadError::Unreadable);
    }
    // `process_vm_readv(pid, local_iov, liovcnt, remote_iov, riovcnt, flags)`:
    // it copies *from* `remote_iov` *into* `local_iov`, so our buffer is local
    // and the target address is remote.
    let local = iovec {
        iov_base: out.as_mut_ptr() as *mut c_void,
        iov_len: out.len(),
    };
    let remote = iovec {
        iov_base: address as *mut c_void,
        iov_len: out.len(),
    };
    // SAFETY: `local` describes our live buffer and `remote` a single address in
    // this process; the syscall either copies or returns an error, and never
    // faults the caller.
    let read = unsafe { process_vm_readv(this_pid(), &local, 1, &remote, 1, 0) };
    if read < 0 {
        return Err(ReadError::Unreadable);
    }
    if (read as usize) < out.len() {
        return Err(ReadError::Short);
    }
    Ok(())
}

/// Reads a `usize` at `address`, or `None` when the read is refused or the value
/// is null.
pub fn read_pointer(address: usize) -> Option<usize> {
    let mut buf = [0u8; core::mem::size_of::<usize>()];
    read_bytes(address, &mut buf).ok()?;
    let value = usize::from_ne_bytes(buf);
    if value == 0 {
        None
    } else {
        Some(value)
    }
}

// --- The validation core (pure, host-tested) ---------------------------------

/// The largest plausible actor list, so a garbage pair is rejected.
pub const MAX_ENTITIES: usize = 65536;

/// Why a candidate was rejected. Reported, never swallowed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CandidateRejection {
    /// `[actor + offset]` was null or unreadable.
    NoLevel,
    /// `[level + level_manager_offset]` was null or unreadable.
    NoManager,
    /// `end < begin` - not a forward range.
    Inverted,
    /// `begin` or `end` is not 8-byte aligned.
    Misaligned,
    /// The span is not a whole number of 8-byte elements.
    Ragged,
    /// The span is larger than [`MAX_ENTITIES`] entries.
    TooLarge,
    /// `begin` is zero.
    NullBegin,
}

/// Validates a `{begin, end}` pair read from a candidate `Level`.
///
/// The rules are the same the preloader's `EntityListWalk` applies, so a value
/// this accepts is a value the walker would accept: both pointers non-null and
/// 8-aligned, `end >= begin`, and the span a whole number of pointers within the
/// cap. A pointer pair that fails any rule is not a vector.
pub fn validate_range(begin: usize, end: usize) -> Result<usize, CandidateRejection> {
    if begin == 0 {
        return Err(CandidateRejection::NullBegin);
    }
    if end < begin {
        return Err(CandidateRejection::Inverted);
    }
    if begin % 8 != 0 || end % 8 != 0 {
        return Err(CandidateRejection::Misaligned);
    }
    let bytes = end - begin;
    if bytes % 8 != 0 {
        return Err(CandidateRejection::Ragged);
    }
    let count = bytes / 8;
    if count > MAX_ENTITIES {
        return Err(CandidateRejection::TooLarge);
    }
    Ok(count)
}

/// The outcome of probing one candidate offset.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CandidateResult {
    pub offset: u64,
    /// The `Level*` read from `[actor + offset]`, or 0.
    pub level: usize,
    /// The `ActorManager*` read from `[level + level_manager_offset]`, or 0.
    pub manager: usize,
    pub begin: usize,
    pub end: usize,
    /// The entity count when the pair validated.
    pub entities: Option<usize>,
    pub rejection: Option<CandidateRejection>,
}

impl CandidateResult {
    pub fn is_valid(&self) -> bool {
        self.entities.is_some()
    }
}

/// Probes one candidate `actor + offset` -> `level + level_manager_offset` ->
/// `{begin, end}`, reading fault-safely at every step.
///
/// `actor` is the live `LocalPlayer*` the game returned (an `Actor*`). Nothing
/// here trusts a pointer: a null or unreadable read stops that candidate and is
/// reported, never dereferenced.
pub fn probe_candidate(actor: usize, offset: u64, level_manager_offset: u64) -> CandidateResult {
    let mut result = CandidateResult {
        offset,
        level: 0,
        manager: 0,
        begin: 0,
        end: 0,
        entities: None,
        rejection: Some(CandidateRejection::NoLevel),
    };

    // [actor + offset] -> Level*
    let Some(level) = read_pointer(actor.wrapping_add(offset as usize)) else {
        return result;
    };
    result.level = level;

    // [level + level_manager_offset] -> ActorManager*
    let Some(manager) = read_pointer(level.wrapping_add(level_manager_offset as usize)) else {
        result.rejection = Some(CandidateRejection::NoManager);
        return result;
    };
    result.manager = manager;

    // [manager + 0x8] = begin, [manager + 0x10] = end (libc++ vector).
    let Some(begin) = read_pointer(manager.wrapping_add(VECTOR_OFFSET)) else {
        result.rejection = Some(CandidateRejection::NullBegin);
        return result;
    };
    let mut end_bytes = [0u8; core::mem::size_of::<usize>()];
    if read_bytes(manager.wrapping_add(VECTOR_OFFSET + 8), &mut end_bytes).is_err() {
        result.rejection = Some(CandidateRejection::Inverted);
        return result;
    }
    let end = usize::from_ne_bytes(end_bytes);
    result.begin = begin;
    result.end = end;

    match validate_range(begin, end) {
        Ok(count) => {
            result.entities = Some(count);
            result.rejection = None;
        }
        Err(reason) => result.rejection = Some(reason),
    }
    result
}

/// The `std::vector` `{begin, end, cap}` offset inside the `ActorManager`.
const VECTOR_OFFSET: usize = 0x8;

/// Probes every candidate offset and returns the results.
///
/// The caller decides: exactly one valid candidate is the answer, and more than
/// one (or none) means the probe has not settled it and the offset stays
/// unrecovered.
pub fn probe_candidates(
    actor: usize,
    offsets: &[u64],
    level_manager_offset: u64,
) -> Vec<CandidateResult> {
    offsets
        .iter()
        .map(|&offset| probe_candidate(actor, offset, level_manager_offset))
        .collect()
}

/// The single offset that validated, or `None` when zero or more than one did.
pub fn decide_offset(results: &[CandidateResult]) -> Option<u64> {
    let mut valid = results.iter().filter(|r| r.is_valid());
    let first = valid.next()?;
    if valid.next().is_some() {
        // Ambiguous: more than one candidate looked like a Level. Do not guess.
        return None;
    }
    Some(first.offset)
}

// --- The syscall binding -----------------------------------------------------

#[repr(C)]
struct iovec {
    iov_base: *mut c_void,
    iov_len: usize,
}

extern "C" {
    fn process_vm_readv(
        pid: i32,
        local_iov: *const iovec,
        liovcnt: usize,
        remote_iov: *const iovec,
        riovcnt: usize,
        flags: usize,
    ) -> isize;
    fn getpid() -> i32;
}

fn this_pid() -> i32 {
    // SAFETY: getpid takes no arguments and cannot fail.
    unsafe { getpid() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_a_sane_range() {
        // begin = 0x1000, end = 0x1000 + 3*8 -> 3 entities.
        assert_eq!(validate_range(0x1000, 0x1018), Ok(3));
    }

    #[test]
    fn empty_range_is_valid_and_zero() {
        assert_eq!(validate_range(0x1000, 0x1000), Ok(0));
    }

    #[test]
    fn rejects_null_begin() {
        assert_eq!(
            validate_range(0, 0x1000),
            Err(CandidateRejection::NullBegin)
        );
    }

    #[test]
    fn rejects_inverted_range() {
        assert_eq!(
            validate_range(0x2000, 0x1000),
            Err(CandidateRejection::Inverted)
        );
    }

    #[test]
    fn rejects_misaligned_pointers() {
        assert_eq!(
            validate_range(0x1004, 0x1018),
            Err(CandidateRejection::Misaligned)
        );
        assert_eq!(
            validate_range(0x1000, 0x1014),
            Err(CandidateRejection::Misaligned)
        );
    }

    #[test]
    fn aligned_pair_is_always_a_whole_number_of_pointers() {
        // Both pointers being 8-aligned means the span is always a multiple of
        // 8, so the `Ragged` guard is defensive rather than reachable here. Pin
        // that: two aligned pointers a pointer apart is exactly one entity.
        assert_eq!(validate_range(0x1000, 0x1008), Ok(1));
        assert_eq!(validate_range(0x1000, 0x1010), Ok(2));
    }

    #[test]
    fn rejects_too_large() {
        let huge = (MAX_ENTITIES + 1) * 8;
        assert_eq!(
            validate_range(0x1000, 0x1000 + huge),
            Err(CandidateRejection::TooLarge)
        );
    }

    #[test]
    fn accepts_the_cap_exactly() {
        let cap = MAX_ENTITIES * 8;
        assert_eq!(validate_range(0x1000, 0x1000 + cap), Ok(MAX_ENTITIES));
    }

    #[test]
    fn decide_returns_the_single_valid_candidate() {
        let mk = |offset, entities: Option<usize>| CandidateResult {
            offset,
            level: 1,
            manager: 2,
            begin: 0x1000,
            end: 0x1010,
            entities,
            rejection: if entities.is_some() {
                None
            } else {
                Some(CandidateRejection::Inverted)
            },
        };
        assert_eq!(
            decide_offset(&[mk(0x20, None), mk(0x1f0, Some(2))]),
            Some(0x1f0)
        );
        assert_eq!(
            decide_offset(&[mk(0x20, Some(2)), mk(0x1f0, None)]),
            Some(0x20)
        );
    }

    #[test]
    fn decide_refuses_when_both_validate() {
        let mk = |offset| CandidateResult {
            offset,
            level: 1,
            manager: 2,
            begin: 0x1000,
            end: 0x1010,
            entities: Some(2),
            rejection: None,
        };
        assert_eq!(decide_offset(&[mk(0x20), mk(0x1f0)]), None);
    }

    #[test]
    fn decide_refuses_when_none_validate() {
        let mk = |offset| CandidateResult {
            offset,
            level: 0,
            manager: 0,
            begin: 0,
            end: 0,
            entities: None,
            rejection: Some(CandidateRejection::NoLevel),
        };
        assert_eq!(decide_offset(&[mk(0x20), mk(0x1f0)]), None);
    }

    #[test]
    fn reading_a_null_address_is_refused_not_faulted() {
        assert_eq!(read_pointer(0), None);
    }

    #[test]
    fn reading_an_unmapped_address_is_refused_not_faulted() {
        // A canonical-but-unmapped address: process_vm_readv returns EFAULT, so
        // this must be an Err, never a crash.
        let unmapped = 0x1000usize; // first page is never mapped
        let mut buf = [0u8; 8];
        assert!(read_bytes(unmapped, &mut buf).is_err());
    }

    #[test]
    fn reading_our_own_memory_works() {
        let value = 0xDEAD_BEEF_u64;
        let addr = &value as *const u64 as usize;
        let mut buf = [0u8; 8];
        assert_eq!(read_bytes(addr, &mut buf), Ok(()));
        assert_eq!(u64::from_ne_bytes(buf), 0xDEAD_BEEF);
    }

    #[test]
    fn read_pointer_reads_a_real_pointer() {
        let target = 0x1234usize;
        let addr = &target as *const usize as usize;
        assert_eq!(read_pointer(addr), Some(0x1234));
    }

    #[test]
    fn probe_of_a_null_actor_is_refused() {
        let r = probe_candidate(0, 0x20, 0x3e0);
        assert!(!r.is_valid());
        assert_eq!(r.rejection, Some(CandidateRejection::NoLevel));
    }
}
