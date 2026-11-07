//! A bounded, lock-free, multi-producer / multi-consumer ring buffer.
//!
//! This is the Vyukov bounded MPMC queue: a fixed array of slots, each carrying
//! a per-slot sequence number, plus a producer and a consumer cursor. Every
//! operation is a single compare-and-swap on the relevant cursor and an
//! acquire/release pair on the slot, so there is no mutex and no lock held across
//! the payload copy.
//!
//! Why it exists: background worker threads finish meshing chunks and hand the
//! vertex data to the render thread. The render thread drains the ring once per
//! frame and uploads. A bounded ring is deliberate - a worker that outruns the
//! renderer gets `Err(value)` back and can drop or defer, instead of growing an
//! unbounded queue that turns a frame hitch into an out-of-memory kill.
//!
//! Safety: `T: Send` is required to move values across threads. The buffer is
//! sized to a power of two so the wrap is a mask, and the per-slot sequence
//! number is what makes the wrap correct without a global lock.

use std::cell::UnsafeCell;
use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicUsize, Ordering};

struct Slot<T> {
    sequence: AtomicUsize,
    value: UnsafeCell<MaybeUninit<T>>,
}

/// A bounded lock-free ring buffer.
pub struct Ring<T> {
    buffer: Box<[Slot<T>]>,
    mask: usize,
    enqueue_pos: AtomicUsize,
    dequeue_pos: AtomicUsize,
}

// SAFETY: the sequence numbers guarantee a slot is only written by one producer
// and only read by one consumer at a time; the `Send` bound covers the payload.
unsafe impl<T: Send> Send for Ring<T> {}
unsafe impl<T: Send> Sync for Ring<T> {}

impl<T> Ring<T> {
    /// Creates a ring that can hold `capacity` items.
    ///
    /// The capacity is rounded up to a power of two, with a minimum of 2. A
    /// single-slot ring cannot work: the sequence number that marks a slot
    /// "filled" would be the same value that marks it "consumed", so full and
    /// empty are indistinguishable. Two slots is the smallest ring that holds
    /// anything reliably.
    pub fn with_capacity(capacity: usize) -> Self {
        let cap = capacity.max(2).next_power_of_two();
        let mut buffer = Vec::with_capacity(cap);
        for i in 0..cap {
            buffer.push(Slot {
                sequence: AtomicUsize::new(i),
                value: UnsafeCell::new(MaybeUninit::uninit()),
            });
        }
        Self {
            buffer: buffer.into_boxed_slice(),
            mask: cap - 1,
            enqueue_pos: AtomicUsize::new(0),
            dequeue_pos: AtomicUsize::new(0),
        }
    }

    /// The usable capacity (a power of two).
    #[inline]
    pub fn capacity(&self) -> usize {
        self.buffer.len()
    }

    /// A best-effort count of the items currently in the ring.
    ///
    /// In the presence of concurrent producers and consumers this is a snapshot,
    /// not a guarantee: it is for diagnostics and for a render thread deciding
    /// whether to drain at all.
    #[inline]
    pub fn len(&self) -> usize {
        let enq = self.enqueue_pos.load(Ordering::Relaxed);
        let deq = self.dequeue_pos.load(Ordering::Relaxed);
        enq.saturating_sub(deq).min(self.capacity())
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Pushes a value. Returns `Err(value)` when the ring is full, so the caller
    /// keeps ownership and decides what to do.
    #[inline]
    pub fn push(&self, value: T) -> Result<(), T> {
        let mut pos = self.enqueue_pos.load(Ordering::Relaxed);
        loop {
            let slot = &self.buffer[pos & self.mask];
            let seq = slot.sequence.load(Ordering::Acquire);
            let dif = seq as isize - pos as isize;

            if dif == 0 {
                match self.enqueue_pos.compare_exchange_weak(
                    pos,
                    pos.wrapping_add(1),
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => {
                        // SAFETY: the CAS reserved this slot for us; no other
                        // producer can write it until we bump the sequence.
                        unsafe {
                            (*slot.value.get()).write(value);
                        }
                        slot.sequence.store(pos.wrapping_add(1), Ordering::Release);
                        return Ok(());
                    }
                    Err(actual) => pos = actual,
                }
            } else if dif < 0 {
                return Err(value);
            } else {
                pos = self.enqueue_pos.load(Ordering::Relaxed);
            }
        }
    }

    /// Pops a value, or `None` when the ring is empty.
    #[inline]
    pub fn pop(&self) -> Option<T> {
        let mut pos = self.dequeue_pos.load(Ordering::Relaxed);
        loop {
            let slot = &self.buffer[pos & self.mask];
            let seq = slot.sequence.load(Ordering::Acquire);
            let dif = seq as isize - pos.wrapping_add(1) as isize;

            if dif == 0 {
                match self.dequeue_pos.compare_exchange_weak(
                    pos,
                    pos.wrapping_add(1),
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => {
                        // SAFETY: the CAS reserved this slot; the acquire load of
                        // the sequence synchronised with the producer's release
                        // store, so the value is fully written.
                        let value = unsafe { (*slot.value.get()).assume_init_read() };
                        slot.sequence.store(
                            pos.wrapping_add(self.mask).wrapping_add(1),
                            Ordering::Release,
                        );
                        return Some(value);
                    }
                    Err(actual) => pos = actual,
                }
            } else if dif < 0 {
                return None;
            } else {
                pos = self.dequeue_pos.load(Ordering::Relaxed);
            }
        }
    }
}

impl<T> Drop for Ring<T> {
    fn drop(&mut self) {
        // Drop any values still in the ring so a `T` with a destructor does not
        // leak when the ring is destroyed with items outstanding.
        while self.pop().is_some() {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn capacity_rounds_up_to_power_of_two() {
        assert_eq!(Ring::<u32>::with_capacity(0).capacity(), 2);
        assert_eq!(Ring::<u32>::with_capacity(1).capacity(), 2);
        assert_eq!(Ring::<u32>::with_capacity(2).capacity(), 2);
        assert_eq!(Ring::<u32>::with_capacity(5).capacity(), 8);
        assert_eq!(Ring::<u32>::with_capacity(1024).capacity(), 1024);
    }

    #[test]
    fn holds_exactly_capacity_items() {
        let ring = Ring::with_capacity(4);
        for i in 0..4 {
            assert_eq!(ring.push(i), Ok(()), "item {i} should fit");
        }
        assert_eq!(ring.push(4), Err(4), "the 5th item must be refused");
    }

    #[test]
    fn fifo_order() {
        let ring = Ring::with_capacity(8);
        for i in 0..5 {
            ring.push(i).unwrap();
        }
        for i in 0..5 {
            assert_eq!(ring.pop(), Some(i));
        }
        assert_eq!(ring.pop(), None);
    }

    #[test]
    fn full_returns_the_value_back() {
        let ring = Ring::with_capacity(4);
        for i in 0..4 {
            ring.push(i).unwrap();
        }
        assert_eq!(ring.push(99), Err(99));
        assert_eq!(ring.pop(), Some(0));
        assert_eq!(ring.push(99), Ok(()));
    }

    #[test]
    fn empty_pop_is_none() {
        let ring = Ring::<u32>::with_capacity(4);
        assert_eq!(ring.pop(), None);
        assert!(ring.is_empty());
    }

    #[test]
    fn wraps_around_correctly() {
        let ring = Ring::with_capacity(4);
        for round in 0..10u32 {
            for i in 0..4 {
                ring.push(round * 4 + i).unwrap();
            }
            for i in 0..4 {
                assert_eq!(ring.pop(), Some(round * 4 + i));
            }
        }
        assert!(ring.is_empty());
    }

    #[test]
    fn len_tracks_occupancy() {
        let ring = Ring::with_capacity(8);
        assert_eq!(ring.len(), 0);
        ring.push(1).unwrap();
        ring.push(2).unwrap();
        assert_eq!(ring.len(), 2);
        ring.pop();
        assert_eq!(ring.len(), 1);
    }

    #[test]
    fn single_producer_single_consumer_across_threads() {
        let ring = Arc::new(Ring::with_capacity(64));
        let producer = {
            let ring = Arc::clone(&ring);
            thread::spawn(move || {
                let mut next = 0u64;
                while next < 10_000 {
                    if ring.push(next).is_ok() {
                        next += 1;
                    } else {
                        thread::yield_now();
                    }
                }
            })
        };
        let mut received = Vec::with_capacity(10_000);
        while received.len() < 10_000 {
            if let Some(v) = ring.pop() {
                received.push(v);
            } else {
                thread::yield_now();
            }
        }
        producer.join().unwrap();
        assert_eq!(received.len(), 10_000);
        for (i, v) in received.iter().enumerate() {
            assert_eq!(*v, i as u64, "FIFO order violated at {i}");
        }
    }

    #[test]
    fn multi_producer_multi_consumer_loses_nothing() {
        const PRODUCERS: u64 = 4;
        const PER_PRODUCER: u64 = 5_000;
        let total = PRODUCERS * PER_PRODUCER;

        let ring = Arc::new(Ring::with_capacity(128));
        let mut handles = Vec::new();

        for p in 0..PRODUCERS {
            let ring = Arc::clone(&ring);
            handles.push(thread::spawn(move || {
                for i in 0..PER_PRODUCER {
                    let value = p * PER_PRODUCER + i;
                    while ring.push(value).is_err() {
                        thread::yield_now();
                    }
                }
            }));
        }

        let received = Arc::new(std::sync::Mutex::new(Vec::with_capacity(total as usize)));
        let mut consumers = Vec::new();
        for _ in 0..4 {
            let ring = Arc::clone(&ring);
            let received = Arc::clone(&received);
            consumers.push(thread::spawn(move || loop {
                match ring.pop() {
                    Some(v) => {
                        let mut g = received.lock().unwrap();
                        if g.len() as u64 == total {
                            return;
                        }
                        g.push(v);
                    }
                    None => {
                        if received.lock().unwrap().len() as u64 == total {
                            return;
                        }
                        thread::yield_now();
                    }
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }
        // Give consumers a moment to drain the last items, then stop them.
        for c in consumers {
            let _ = c.join();
        }

        let got = received.lock().unwrap();
        assert_eq!(got.len() as u64, total, "some values were lost");
        // Every value must appear exactly once.
        let mut sorted = got.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len() as u64, total, "duplicate or missing values");
    }

    #[test]
    fn payload_with_destructor_is_dropped_on_drop() {
        use std::sync::atomic::{AtomicUsize, Ordering as O};
        static DROPS: AtomicUsize = AtomicUsize::new(0);
        #[derive(Debug)]
        struct Noisy;
        impl Drop for Noisy {
            fn drop(&mut self) {
                DROPS.fetch_add(1, O::SeqCst);
            }
        }
        {
            let ring = Ring::with_capacity(4);
            for _ in 0..3 {
                ring.push(Noisy).unwrap();
            }
            // Dropping the ring with three items outstanding must drop them.
        }
        assert_eq!(DROPS.load(O::SeqCst), 3);
    }

    #[test]
    fn strings_round_trip() {
        let ring = Ring::with_capacity(4);
        ring.push(String::from("chunk-0")).unwrap();
        ring.push(String::from("chunk-1")).unwrap();
        assert_eq!(ring.pop().unwrap(), "chunk-0");
        assert_eq!(ring.pop().unwrap(), "chunk-1");
    }
}
