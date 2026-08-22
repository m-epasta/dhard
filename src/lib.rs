//! # dhard
//!
//! Data structures and traits for sharding data in memory and persisting shards to disk.
//!
//! The core type is [`ShardCollection`]: a fixed set of [`Shard`]s, each guarding its own
//! items behind a [`parking_lot::RwLock`]. Items are distributed
//! round-robin at push time, so concurrent writers mostly contend on different locks.
//! Items are never removed, hence every index returned by a push stays valid for the
//! whole lifetime of the collection.
//!
//! ## Quick start
//!
//! ```
//! use dhard::ShardCollection;
//!
//! let collection: ShardCollection<u32> = ShardCollection::new(4);
//!
//! let (shard_idx, item_idx) = collection.push(42).expect("collection has shards");
//! let item = collection
//!     .get_shard(shard_idx)
//!     .and_then(|shard| shard.get_cloned(item_idx));
//!
//! assert_eq!(item, Some(42));
//! ```
//!
//! ## Custom sharding logic
//!
//! Implement [`ShardExt`] to define how a piece of data is divided into shards.
//! The number of shards is derived from a threshold of max items per shard, which
//! defaults to [`ShardExt::THRESHOLD`] and can be overridden per call via
//! [`ShardExt::shard_with`].
//!
//! ```
//! use std::collections::HashMap;
//! use std::hash::Hash;
//!
//! use dhard::{ShardCollection, ShardExt};
//!
//! struct Chunked<K, V>(HashMap<K, V>);
//!
//! impl<K, V> ShardExt<HashMap<K, V>> for Chunked<K, V>
//! where
//!     K: Clone + Eq + Hash,
//!     V: Clone,
//! {
//!     type Item = (K, V);
//!
//!     fn shard_with(data: &HashMap<K, V>, threshold: usize) -> ShardCollection<(K, V)> {
//!         let num_shards = data.len().div_ceil(threshold.max(1)).max(1);
//!         let shards = ShardCollection::new(num_shards);
//!         for (k, v) in data {
//!             shards.push((k.clone(), v.clone()));
//!         }
//!         shards
//!     }
//! }
//!
//! let map: HashMap<u32, u32> = (0..100).map(|i| (i, i)).collect();
//! let shards = Chunked::shard(&map);
//!
//! assert_eq!(shards.len(), 100);
//! assert!(shards.num_shards() > 1);
//! ```
//!
//! ## Persistence
//!
//! Disk persistence is expressed through the [`Writable`], [`Readable`] and
//! [`ShardFormat`] traits: implement them for your types to serialize shards onto
//! any [`std::io::Write`]/[`std::io::Read`] sink, with [`ShardWriter`] and
//! [`ShardReader`] as the driving handles.
//!
//! ## Data structures
//!
//! Beyond the primitives, dhard ships three concurrent structures built on the same
//! sharding pattern — each occupies a different point of the trade-off space, and all
//! are covered by the benchmark suite (`cargo bench --bench throughput`):
//!
//! - [`RwShardedDlhtMap`] is the read/delete specialist: fingerprints in cache-line-
//!   chained bins point into a side arena, hashing each key once and touching a single
//!   cache line on most lookups. Fastest keyed lookups and removals, at the cost of
//!   roughly half-speed inserts. Design inspired by DLHT (HPDC'24,
//!   <https://arxiv.org/abs/2406.09986>).
//! - [`RwShardedSlotMap`] mints stable [`SlotKey`] handles at insert time
//!   (shard + generational slot): O(1) lookups and removals without any hashing, and
//!   the fastest structure in the crate overall — provided deletions can present the
//!   stored handle rather than an arbitrary key. Each shard is a battle-tested
//!   [`slotmap`](https://docs.rs/slotmap) arena.
//!
//! ## Choosing a structure
//!
//! | You need | Use |
//! |---|---|
//! | Keyed access, not heavy workload | [`RwShardedDlhtMap`] |
//! | Stable handles, O(1) remove, zero hashing | [`RwShardedSlotMap`] |
//! | Maximum raw append throughput | [`ShardCollection`] |
//! | Single-threaded keyed storage | `slotmap::SlotMap` |
//!
//! The sharded structures pay for their concurrency with per-operation locking:
//! single-threaded code is better served by the standard library.

use std::{
    error::Error,
    io::{Read, Write},
    marker::PhantomData,
    ops::{Deref, DerefMut},
    sync::atomic::{AtomicUsize, Ordering},
};

use crossbeam_utils::CachePadded;
use parking_lot::{RwLock, RwLockReadGuard, RwLockWriteGuard};

pub mod dlht_map;
pub mod slot_map;

mod hashing;

pub use dlht_map::RwShardedDlhtMap;
pub use slot_map::{RwShardedSlotMap, SlotKey};

/// A collection of [`Shard`]s that distributes items across multiple shards
///
/// Items are never removed from a shard, so any `(shard, item)` index pair returned by
/// [`ShardCollection::push`] stays valid for the whole lifetime of the collection.
///
/// Shards and the round-robin counter are cache-line padded ([`crossbeam_utils::CachePadded`])
/// so that concurrently accessed shards do not invalidate each other's cache lines.
pub struct ShardCollection<V> {
    shards: Vec<CachePadded<Shard<V>>>,
    rr_counter: CachePadded<AtomicUsize>,
}

/// A single shard containing items `V` behind a [`parking_lot::RwLock`]
pub struct Shard<V> {
    items: RwLock<Vec<V>>,
    items_count: AtomicUsize,
}

/// Container over a [`RwLockReadGuard`] `Vec<V>` (which is items in [`Shard`]),
/// It also keeps an index to retrieve a reference over `self.items[idx]`
///
/// NOTE: Like its inner guard, this type is `!Send` by default (see parking_lot's
/// `send_guard` feature) but implements [`Sync`], so it can be shared across threads
/// by reference. It implements [`Deref`] for direct access to the item.
pub struct ShardRef<'a, V> {
    guard: RwLockReadGuard<'a, Vec<V>>,
    idx: usize,
}

/// Container over a [`RwLockWriteGuard`] `Vec<V>` (which is items in [`Shard`]),
/// It also keeps a index to retrieve a mutable reference over `self.items[idx]`
/// NOTE: [`RwLockWriteGuard`] locks the whole `Vec` because we take a write lock
///
/// NOTE: Like its inner guard, this type is `!Send` by default (see parking_lot's
/// `send_guard` feature) but implements [`Sync`], so it can be shared across threads
/// by reference. It implements [`Deref`] and [`DerefMut`] for direct access to the item.
pub struct ShardMutRef<'a, V> {
    guard: Option<RwLockWriteGuard<'a, Vec<V>>>,
    idx: usize,
}

/// [`ShardWriter`] is a public interface for writing to disk your [`Shard`]s
#[allow(dead_code)]
pub struct ShardWriter<W, D, E> {
    writer: W,
    data_size: u64,
    data: D,
    checksum: u64,
    _e: PhantomData<E>,
}

/// [`ShardReader`] is a public interface for reading
/// your [`Shard`]s that have been written with [`ShardWriter`] to disk
#[allow(dead_code)]
pub struct ShardReader<R, D, E> {
    reader: R,
    data_size: u64,
    data: D,
    checksum: u64,
    _e: PhantomData<E>,
}

/// [`ShardExt`] is a public interface where you can define
/// the sharding logic of your `Shard<YourType>`. The only constraint
/// is that you have to collect the shards into a [`ShardCollection`]
pub trait ShardExt<D: ?Sized> {
    /// The item type stored inside each shard.
    type Item;

    /// Default maximum items targeted per shard.
    /// Override this const in your impl to retune the default for your type.
    const THRESHOLD: usize = 32;

    /// Shards `data` using [`Self::THRESHOLD`] as max items per shard.
    fn shard(data: &D) -> ShardCollection<Self::Item> {
        Self::shard_with(data, Self::THRESHOLD)
    }

    /// Shards `data`, targeting at most `threshold` items per shard.
    fn shard_with(data: &D, threshold: usize) -> ShardCollection<Self::Item>;
}

/// This trait permits you to implement a general write logic into any [`std::io::Write`] sink
/// (files, sockets, buffers, ...)
pub trait Writable {
    type Error;

    fn write_to<W: Write>(&mut self, writer: &mut W) -> Result<u64, Self::Error>;
}

/// This trait permits you to implement a general read logic over any Sized type
pub trait Readable: Sized {
    type Error;

    fn read_from<R: Read>(reader: &mut R) -> Result<Self, Self::Error>;
}

/// This trait permits you to validate via [`Writable`] and [`Readable`] the possibility
/// of writing to file your shards. Do prefer this trait for writing and reading your shards
pub trait ShardFormat<W: Write, D, E: Error> {
    fn write_shard<V: Writable<Error = E>>(&mut self, shard: &Shard<V>) -> Result<(), E>;
    fn read_shard<V: Readable<Error = E>>(&mut self) -> Result<Shard<V>, E>;
}

impl<V> ShardCollection<V> {
    /// Creates a new [`ShardCollection`] with `num_shards` empty shards
    pub fn new(num_shards: usize) -> Self {
        Self::with_capacity(num_shards, 0)
    }

    /// Creates a new [`ShardCollection`] with `num_shards` shards, each with pre-allocated
    /// capacity for `per_shard_capacity` items
    pub fn with_capacity(num_shards: usize, per_shard_capacity: usize) -> Self {
        Self {
            shards: (0..num_shards)
                .map(|_| CachePadded::new(Shard::with_capacity(per_shard_capacity)))
                .collect(),
            rr_counter: CachePadded::new(AtomicUsize::new(0)),
        }
    }

    /// Push an `item` of type `V` into a shard (round-robin distribution) and returns the index
    /// of which shard was written and the index of the item in the inner vector of items
    ///
    /// NOTE: Do not call this method while holding a [`ShardMutRef`] obtained from one of this
    /// collection's shards: if the round-robin targets that same shard, the write lock would be
    /// acquired twice, deadlocking (parking_lot locks are not reentrant).
    #[inline]
    pub fn push(&self, item: V) -> Option<(usize, usize)> {
        let n = self.shards.len();
        if n == 0 {
            return None;
        }
        let ticket = self.rr_counter.fetch_add(1, Ordering::Relaxed);
        let shard_idx = if n.is_power_of_two() {
            ticket & (n - 1)
        } else {
            ticket % n
        };
        let item_idx = self.shards[shard_idx].push(item);
        Some((shard_idx, item_idx))
    }

    /// Pushes all `items` into the collection following the same round-robin distribution
    /// of [`ShardCollection::push`]
    ///
    /// When the iterator reports enough items, they are distributed with a single atomic
    /// ticket reservation and flushed with one lock acquisition per shard; small inputs
    /// fall back to repeated [`ShardCollection::push`] calls.
    ///
    /// NOTE: Do not call this method while holding a [`ShardMutRef`] obtained from one of
    /// this collection's shards, as the distribution may target that same shard and deadlock.
    pub fn extend<I: IntoIterator<Item = V>>(&self, items: I) {
        let n = self.shards.len();
        if n == 0 {
            return;
        }
        let mut iter = items.into_iter();
        let (lower, _) = iter.size_hint();
        if lower < n.saturating_mul(2) {
            for item in iter {
                self.push(item);
            }
            return;
        }
        let base = self.rr_counter.fetch_add(lower, Ordering::Relaxed);
        let mut buffers: Vec<Vec<V>> = (0..n).map(|_| Vec::new()).collect();
        for (handled, item) in iter.by_ref().enumerate() {
            if handled == lower {
                break;
            }
            let ticket = base.wrapping_add(handled);
            let shard_idx = if n.is_power_of_two() {
                ticket & (n - 1)
            } else {
                ticket % n
            };
            buffers[shard_idx].push(item);
        }
        for (shard_idx, buffer) in buffers.into_iter().enumerate() {
            if !buffer.is_empty() {
                self.shards[shard_idx].extend(buffer);
            }
        }
        for item in iter {
            self.push(item);
        }
    }

    /// Returns a reference to a [`Shard`] at `idx`
    #[inline]
    pub fn get_shard(&self, idx: usize) -> Option<&Shard<V>> {
        self.shards.get(idx).map(|shard| &**shard)
    }

    /// Returns the number of shards
    #[inline]
    pub fn num_shards(&self) -> usize {
        self.shards.len()
    }

    /// Return the total number of items across all shards
    pub fn len(&self) -> usize {
        self.shards.iter().map(|b| b.len()).sum()
    }

    /// Returns whether or not all shards are empty
    pub fn is_empty(&self) -> bool {
        self.shards.iter().all(|b| b.is_empty())
    }
}

impl<V> Default for ShardCollection<V> {
    fn default() -> Self {
        Self::new(1)
    }
}

impl<V> Shard<V> {
    /// Creates a new [`Shard`]
    pub const fn new() -> Self {
        Self {
            items: RwLock::new(Vec::new()),
            items_count: AtomicUsize::new(0),
        }
    }

    /// Creates a new [`Shard`] with pre-allocated capacity for `capacity` items
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            items: RwLock::new(Vec::with_capacity(capacity)),
            items_count: AtomicUsize::new(0),
        }
    }

    /// Push an `item` of type `V` into `self.items` and returns the length of `self.items`
    ///
    /// NOTE: Do not call this method while holding a [`ShardMutRef`] over the same shard,
    /// the write lock would be acquired twice, deadlocking (parking_lot locks are not reentrant).
    #[inline]
    pub fn push(&self, item: V) -> usize {
        let mut items = self.items.write();
        items.push(item);
        self.items_count.fetch_add(1, Ordering::Relaxed);
        items.len() - 1
    }

    /// Pushes all `items` into `self.items`, acquiring the write lock once and reserving
    /// capacity for the whole batch
    ///
    /// NOTE: Do not call this method while holding a [`ShardMutRef`] over the same shard,
    /// the write lock would be acquired twice, deadlocking (parking_lot locks are not reentrant).
    pub fn extend<I: IntoIterator<Item = V>>(&self, items: I) {
        let batch: Vec<V> = items.into_iter().collect();
        if batch.is_empty() {
            return;
        }
        let count = batch.len();
        let mut guard = self.items.write();
        guard.reserve(count);
        guard.extend(batch);
        self.items_count.fetch_add(count, Ordering::Relaxed);
    }

    /// Returns a [`RwLockReadGuard`] over a Vec of items `V`
    /// To have a clean `&[V]` type you have to reference the result of this function such as:
    /// ```
    /// use dhard::Shard;
    ///
    /// let my_shard: Shard<i32> = Shard::new();
    /// my_shard.push(1);
    /// my_shard.push(2);
    ///
    /// let guard = my_shard.items();
    ///
    /// // There are 3 ways to get the guard value as a slice
    /// let items: &[i32] = guard.as_slice(); // Do prefer this method because it is the most explicit.
    /// let items = &guard[..];
    /// let items: &[i32] = &*guard;
    ///
    /// assert_eq!(items, &[1, 2]);
    /// ```
    /// To learn more, go read [`parking_lot documentation`](https://docs.rs/parking_lot/0.12.5/parking_lot/type.RwLockReadGuard.html)
    pub fn items(&self) -> RwLockReadGuard<'_, Vec<V>> {
        self.items.read()
    }

    /// Returns a cloned value of a `V` item in `self.items`
    /// NOTE: Use [`ShardRef`] helper if your type is expensive and/or
    /// does not impement [`Clone`]. Or if you need a **mutable** reference use [`ShardMutRef`]
    pub fn get_cloned(&self, idx: usize) -> Option<V>
    where
        V: Clone,
    {
        let items = self.items.read();
        items.get(idx).cloned()
    }

    /// Returns a [`ShardRef`] object that you can use as
    /// ```
    /// use dhard::Shard;
    ///
    /// let shard: Shard<i32> = Shard::new();
    /// shard.push(10);
    ///
    /// let shard_ref = shard.get_ref(0).unwrap();
    /// let first_value: &i32 = shard_ref.get_ref();
    ///
    /// assert_eq!(*first_value, 10);
    /// ```
    pub fn get_ref(&self, idx: usize) -> Option<ShardRef<'_, V>> {
        ShardRef::new(self.items.read(), idx)
    }

    /// Returns a [`ShardMutRef`] object that you can use as
    /// ```
    /// use dhard::Shard;
    ///
    /// let shard: Shard<String> = Shard::new();
    /// shard.push("hello".to_string());
    ///
    /// let mut shard_ref = shard.get_mut(0).unwrap();
    /// let value = shard_ref.get_mut_ref(); // &mut String
    /// value.push_str(", world");
    /// drop(shard_ref);
    ///
    /// assert_eq!(shard.get_cloned(0).unwrap(), "hello, world");
    /// ```
    pub fn get_mut(&self, idx: usize) -> Option<ShardMutRef<'_, V>> {
        ShardMutRef::new(self.items.write(), idx)
    }

    /// Return the number of `items` in self
    #[inline]
    pub fn len(&self) -> usize {
        self.items_count.load(Ordering::Relaxed)
    }

    /// Returns wheter or not `items` is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.items_count.load(Ordering::Relaxed) == 0
    }
}

impl<V> Default for Shard<V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a, V> ShardRef<'a, V> {
    fn new(guard: RwLockReadGuard<'a, Vec<V>>, idx: usize) -> Option<Self> {
        if idx < guard.len() {
            Some(Self { guard, idx })
        } else {
            None
        }
    }

    /// Returns a reference over the item in `items[self.idx]`
    #[inline]
    pub fn get_ref(&self) -> &V {
        self
    }
}

impl<V> Deref for ShardRef<'_, V> {
    type Target = V;

    #[inline]
    fn deref(&self) -> &V {
        &self.guard[self.idx]
    }
}

impl<'a, V> ShardMutRef<'a, V> {
    fn new(guard: RwLockWriteGuard<'a, Vec<V>>, idx: usize) -> Option<Self> {
        if idx < guard.len() {
            Some(Self {
                guard: Some(guard),
                idx,
            })
        } else {
            None
        }
    }

    /// Returns a mutable reference over the item in `items[self.idx]`
    /// NOTE: This method uses a [`RwLockWriteGuard`] which blocks reads and write
    /// during the access of the mutable reference to the `V` item
    #[inline]
    pub fn get_mut_ref(&mut self) -> &mut V {
        self
    }

    /// Releases the write lock early, allowing other threads to read/write
    /// NOTE: After calling this, [`ShardMutRef::get_mut_ref`] will panic
    #[inline]
    pub fn release_lock(&mut self) {
        self.guard.take();
    }
}

impl<V> Deref for ShardMutRef<'_, V> {
    type Target = V;

    #[inline]
    fn deref(&self) -> &V {
        &self.guard.as_ref().expect("lock already released")[self.idx]
    }
}

impl<V> DerefMut for ShardMutRef<'_, V> {
    #[inline]
    fn deref_mut(&mut self) -> &mut V {
        &mut self.guard.as_mut().expect("lock already released")[self.idx]
    }
}

#[cfg(test)]
mod tests;
