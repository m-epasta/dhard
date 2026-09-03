//! # dhard
//!
//! Data structures and traits for sharding data in memory and persisting shards to disk.
//!
//! The crate ships two layers:
//!
//! - **Persistence**: expressed through the [`Writable`], [`Readable`] and `ShardFormat`
//!   traits: implement them for your types to serialize shards onto any
//!   [`std::io::Write`]/[`std::io::Read`] sink, with `ShardWriter` and `ShardReader` as
//!   the driving handles. Always available, even with `--no-default-features`.
//! - **Sharding primitives** (default `multithreaded` feature): `ShardCollection`
//!   distributes items round-robin across cache-line-padded shards, each guarding its
//!   own items behind a `parking_lot::RwLock`. Items are distributed round-robin at
//!   push time, so concurrent writers mostly contend on different locks; every index
//!   returned by a push stays valid for the whole lifetime of the collection.
//!
//! On top of the same sharding pattern, the feature also enables two concurrent
//! *slot* maps — each occupies a different point of the trade-off space, and both are
//! covered by the benchmark suite (`cargo bench --bench throughput`):
//!
//! - `RwShardedSlotMap` mints stable `SlotKey` handles at insert time
//!   (shard + generational slot): O(1) lookups and removals without any hashing, and
//!   the fastest *concurrent* structure in the crate overall — provided deletions can
//!   present the stored handle rather than an arbitrary key. Each shard is a
//!   battle-tested [`slotmap`](https://docs.rs/slotmap) arena.
//! - `ShardedSlotMap` is a single-threaded sharded slot map: same round-robin
//!   sharding and `SlotKey` handles as `RwShardedSlotMap`, but with **no** locks and
//!   **no** atomics. Used from a single thread it runs an order of magnitude faster
//!   than the sharded, locked variant, and — uniquely in this crate — returns borrowed
//!   `&V`/`&mut V` references instead of only clones.
//!
//! ## Choosing a structure
//!
//! | You need | Use |
//! |---|---|
//! | Stable handles, O(1) remove, zero hashing, concurrent | `RwShardedSlotMap` |
//! | Stable handles, O(1) remove, zero hashing, single-threaded | `ShardedSlotMap` |
//! | Maximum raw append throughput | `ShardCollection` |
//! | Single-threaded keyed storage (bare arena) | `slotmap::SlotMap` |
//!
//! The sharded structures pay for their concurrency with per-operation locking:
//! single-threaded code is better served by the crate's own [`ShardedSlotMap`], which
//! drops the locks entirely.

use std::{
    error::Error,
    io::{Read, Write},
    marker::PhantomData,
};

#[cfg(feature = "multithreaded")]
use std::{
    ops::{Deref, DerefMut},
    sync::atomic::{AtomicUsize, Ordering},
};

#[cfg(feature = "multithreaded")]
use crossbeam_utils::CachePadded;
#[cfg(feature = "multithreaded")]
use parking_lot::{RwLock, RwLockReadGuard, RwLockWriteGuard};

#[cfg(feature = "multithreaded")]
pub mod slot_map;

#[cfg(feature = "multithreaded")]
pub use slot_map::{RwShardedSlotMap, ShardedSlotMap, SlotKey};

/// A collection of [`Shard`]s that distributes items across multiple shards
///
/// Items are never removed from a shard, so any `(shard, item)` index pair returned by
/// [`ShardCollection::push`] stays valid for the whole lifetime of the collection.
///
/// Shards and the round-robin counter are cache-line padded ([`crossbeam_utils::CachePadded`])
/// so that concurrently accessed shards do not invalidate each other's cache lines.
///
/// # Quick start
///
/// ```
/// use dhard::ShardCollection;
///
/// let collection: ShardCollection<u32> = ShardCollection::new(4);
///
/// let (shard_idx, item_idx) = collection.push(42).expect("collection has shards");
/// let item = collection
///     .get_shard(shard_idx)
///     .and_then(|shard| shard.get_cloned(item_idx));
///
/// assert_eq!(item, Some(42));
/// ```
#[cfg(feature = "multithreaded")]
pub struct ShardCollection<V> {
    shards: Vec<CachePadded<Shard<V>>>,
    rr_counter: CachePadded<AtomicUsize>,
}

/// A single shard containing items `V` behind a [`parking_lot::RwLock`]
#[cfg(feature = "multithreaded")]
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
#[cfg(feature = "multithreaded")]
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
#[cfg(feature = "multithreaded")]
pub struct ShardMutRef<'a, V> {
    guard: Option<RwLockWriteGuard<'a, Vec<V>>>,
    idx: usize,
}

/// [`ShardWriter`] is a public interface for writing to disk your [`Shard`]s.
///
/// It is generic over:
/// - `W`: the [`std::io::Write`] sink (file, buffer, socket, ...),
/// - `D`: an arbitrary data/config payload that [`ShardFormat`] can inspect,
/// - `E`: the [`std::error::Error`] type produced by items [`Writable`] on their
///   way out.
///
/// The writer tracks the total number of bytes written so far in
/// [`ShardWriter::data_size`].
#[cfg(feature = "multithreaded")]
pub struct ShardWriter<W, D, E> {
    writer: W,
    data_size: u64,
    data: D,
    checksum: u64,
    _e: PhantomData<E>,
}

/// [`ShardReader`] is a public interface for reading
/// your [`Shard`]s that have been written with [`ShardWriter`] to disk.
///
/// It is generic over:
/// - `R`: the [`std::io::Read`] source,
/// - `D`: an arbitrary data/config payload that [`ShardFormat`] can inspect,
/// - `E`: the [`std::error::Error`] type produced by items [`Readable`] on their
///   way in.
///
/// The reader tracks the total number of bytes it has consumed in
/// [`ShardReader::data_size`].
#[cfg(feature = "multithreaded")]
pub struct ShardReader<R, D, E> {
    reader: R,
    data_size: u64,
    data: D,
    checksum: u64,
    _e: PhantomData<E>,
}

#[cfg(feature = "multithreaded")]
impl<W: Write, D, E> ShardWriter<W, D, E> {
    /// Create a new [`ShardWriter`] over `writer`, carrying a copy of `data`
    /// so [`ShardFormat`] implementations (and callers) can adjust behaviour
    /// per format.
    pub fn new(writer: W, data: D) -> Self {
        Self {
            writer,
            data_size: 0,
            data,
            checksum: 0,
            _e: PhantomData,
        }
    }

    /// Immutable access to the underlying `data` payload.
    pub fn data(&self) -> &D {
        &self.data
    }

    /// Mutable access to the underlying `data` payload.
    pub fn data_mut(&mut self) -> &mut D {
        &mut self.data
    }

    /// Total number of bytes written so far.
    pub fn data_size(&self) -> u64 {
        self.data_size
    }

    /// The running checksum (currently reserved for format-specific use).
    pub fn checksum(&self) -> u64 {
        self.checksum
    }

    /// Consume the writer, returning the wrapped [`std::io::Write`] sink.
    pub fn into_writer(self) -> W {
        self.writer
    }
}

#[cfg(feature = "multithreaded")]
impl<R: Read, D, E> ShardReader<R, D, E> {
    /// Create a new [`ShardReader`] over `reader`, carrying a copy of `data`
    /// so [`ShardFormat`] implementations can adjust behaviour per format.
    pub fn new(reader: R, data: D) -> Self {
        Self {
            reader,
            data_size: 0,
            data,
            checksum: 0,
            _e: PhantomData,
        }
    }

    /// Immutable access to the underlying `data` payload.
    pub fn data(&self) -> &D {
        &self.data
    }

    /// Mutable access to the underlying `data` payload.
    pub fn data_mut(&mut self) -> &mut D {
        &mut self.data
    }

    /// Total number of bytes read so far.
    pub fn data_size(&self) -> u64 {
        self.data_size
    }

    /// The running checksum (currently reserved for format-specific use).
    pub fn checksum(&self) -> u64 {
        self.checksum
    }

    /// Consume the reader, returning the wrapped [`std::io::Read`] source.
    pub fn into_reader(self) -> R {
        self.reader
    }
}

/// [`ShardExt`] is a public interface where you can define
/// the sharding logic of your `Shard<YourType>`. The only constraint
/// is that you have to collect the shards into a [`ShardCollection`]
///
/// The number of shards is derived from a threshold of max items per shard, which
/// defaults to [`ShardExt::THRESHOLD`] and can be overridden per call via
/// [`ShardExt::shard_with`].
///
/// # Custom sharding logic
///
/// ```
/// use std::collections::HashMap;
/// use std::hash::Hash;
///
/// use dhard::{ShardCollection, ShardExt};
///
/// struct Chunked<K, V>(HashMap<K, V>);
///
/// impl<K, V> ShardExt<HashMap<K, V>> for Chunked<K, V>
/// where
///     K: Clone + Eq + Hash,
///     V: Clone,
/// {
///     type Item = (K, V);
///
///     fn shard_with(data: &HashMap<K, V>, threshold: usize) -> ShardCollection<(K, V)> {
///         let num_shards = data.len().div_ceil(threshold.max(1)).max(1);
///         let shards = ShardCollection::new(num_shards);
///         for (k, v) in data {
///             shards.push((k.clone(), v.clone()));
///         }
///         shards
///     }
/// }
///
/// let map: HashMap<u32, u32> = (0..100).map(|i| (i, i)).collect();
/// let shards = Chunked::shard(&map);
///
/// assert_eq!(shards.len(), 100);
/// assert!(shards.num_shards() > 1);
/// ```
#[cfg(feature = "multithreaded")]
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
/// (files, sockets, buffers, ...).
///
/// `write_to` takes `&self` so that items living inside a
/// [`Shard`], which are read-guarded behind a `parking_lot` RwLock, can be
/// written out without needing exclusive mutable access.
pub trait Writable {
    type Error;

    fn write_to<W: Write>(&self, writer: &mut W) -> Result<u64, Self::Error>;
}

/// This trait permits you to implement a general read logic over any Sized type
pub trait Readable: Sized {
    type Error;

    fn read_from<R: Read>(reader: &mut R) -> Result<Self, Self::Error>;
}

/// This trait permits you to validate via [`Writable`] the possibility of writing
/// a whole [`Shard<V>`] to a [`std::io::Write`] sink. The default
/// implementation is provided for [`ShardWriter`]; implement it on your own
/// types to layer header/footer or per-format framing around the body.
#[cfg(feature = "multithreaded")]
pub trait ShardWriteFormat<W: Write, D, E: Error> {
    fn write_shard<V: Writable<Error = E>>(&mut self, shard: &Shard<V>) -> Result<(), E>;
}

/// This trait permits you to validate via [`Readable`] the possibility of reading
/// a whole [`Shard<V>`] from a [`std::io::Read`] source. The default
/// implementation is provided for [`ShardReader`]; implement it on your own
/// types to layer header/footer or per-format framing around the body.
#[cfg(feature = "multithreaded")]
pub trait ShardReadFormat<R: Read> {
    fn read_shard<V: Readable>(&mut self, count: usize) -> Result<Shard<V>, V::Error>;
}

/// The default [`ShardFormat`] implementation behind [`ShardWriter`]: writes
/// every item of a [`Shard<V>`] sequentially (in index order) via
/// [`Writable::write_to`], accumulating the total byte count into
/// [`ShardWriter::data_size`].
#[cfg(feature = "multithreaded")]
impl<W: Write, D, E: Error> ShardWriteFormat<W, D, E> for ShardWriter<W, D, E> {
    fn write_shard<V: Writable<Error = E>>(&mut self, shard: &Shard<V>) -> Result<(), E> {
        let items = shard.items();
        for item in items.iter() {
            let n = item.write_to(&mut self.writer)?;
            self.data_size = self.data_size.wrapping_add(n);
        }
        Ok(())
    }
}

/// The default [`ShardReadFormat`] implementation behind [`ShardReader`]: reads
/// exactly `count` items sequentially (in index order) via [`Readable::read_from`],
/// reconstructing a [`Shard<V>`].
#[cfg(feature = "multithreaded")]
impl<R: Read, D, E> ShardReadFormat<R> for ShardReader<R, D, E> {
    fn read_shard<V: Readable>(&mut self, count: usize) -> Result<Shard<V>, V::Error> {
        let shard = Shard::with_capacity(count);
        for _ in 0..count {
            let item = V::read_from(&mut self.reader)?;
            shard.push(item);
        }
        Ok(shard)
    }
}

#[cfg(feature = "multithreaded")]
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

#[cfg(feature = "multithreaded")]
impl<V> Default for ShardCollection<V> {
    fn default() -> Self {
        Self::new(1)
    }
}

#[cfg(feature = "multithreaded")]
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

#[cfg(feature = "multithreaded")]
impl<V> Default for Shard<V> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "multithreaded")]
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

#[cfg(feature = "multithreaded")]
impl<V> Deref for ShardRef<'_, V> {
    type Target = V;

    #[inline]
    fn deref(&self) -> &V {
        &self.guard[self.idx]
    }
}

#[cfg(feature = "multithreaded")]
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

#[cfg(feature = "multithreaded")]
impl<V> Deref for ShardMutRef<'_, V> {
    type Target = V;

    #[inline]
    fn deref(&self) -> &V {
        &self.guard.as_ref().expect("lock already released")[self.idx]
    }
}

#[cfg(feature = "multithreaded")]
impl<V> DerefMut for ShardMutRef<'_, V> {
    #[inline]
    fn deref_mut(&mut self) -> &mut V {
        &mut self.guard.as_mut().expect("lock already released")[self.idx]
    }
}

#[cfg(all(test, feature = "multithreaded"))]
mod tests;
