use std::{
    error::Error,
    io::{Read, Write},
    marker::PhantomData,
    sync::atomic::{AtomicUsize, Ordering},
};

use parking_lot::{RwLock, RwLockReadGuard, RwLockWriteGuard};

/// A collection of [`Shard`]s that distributes items across multiple shards
pub struct ShardCollection<T, V> {
    shards: Vec<Shard<T, V>>,
    counter: AtomicUsize,
    _marker: PhantomData<T>,
}

/// A single shard containing items `V` behind a [`parking_lot::RwLock`]
pub struct Shard<T, V> {
    items: RwLock<Vec<V>>,
    _marker: PhantomData<T>,
}

/// Container over a [`RwLockReadGuard`] Vec<V> (which is items in [`Shard`]),
/// It also keeps a index to retrieve a reference over `self.items[idx]`
pub struct ShardRef<'a, V> {
    guard: RwLockReadGuard<'a, Vec<V>>,
    idx: usize,
}

/// Container over a [`RwLockWriteGuard`] Vec<V> (which is items in [`Shard`]),
/// It also keeps a index to retrieve a mutable reference over `self.items[idx]`
/// NOTE: [`RwLockWriteGuard`] locks the whole `Vec` because we take a write lock
pub struct ShardMutRef<'a, V> {
    guard: Option<RwLockWriteGuard<'a, Vec<V>>>,
    idx: usize,
}

/// [`ShardWriter`] is a public interface for writing to disk your `Shard<T>`
#[allow(dead_code)]
pub struct ShardWriter<W: Write, T, D, E: Error> {
    writer: W,
    data_size: u64,
    data: D,
    checksum: u64,
    _marker: PhantomData<T>,
    _e: E,
}

/// [`ShardWriter`] is a public interface for reading
/// your `Shard<T>` that has been written witth [`ShardWriter`] to disk
#[allow(dead_code)]
pub struct ShardReader<R: Read, T, D, E: Error> {
    reader: R,
    data_size: u64,
    data: D,
    checksum: u64,
    _marker: PhantomData<T>,
    _e: E,
}

/// [`ShardExt`] is a public interface where you can define
/// the sharding logic of your Shard<YourType>. The only constraint
/// is that you have to collect the shards as `Vec<V>`
pub trait ShardExt<T, V> {
    fn shard(data: &T) -> ShardCollection<T, V>;
}

/// This trait permits you to implement a general write logic to any Write compatible type
/// (so a type that can convert into &[u8])
pub trait Writable {
    type Error;

    fn write_to<W: Write>(&mut self, writer: W) -> Result<u64, Self::Error>;
}

/// This trait permits you to implement a general read logic over any Sized type
pub trait Readable: Sized {
    type Error;

    fn read_from<R: Read>(reader: &mut R) -> Result<Self, Self::Error>;
}

/// This trait permits you to validate via [`Writable`] and [`Readable`] the possibility
/// of writing to file your shards. Do prefer this trait for writing and reading your shards
pub trait ShardFormat<W: Write, T, D, E: Error> {
    fn write_shard<V: Writable>(&mut self, shard: &Shard<T, V>) -> Result<(), E>;
    fn read_shard<V: Readable>(&mut self) -> Result<Shard<T, V>, E>;
}

impl<T, V> ShardCollection<T, V> {
    /// Creates a new [`ShardCollection`] with `num_shards` empty shards
    pub fn new(num_shards: usize) -> Self {
        Self {
            shards: (0..num_shards).map(|_| Shard::new()).collect(),
            counter: AtomicUsize::new(0),
            _marker: PhantomData,
        }
    }

    /// Push an `item` of type [`V`] into a bucket (round-robin distribution)
    pub fn push(&self, item: V) {
        if self.shards.is_empty() {
            return;
        }
        let idx = self.counter.fetch_add(1, Ordering::Relaxed) % self.shards.len();
        self.shards[idx].push(item);
    }

    /// Returns a reference to a [`Shard`] at `idx`
    pub fn get_bucket(&self, idx: usize) -> Option<&Shard<T, V>> {
        self.shards.get(idx)
    }

    /// Returns the number of shards
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

impl<T, V> Default for ShardCollection<T, V> {
    fn default() -> Self {
        Self::new(1)
    }
}

impl<T, V> Shard<T, V> {
    /// Creates a new [`Shard`]
    pub fn new() -> Self {
        Self {
            items: RwLock::new(vec![]),
            _marker: PhantomData,
        }
    }

    /// Push an `item` of type [`V`] into `self.items`
    pub fn push(&self, item: V) {
        self.items.write().push(item);
    }

    /// Returns a [`RwLockReadGuard`] over a Vec of items [`V`]
    /// To have a clean `&[V]` type you have to reference the result of this function such as:
    /// ```ignore
    /// let guard = my_shard.items();
    ///
    /// // There are 3 ways to get the guard value as a slice
    /// let items: &[V] = guard.as_slice(); // Do prefer this method because it is the most explicit.
    /// let items = &guard[..];
    /// let items: &[V] = &*guard;
    /// ```
    /// To learn more, go read [parking_lot documentation](https://docs.rs/parking_lot/0.12.5/parking_lot/type.RwLockReadGuard.html)
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
    /// ```ignore
    /// let shard_ref = shard.get_ref(0).unwrap();
    /// let first_value: &V = shard_ref.get_ref();
    /// ```
    pub fn get_ref(&self, idx: usize) -> Option<ShardRef<'_, V>> {
        ShardRef::new(self.items.read(), idx)
    }

    /// Returns a [`ShardMutRef`] object that you can use as
    /// ```ignore
    /// let mut shard_ref = shard.get_mut(0).unwrap();
    /// let value = shard_ref.get_mut_ref(); // &mut V
    /// ```
    pub fn get_mut(&self, idx: usize) -> Option<ShardMutRef<'_, V>> {
        ShardMutRef::new(self.items.write(), idx)
    }

    /// Return the number of `items` in self
    pub fn len(&self) -> usize {
        self.items.read().len()
    }

    /// Returns wheter or not `items` is empty
    pub fn is_empty(&self) -> bool {
        self.items.read().is_empty()
    }
}

impl<T, V> Default for Shard<T, V> {
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
    pub fn get_ref(&self) -> &V {
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
    pub fn get_mut_ref(&mut self) -> &mut V {
        &mut self.guard.as_mut().expect("lock already released")[self.idx]
    }

    /// Releases the write lock early, allowing other threads to read/write
    /// NOTE: After calling this, [`get_mut_ref`] will panic
    pub fn release_lock(&mut self) {
        self.guard.take();
    }
}

#[cfg(test)]
mod integration_tests;
#[cfg(test)]
mod unit_tests;
