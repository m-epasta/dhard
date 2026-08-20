use std::{
    error::Error,
    io::{Read, Write},
    marker::PhantomData,
};

use parking_lot::{RwLock, RwLockReadGuard, RwLockWriteGuard};

/// Data structure for sharding your `T` into
/// a collection of items `V`
/// NOTE: items are thread safe because of a [`parking_lot::RwLock`] wrapper
pub struct Shard<T, V> {
    pub items: RwLock<Vec<V>>,
    _marker: PhantomData<T>,
}

pub struct ShardRef<'a, V> {
    guard: RwLockReadGuard<'a, Vec<V>>,
    idx: usize,
}

pub struct ShardMutRef<'a, V> {
    guard: RwLockWriteGuard<'a, Vec<V>>,
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
    fn shard(data: &T) -> Shard<T, V>;
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

    pub fn get_ref(&self) -> &V {
        &self.guard[self.idx]
    }
}

impl<'a, V> ShardMutRef<'a, V> {
    fn new(guard: RwLockWriteGuard<'a, Vec<V>>, idx: usize) -> Option<Self> {
        if idx < guard.len() {
            Some(Self { guard, idx })
        } else {
            None
        }
    }

    /// NOTE: This method uses a [`RwLockWriteGuard`] which blocks reads and write
    /// during the access of the mutable reference to the `V` item
    pub fn get_mut_ref(&mut self) -> &mut V {
        &mut self.guard[self.idx]
    }
}

/// This trait permits you to validate via [`Writable`] and [`Readable`] the possibility
/// of writing to file your shards. Do prefer this trait for writing and reading your shards
pub trait ShardFormat<W: Write, T, D, E: Error> {
    fn write_shard<V: Writable>(&mut self, shard: &Shard<T, V>) -> Result<(), E>;
    fn read_shard<V: Readable>(&mut self) -> Result<Shard<T, V>, E>;
}

#[cfg(test)]
mod tests;
