//! A concurrent hashmap built on hash-based sharding.
//!
//! Each shard wraps its own [`std::collections::HashMap`] behind a cache-line padded
//! [`parking_lot::RwLock`]. Keys are hashed to select a shard, so operations on disjoint
//! keys only ever contend on the lock of the matching shard.
//!
//! Hashing happens twice by design: once with a cryptographically seeded
//! [`RandomState`] to pick the shard (DoS-safe placement), then a cheap multiply-xor
//! pass (a seeded multiply-xor hasher) to index the per-shard table.

use std::collections::HashMap;
use std::collections::hash_map::RandomState;
use std::hash::{BuildHasher, Hash};

pub(crate) use crate::hashing::ShardBuildHasher;
use crossbeam_utils::CachePadded;
use parking_lot::{
    MappedRwLockReadGuard, MappedRwLockWriteGuard, RwLock, RwLockReadGuard, RwLockWriteGuard,
};

/// A concurrent hashmap sharded by key hash.
///
/// Operations only ever lock the shard matching the key's hash: writers take the
/// shard's write lock, readers share the read lock. Shards are cache-line padded so
/// concurrently accessed shards do not invalidate each other's cache lines.
///
/// NOTE: [`RwShardedHashMap::len`] and [`RwShardedHashMap::is_empty`] visit every
/// shard, they are O(num_shards) rather than O(1).
pub struct RwShardedHashMap<K, V> {
    shards: Vec<CachePadded<RwLock<HashMap<K, V, ShardBuildHasher>>>>,
    build_hasher: RandomState,
}

impl<K, V> RwShardedHashMap<K, V> {
    /// Creates a new [`RwShardedHashMap`] with `num_shards` empty shards
    pub fn new(num_shards: usize) -> Self {
        Self::with_capacity(num_shards, 0)
    }

    /// Creates a new [`RwShardedHashMap`] with `num_shards` shards, pre-allocating space
    /// for about `capacity_hint` items in total
    pub fn with_capacity(num_shards: usize, capacity_hint: usize) -> Self {
        let per_shard = capacity_hint.checked_div(num_shards).unwrap_or(0);
        let inner_seed = ShardBuildHasher::random();
        Self {
            shards: (0..num_shards)
                .map(|_| {
                    CachePadded::new(RwLock::new(HashMap::with_capacity_and_hasher(
                        per_shard, inner_seed,
                    )))
                })
                .collect(),
            build_hasher: RandomState::new(),
        }
    }

    /// Returns the number of shards
    #[inline]
    pub fn num_shards(&self) -> usize {
        self.shards.len()
    }

    /// Returns the total number of entries across all shards
    ///
    /// NOTE: This visits every shard, acquiring each read lock briefly.
    pub fn len(&self) -> usize {
        self.shards.iter().map(|shard| shard.read().len()).sum()
    }

    /// Returns whether or not the map contains no entries
    pub fn is_empty(&self) -> bool {
        self.shards.iter().all(|shard| shard.read().is_empty())
    }
}

impl<K, V> RwShardedHashMap<K, V>
where
    K: Eq + Hash,
{
    #[inline]
    fn shard_index(&self, key: &K) -> Option<usize> {
        let n = self.shards.len();
        if n == 0 {
            return None;
        }
        let digest = self.build_hasher.hash_one(key);
        Some(if n.is_power_of_two() {
            (digest as usize) & (n - 1)
        } else {
            // Multiply-shift range reduction (Lemire): cheaper than integer division
            (((digest >> 32) * n as u64) >> 32) as usize
        })
    }

    /// Inserts a key-value pair into the map, returning the previous value if the key
    /// was already present
    ///
    /// # Panics
    /// Panics if the map was created with zero shards.
    pub fn insert(&self, key: K, value: V) -> Option<V> {
        let idx = self
            .shard_index(&key)
            .expect("map must have at least one shard");
        let mut shard = self.shards[idx].write();
        shard.insert(key, value)
    }

    /// Returns a clone of the value associated with `key`, if present
    pub fn get_cloned(&self, key: &K) -> Option<V>
    where
        V: Clone,
    {
        let idx = self.shard_index(key)?;
        let shard = self.shards[idx].read();
        shard.get(key).cloned()
    }

    /// Returns a reference to the value associated with `key`, valid while the returned
    /// guard is held
    ///
    /// NOTE: The guard holds the read lock of the whole shard, blocking writers on that
    /// shard until dropped. Use [`RwShardedHashMap::get_cloned`] to avoid holding it.
    pub fn get<'map>(&'map self, key: &K) -> Option<MappedRwLockReadGuard<'map, V>> {
        let idx = self.shard_index(key)?;
        let shard = self.shards[idx].read();
        RwLockReadGuard::try_map(shard, |map| map.get(key)).ok()
    }

    /// Returns a mutable reference to the value associated with `key`, valid while the
    /// returned guard is held
    pub fn get_mut<'map>(&'map self, key: &K) -> Option<MappedRwLockWriteGuard<'map, V>> {
        let idx = self.shard_index(key)?;
        let shard = self.shards[idx].write();
        RwLockWriteGuard::try_map(shard, |map| map.get_mut(key)).ok()
    }

    /// Removes the entry for `key`, returning its value if it was present
    pub fn remove(&self, key: &K) -> Option<V> {
        let idx = self.shard_index(key)?;
        let mut shard = self.shards[idx].write();
        shard.remove(key)
    }

    /// Returns whether or not the map contains an entry for `key`
    pub fn contains_key(&self, key: &K) -> bool {
        let idx = match self.shard_index(key) {
            Some(idx) => idx,
            None => return false,
        };
        let shard = self.shards[idx].read();
        shard.contains_key(key)
    }
}

impl<K, V> Default for RwShardedHashMap<K, V> {
    fn default() -> Self {
        Self::new(32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn test_insert_and_get() {
        let map: RwShardedHashMap<u32, String> = RwShardedHashMap::new(8);
        assert!(map.is_empty());

        assert_eq!(map.insert(1, "one".to_string()), None);
        assert_eq!(map.insert(2, "two".to_string()), None);
        assert_eq!(map.len(), 2);

        assert_eq!(map.get_cloned(&1), Some("one".to_string()));
        assert_eq!(map.get_cloned(&3), None);
        assert!(map.contains_key(&2));
        assert!(!map.contains_key(&3));
    }

    #[test]
    fn test_insert_updates_in_place() {
        let map: RwShardedHashMap<u32, u32> = RwShardedHashMap::new(4);

        assert_eq!(map.insert(7, 70), None);
        assert_eq!(map.len(), 1);
        assert_eq!(map.insert(7, 71), Some(70));
        assert_eq!(map.len(), 1);
        assert_eq!(map.get_cloned(&7), Some(71));
    }

    #[test]
    fn test_remove_entries() {
        let map: RwShardedHashMap<u32, u32> = RwShardedHashMap::new(4);

        assert_eq!(map.remove(&1), None);
        map.insert(1, 10);
        assert_eq!(map.remove(&1), Some(10));
        assert_eq!(map.remove(&1), None);
        assert!(map.is_empty());

        map.insert(2, 20);
        assert_eq!(map.remove(&9), None);
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn test_many_entries_across_shards() {
        let map = RwShardedHashMap::<u64, u64>::with_capacity(16, 10_000);

        for i in 0..10_000u64 {
            assert_eq!(map.insert(i, i * 2), None);
        }
        assert_eq!(map.len(), 10_000);
        for i in 0..10_000u64 {
            assert_eq!(map.get_cloned(&i), Some(i * 2));
        }

        for i in 0..100u64 {
            assert_eq!(map.insert(i, i), Some(i * 2));
        }
        assert_eq!(map.len(), 10_000);
    }

    #[test]
    fn test_concurrent_disjoint_inserts_and_removes() {
        const THREADS: usize = 8;
        const PER_THREAD: usize = 2_000;

        let map = Arc::new(RwShardedHashMap::<usize, usize>::with_capacity(
            16,
            THREADS * PER_THREAD,
        ));
        let mut handles = vec![];

        for t in 0..THREADS {
            let m = Arc::clone(&map);
            handles.push(thread::spawn(move || {
                for i in 0..PER_THREAD {
                    let k = t * PER_THREAD + i;
                    m.insert(k, k * 3);
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }

        assert_eq!(map.len(), THREADS * PER_THREAD);
        for t in 0..THREADS {
            for i in 0..PER_THREAD {
                let k = t * PER_THREAD + i;
                assert_eq!(map.get_cloned(&k), Some(k * 3));
            }
        }

        let mut handles = vec![];
        for t in 0..THREADS {
            let m = Arc::clone(&map);
            handles.push(thread::spawn(move || {
                for i in 0..PER_THREAD {
                    let k = t * PER_THREAD + i;
                    m.remove(&k);
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }

        assert!(map.is_empty());
    }

    #[test]
    fn test_get_and_get_mut_guards() {
        let map: RwShardedHashMap<u32, Vec<u32>> = RwShardedHashMap::new(4);

        assert!(map.get(&1).is_none());
        assert!(map.get_mut(&1).is_none());

        map.insert(1, vec![1]);
        {
            let mut v = map.get_mut(&1).unwrap();
            v.push(2);
        }
        {
            let v = map.get(&1).unwrap();
            assert_eq!(v.as_slice(), &[1, 2]);
        }

        assert_eq!(map.remove(&1), Some(vec![1, 2]));
        assert!(map.get(&1).is_none());
    }

    #[test]
    fn test_default_has_shards() {
        let map: RwShardedHashMap<u8, u8> = RwShardedHashMap::default();
        assert_eq!(map.num_shards(), 32);
        assert!(map.is_empty());
    }
}
