//! A concurrent slot map built on hash-free sharding.
//!
//! Unlike [`crate::RwShardedHashMap`], keys are generated at insert time instead of
//! hashed: each [`SlotKey`] pins the shard, the slot index and a generation counter,
//! giving O(1) insert, lookup and removal with stable handles. Each shard is a small
//! arena (`Vec` of slots plus a free list) behind a cache-line padded
//! [`parking_lot::RwLock`]; inserts pick a shard via an atomic round-robin ticket.

use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicUsize, Ordering};

use crossbeam_utils::CachePadded;
use parking_lot::RwLock;

/// A handle to a value stored in a [`RwShardedSlotMap`]
///
/// Keys are cheap to copy and compare, and remain valid until the value they point to
/// is removed. After removal the slot may be reused by later inserts; the generation
/// counter inside the key makes stale keys detectably invalid.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct SlotKey {
    shard: u32,
    slot: u32,
    generation: u32,
}

impl SlotKey {
    /// Index of the shard holding this key's slot
    #[inline]
    pub fn shard(&self) -> usize {
        self.shard as usize
    }

    /// Raw slot index within the shard (not unique across removals)
    #[inline]
    pub fn slot(&self) -> usize {
        self.slot as usize
    }

    /// Generation of the slot this key was minted in
    #[inline]
    pub fn generation(&self) -> u32 {
        self.generation
    }
}

impl fmt::Debug for SlotKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SlotKey({}:{:#x}:v{})",
            self.shard, self.slot, self.generation
        )
    }
}

impl Hash for SlotKey {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        (self.shard, self.slot, self.generation).hash(state);
    }
}

struct SlotEntry<V> {
    generation: u32,
    value: Option<V>,
}

struct SlotArena<V> {
    entries: Vec<SlotEntry<V>>,
    free: Vec<u32>,
    count: usize,
}

impl<V> SlotArena<V> {
    fn new(capacity: usize) -> Self {
        Self {
            entries: Vec::with_capacity(capacity),
            free: Vec::new(),
            count: 0,
        }
    }
}

/// A concurrent slot map sharded into independent arenas.
///
/// Inserts distribute values round-robin across shards using an atomic ticket;
/// every subsequent operation on a [`SlotKey`] touches exactly one shard's lock.
/// Removed slots are recycled through a per-shard free list, so memory stays
/// bounded by peak occupancy.
///
/// NOTE: [`RwShardedSlotMap::len`] and [`RwShardedSlotMap::is_empty`] visit every
/// shard, they are O(num_shards) rather than O(1).
pub struct RwShardedSlotMap<V> {
    shards: Vec<CachePadded<RwLock<SlotArena<V>>>>,
    next_ticket: CachePadded<AtomicUsize>,
}

impl<V> RwShardedSlotMap<V> {
    /// Creates a new [`RwShardedSlotMap`] with `num_shards` empty shards
    pub fn new(num_shards: usize) -> Self {
        Self::with_capacity(num_shards, 0)
    }

    /// Creates a new [`RwShardedSlotMap`] with `num_shards` shards, pre-allocating space
    /// for about `capacity_hint` items in total
    pub fn with_capacity(num_shards: usize, capacity_hint: usize) -> Self {
        let per_shard = capacity_hint.checked_div(num_shards).unwrap_or(0);
        Self {
            shards: (0..num_shards)
                .map(|_| CachePadded::new(RwLock::new(SlotArena::new(per_shard))))
                .collect(),
            next_ticket: CachePadded::new(AtomicUsize::new(0)),
        }
    }

    /// Returns the number of shards
    #[inline]
    pub fn num_shards(&self) -> usize {
        self.shards.len()
    }

    /// Returns the total number of live values across all shards
    ///
    /// NOTE: This visits every shard, acquiring each read lock briefly.
    pub fn len(&self) -> usize {
        self.shards.iter().map(|shard| shard.read().count).sum()
    }

    /// Returns whether or not the map holds no values
    pub fn is_empty(&self) -> bool {
        self.shards.iter().all(|shard| shard.read().count == 0)
    }

    #[inline]
    fn shard_index(&self) -> Option<usize> {
        let n = self.shards.len();
        if n == 0 {
            return None;
        }
        let ticket = self.next_ticket.fetch_add(1, Ordering::Relaxed);
        Some(if n.is_power_of_two() {
            ticket & (n - 1)
        } else {
            ticket % n
        })
    }
}

impl<V> RwShardedSlotMap<V> {
    /// Inserts a value into the map and returns its stable [`SlotKey`]
    ///
    /// # Panics
    /// Panics if the map was created with zero shards.
    pub fn insert(&self, value: V) -> SlotKey {
        let shard_idx = self
            .shard_index()
            .expect("slot map must have at least one shard");
        let mut arena = self.shards[shard_idx].write();

        match arena.free.pop() {
            Some(slot) => {
                let entry = &mut arena.entries[slot as usize];
                entry.value = Some(value);
                let key_generation = entry.generation;
                arena.count += 1;
                SlotKey {
                    shard: shard_idx as u32,
                    slot,
                    generation: key_generation,
                }
            }
            None => {
                let slot = arena.entries.len() as u32;
                arena.entries.push(SlotEntry {
                    generation: 0,
                    value: Some(value),
                });
                arena.count += 1;
                SlotKey {
                    shard: shard_idx as u32,
                    slot,
                    generation: 0,
                }
            }
        }
    }
}

impl<V> RwShardedSlotMap<V> {
    /// Returns a clone of the value behind `key`, if it is still live
    pub fn get_cloned(&self, key: SlotKey) -> Option<V>
    where
        V: Clone,
    {
        let shard = self.shards.get(key.shard as usize)?;
        let arena = shard.read();
        arena
            .entries
            .get(key.slot as usize)
            .filter(|entry| entry.generation == key.generation && entry.value.is_some())
            .and_then(|entry| entry.value.clone())
    }

    /// Returns whether or not `key` still points to a live value
    pub fn contains(&self, key: SlotKey) -> bool {
        let Some(shard) = self.shards.get(key.shard as usize) else {
            return false;
        };
        let arena = shard.read();
        arena
            .entries
            .get(key.slot as usize)
            .is_some_and(|entry| entry.generation == key.generation && entry.value.is_some())
    }

    /// Removes the value behind `key`, returning it if the key was live.
    /// The slot becomes reusable by later inserts; using `key` afterwards returns
    /// `None`
    pub fn remove(&self, key: SlotKey) -> Option<V> {
        let mut arena = self.shards.get(key.shard as usize)?.write();
        let entry = arena.entries.get_mut(key.slot as usize)?;
        if entry.generation != key.generation || entry.value.is_none() {
            return None;
        }
        entry.generation = entry.generation.wrapping_add(1);
        let value = entry.value.take();
        arena.free.push(key.slot);
        arena.count -= 1;
        value
    }
}

impl<V> Default for RwShardedSlotMap<V> {
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
    fn test_insert_and_get_cloned() {
        let map: RwShardedSlotMap<String> = RwShardedSlotMap::new(8);
        assert!(map.is_empty());

        let k1 = map.insert("one".to_string());
        let k2 = map.insert("two".to_string());
        assert_eq!(map.len(), 2);

        assert_eq!(map.get_cloned(k1), Some("one".to_string()));
        assert_eq!(map.get_cloned(k2), Some("two".to_string()));
        assert!(map.contains(k1));
        assert!(map.contains(k2));
    }

    #[test]
    fn test_remove_invalidates_key_and_recycles_slot() {
        let map: RwShardedSlotMap<u32> = RwShardedSlotMap::new(1);

        let key = map.insert(10);
        assert_eq!(key.shard(), 0);
        assert_eq!(key.generation(), 0);
        assert_eq!(map.remove(key), Some(10));

        assert_eq!(map.get_cloned(key), None);
        assert!(!map.contains(key));
        assert_eq!(map.remove(key), None);
        assert!(map.is_empty());

        let recycled = map.insert(20);
        assert_eq!(recycled.slot(), key.slot());
        assert_eq!(recycled.generation(), 1);
        assert_ne!(recycled, key);
        assert_eq!(map.get_cloned(recycled), Some(20));
        assert_eq!(map.get_cloned(key), None);

        let fresh_slot = map.insert(30);
        assert_eq!(fresh_slot.slot(), key.slot() + 1);
        assert_eq!(map.len(), 2);
    }

    #[test]
    fn test_removed_slots_recycled_within_their_shard() {
        let map: RwShardedSlotMap<u32> = RwShardedSlotMap::new(2);

        let key = map.insert(10);
        map.insert(11);
        assert_eq!(map.remove(key), Some(10));

        let mut reused = None;
        while reused.is_none() {
            let candidate = map.insert(u32::MAX - map.len() as u32);
            if candidate.shard() == key.shard() && candidate.slot() == key.slot() {
                reused = Some(candidate);
                break;
            }
        }
        let reused = reused.unwrap();
        assert_eq!(reused.generation(), key.generation() + 1);
        assert_eq!(map.get_cloned(key), None);
    }

    #[test]
    fn test_keys_resolve_across_shards() {
        let map: RwShardedSlotMap<u64> = RwShardedSlotMap::new(4);
        let mut keys = Vec::new();
        for i in 0..100u64 {
            keys.push(map.insert(i));
        }
        for (i, key) in keys.iter().enumerate() {
            assert_eq!(map.get_cloned(*key), Some(i as u64));
        }
    }

    #[test]
    fn test_churn_reuses_slots() {
        let map: RwShardedSlotMap<usize> = RwShardedSlotMap::with_capacity(4, 64);

        for round in 0..10 {
            let keys: Vec<SlotKey> = (0..100).map(|v| map.insert(v + round)).collect();
            assert_eq!(map.len(), 100);
            for (i, key) in keys.iter().enumerate() {
                assert_eq!(map.get_cloned(*key), Some(i + round));
            }
            for key in keys {
                assert!(map.remove(key).is_some());
            }
            assert!(map.is_empty());
        }
    }

    #[test]
    fn test_concurrent_insert_get_remove() {
        const THREADS: usize = 8;
        const PER_THREAD: usize = 5_000;

        let map = Arc::new(RwShardedSlotMap::<usize>::with_capacity(
            16,
            THREADS * PER_THREAD,
        ));

        let mut handles = vec![];
        for t in 0..THREADS {
            let m = Arc::clone(&map);
            handles.push(thread::spawn(move || {
                (0..PER_THREAD)
                    .map(|i| m.insert(t * PER_THREAD + i))
                    .collect::<Vec<_>>()
            }));
        }
        let all_keys: Vec<Vec<SlotKey>> = handles.into_iter().map(|h| h.join().unwrap()).collect();

        assert_eq!(map.len(), THREADS * PER_THREAD);
        for (t, keys) in all_keys.iter().enumerate() {
            for (i, key) in keys.iter().enumerate() {
                assert_eq!(map.get_cloned(*key), Some(t * PER_THREAD + i));
            }
        }

        let mut handles = vec![];
        for (t, keys) in all_keys.iter().enumerate() {
            let m = Arc::clone(&map);
            let keys = keys.clone();
            handles.push(thread::spawn(move || {
                for (i, key) in keys.into_iter().enumerate() {
                    if i % 2 == t % 2 {
                        assert!(m.remove(key).is_some());
                    }
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }

        assert_eq!(map.len(), THREADS * PER_THREAD / 2);
        for (t, keys) in all_keys.iter().enumerate() {
            for (i, key) in keys.iter().enumerate() {
                if i % 2 == t % 2 {
                    assert_eq!(map.get_cloned(*key), None);
                } else {
                    assert_eq!(map.get_cloned(*key), Some(t * PER_THREAD + i));
                }
            }
        }
    }

    #[test]
    fn test_default_has_shards() {
        let map: RwShardedSlotMap<u8> = RwShardedSlotMap::default();
        assert_eq!(map.num_shards(), 32);
        assert!(map.is_empty());
    }
}
