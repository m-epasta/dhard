//! A concurrent hashmap using cache-line-chained bins, inspired by the DLHT paper
//! ("DLHT: A Non-blocking Resizable Hashtable with Fast Deletes and Memory-awareness",
//! HPDC'24, <https://arxiv.org/abs/2406.09986>).
//!
//! Like the paper's design, each shard stores its index as an array of 64-byte
//! cache-line-aligned buckets holding fingerprint slots that point into a side arena,
//! keys hash once (shard from the high bits, bin from the low bits), gets scan at most
//! a small bounded chain of buckets, and deletes reclaim their slot instantly through
//! the arena free list.
//!
//! What is deliberately simplified compared to the paper: operations take the shard's
//! [`parking_lot::RwLock`] instead of CAS-ing packed bucket headers, reads are not
//! seqlock-validated, resizes stop the shard rather than migrating bins in parallel,
//! and no epoch-based garbage collection is needed since values live in the arena.
//! The memory-access-aware layout (one hash, one cache line per get, instant slot
//! reuse) is what carries over.

use std::hash::{BuildHasher, Hash};

use crossbeam_utils::CachePadded;
use parking_lot::{MappedRwLockReadGuard, RwLock, RwLockReadGuard};

use crate::hashing::ShardBuildHasher;

const SLOTS_PER_BUCKET: usize = 7;
const CHAIN_DEPTH: usize = 4;
const GROW_THRESHOLD_NUM: usize = 3;
const GROW_THRESHOLD_DEN: usize = 5;
const OCC_MASK: u64 = (1 << SLOTS_PER_BUCKET) - 1;
const HW_SHIFT: u64 = 8;

#[inline]
fn occ_of(meta: u64) -> u64 {
    meta & OCC_MASK
}

#[inline]
fn hw_of(meta: u64) -> u64 {
    (meta >> HW_SHIFT) & OCC_MASK
}

/// One fingerprint slot: nonzero fingerprint in the high 32 bits, arena index in the
/// low 32 bits. An all-zero word marks an empty slot.
#[derive(Clone, Copy)]
#[repr(transparent)]
struct Slot(u64);

impl Slot {
    const EMPTY: Slot = Slot(0);

    #[inline]
    fn pack(fp: u32, idx: u32) -> Slot {
        Slot(((fp as u64) << 32) | idx as u64)
    }

    #[inline]
    fn fp(&self) -> u32 {
        (self.0 >> 32) as u32
    }

    #[inline]
    fn idx(&self) -> u32 {
        self.0 as u32
    }

    #[inline]
    #[allow(dead_code)]
    fn is_empty(&self) -> bool {
        self.0 == 0
    }
}

/// Exactly one cache line: packed metadata plus seven fingerprint slots. The low bits
/// of `meta` mark occupied slots; bits 8-14 hold the bucket's high-water frontier —
/// the number of slots ever written. Deletes only clear occupancy bits, so scans skip
/// tombstoned slots via the mask without rescanning dead words.
#[derive(Clone, Copy)]
#[repr(C, align(64))]
struct Bucket {
    meta: u64,
    slots: [Slot; SLOTS_PER_BUCKET],
}

impl Bucket {
    fn new() -> Self {
        Bucket {
            meta: 0,
            slots: [Slot::EMPTY; SLOTS_PER_BUCKET],
        }
    }
}

struct Inner<K, V> {
    buckets: Vec<Bucket>,
    mask: usize,
    entries: Vec<Option<(K, V)>>,
    free: Vec<u32>,
    count: usize,
}

#[inline]
fn fp_of(digest: u64) -> u32 {
    let fp = (digest >> 32) as u32;
    if fp == 0 { 1 } else { fp }
}

#[inline]
fn bin_of(digest: u64, mask: usize) -> usize {
    (digest as u32 as usize) & mask
}

/// Slot coordinates plus arena index of a located key
type Hit = (usize, usize, u32);

impl<K, V> Inner<K, V> {
    fn new(buckets_len: usize, entry_capacity: usize) -> Self {
        let buckets_len = buckets_len.max(16).next_power_of_two();
        Inner {
            buckets: vec![Bucket::new(); buckets_len],
            mask: buckets_len - 1,
            entries: Vec::with_capacity(entry_capacity),
            free: Vec::new(),
            count: 0,
        }
    }
}

impl<K: Eq + Hash, V> Inner<K, V> {
    /// Locates `key` inside the chain rooted at `bin`, returning its slot coordinates
    /// and arena index
    fn locate(&self, bin: usize, fp: u32, key: &K) -> Option<Hit> {
        for d in 0..CHAIN_DEPTH {
            let bidx = (bin + d) & self.mask;
            let bucket = &self.buckets[bidx];
            let mut bits = occ_of(bucket.meta) & ((1 << hw_of(bucket.meta)) - 1);
            while bits != 0 {
                let s = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                let slot = bucket.slots[s];
                if slot.fp() != fp {
                    continue;
                }
                if let Some((k, _)) = &self.entries[slot.idx() as usize]
                    && k == key
                {
                    return Some((bidx, s, slot.idx()));
                }
            }
        }
        None
    }

    /// Single pass over the chain that both locates `key` and records the first free
    /// slot, so inserts walk each cache line at most once
    fn scan(
        &self,
        bin: usize,
        fp: u32,
        key: &K,
    ) -> (Option<Hit>, Option<(usize, usize)>) {
        let mut first_free = None;
        for d in 0..CHAIN_DEPTH {
            let bidx = (bin + d) & self.mask;
            let bucket = &self.buckets[bidx];
            let meta = bucket.meta;
            let hw = hw_of(meta);
            let mut bits = occ_of(meta) & ((1 << hw) - 1);
            while bits != 0 {
                let s = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                let slot = bucket.slots[s];
                if slot.fp() != fp {
                    continue;
                }
                if let Some((k, _)) = &self.entries[slot.idx() as usize]
                    && k == key
                {
                    return (Some((bidx, s, slot.idx())), first_free);
                }
            }
            if first_free.is_none() {
                let gaps = !occ_of(meta) & ((1 << hw) - 1);
                let target = if gaps != 0 {
                    gaps.trailing_zeros() as usize
                } else if hw < SLOTS_PER_BUCKET as u64 {
                    hw as usize
                } else {
                    continue; // bucket exhausted, try the next one
                };
                first_free = Some((bidx, target));
            }
        }
        (None, first_free)
    }

    /// Marks slot `s` of a bucket occupied, advancing its high-water frontier when the
    /// frontier itself was claimed
    fn claim(&mut self, bidx: usize, s: usize) {
        let meta = self.buckets[bidx].meta;
        let hw = hw_of(meta);
        self.buckets[bidx].meta = if s as u64 == hw {
            meta | (1 << s) | ((hw + 1) << HW_SHIFT)
        } else {
            meta | (1 << s)
        };
    }

    /// Packs `(fp, idx)` into the chain rooted at `bin`, reusing a tombstoned gap or
    /// appending at the frontier of the earliest bucket with room
    fn place(&mut self, bin: usize, fp: u32, idx: u32) -> bool {
        for d in 0..CHAIN_DEPTH {
            let bidx = (bin + d) & self.mask;
            let meta = self.buckets[bidx].meta;
            let hw = hw_of(meta);
            let gaps = !occ_of(meta) & ((1 << hw) - 1);
            let target = if gaps != 0 {
                gaps.trailing_zeros() as usize
            } else if hw < SLOTS_PER_BUCKET as u64 {
                hw as usize
            } else {
                continue;
            };
            self.buckets[bidx].slots[target] = Slot::pack(fp, idx);
            self.claim(bidx, target);
            return true;
        }
        false
    }

    fn grow(&mut self, build_hasher: &ShardBuildHasher)
    where
        K: Hash,
    {
        let new_len = self.buckets.len() * 2;
        self.buckets = vec![Bucket::new(); new_len];
        self.mask = new_len - 1;

        for idx in 0..self.entries.len() {
            let Some((k, _)) = &self.entries[idx] else {
                continue;
            };
            let digest = build_hasher.hash_one(k);
            let fp = fp_of(digest);
            while !self.place(bin_of(digest, self.mask), fp, idx as u32) {
                self.grow(build_hasher);
            }
        }
    }
}

/// A concurrent hashmap sharding keys into cache-line-chained bin tables.
///
/// Each operation hashes its key exactly once: the digest's high bits select the
/// shard, the low bits select the primary bin inside that shard's table. Lookups scan
/// the bin's chain of up to four cache-line-aligned buckets comparing
/// fingerprints before touching the side arena, so most hits cost a single bucket
/// access. Removals recycle both the slot and the arena position immediately.
///
/// NOTE: [`RwShardedDlhtMap::len`] and [`RwShardedDlhtMap::is_empty`] visit every
/// shard, they are O(num_shards) rather than O(1).
pub struct RwShardedDlhtMap<K, V> {
    shards: Vec<CachePadded<RwLock<Inner<K, V>>>>,
    build_hasher: ShardBuildHasher,
}

impl<K, V> RwShardedDlhtMap<K, V> {
    /// Creates a new [`RwShardedDlhtMap`] with `num_shards` empty shards
    pub fn new(num_shards: usize) -> Self {
        Self::with_capacity(num_shards, 0)
    }

    /// Creates a new [`RwShardedDlhtMap`] with `num_shards` shards, pre-allocating space
    /// for about `capacity_hint` items in total
    pub fn with_capacity(num_shards: usize, capacity_hint: usize) -> Self {
        let per_shard = capacity_hint.checked_div(num_shards.max(1)).unwrap_or(0);
        let effective_slots_per_bucket =
            SLOTS_PER_BUCKET * GROW_THRESHOLD_NUM / GROW_THRESHOLD_DEN;
        let buckets_hint = per_shard
            .div_ceil(effective_slots_per_bucket)
            .max(1);
        Self {
            shards: (0..num_shards)
                .map(|_| {
                    CachePadded::new(RwLock::new(Inner::new(
                        buckets_hint.next_power_of_two(),
                        per_shard,
                    )))
                })
                .collect(),
            build_hasher: ShardBuildHasher::random(),
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
        self.shards.iter().map(|shard| shard.read().count).sum()
    }

    /// Returns whether or not the map contains no entries
    pub fn is_empty(&self) -> bool {
        self.shards.iter().all(|shard| shard.read().count == 0)
    }

    /// Asserts the structural invariants of every shard's bucket chains
    #[cfg(test)]
    fn check_invariants(&self) {
        for shard in &self.shards {
            let inner = shard.read();
            let mut occupied = 0usize;
            for bucket in &inner.buckets {
                let meta = bucket.meta;
                let hw = hw_of(meta);
                assert!(hw <= SLOTS_PER_BUCKET as u64, "frontier within range");
                let occ = occ_of(meta);
                assert_eq!(
                    meta & !(OCC_MASK | (OCC_MASK << HW_SHIFT)),
                    0,
                    "meta bits outside occupancy/frontier fields"
                );
                assert_eq!(occ & !((1 << hw) - 1), 0, "occupancy below frontier");
                for s in 0..hw as usize {
                    if occ & (1 << s) == 0 {
                        continue; // tombstoned gap
                    }
                    assert!(!bucket.slots[s].is_empty(), "live slot must be set");
                }
                for s in hw as usize..SLOTS_PER_BUCKET {
                    assert!(bucket.slots[s].is_empty(), "beyond frontier must be empty");
                }
                occupied += occ.count_ones() as usize;
            }
            assert_eq!(occupied, inner.count, "slot occupancy must match count");

            let mut live = vec![false; inner.entries.len()];
            for (i, entry) in inner.entries.iter().enumerate() {
                live[i] = entry.is_some();
            }
            for &free in &inner.free {
                assert!(!live[free as usize], "freed index must not hold an entry");
                live[free as usize] = true;
            }
        }
    }
}

impl<K, V> RwShardedDlhtMap<K, V>
where
    K: Eq + Hash,
{
    #[inline]
    fn hash(&self, key: &K) -> u64 {
        self.build_hasher.hash_one(key)
    }

    #[inline]
    fn shard_index(&self, digest: u64) -> Option<usize> {
        let n = self.shards.len();
        if n == 0 {
            return None;
        }
        // Multiply-shift range reduction (Lemire): cheaper than integer division
        Some((((digest >> 32) * n as u64) >> 32) as usize)
    }

    /// Inserts a key-value pair into the map, returning the previous value if the key
    /// was already present. Grows the target shard's table when occupancy exceeds 60%.
    ///
    /// # Panics
    /// Panics if the map was created with zero shards.
    pub fn insert(&self, key: K, value: V) -> Option<V> {
        let digest = self.hash(&key);
        let fp = fp_of(digest);
        let shard_idx = self
            .shard_index(digest)
            .expect("map must have at least one shard");
        let mut inner = self.shards[shard_idx].write();

        let (hit, first_free) = inner.scan(bin_of(digest, inner.mask), fp, &key);
        if let Some((.., idx)) = hit {
            let old = inner.entries[idx as usize].replace((key, value));
            return old.map(|(_, v)| v);
        }

        let mut grew = false;
        if (inner.count + 1) * GROW_THRESHOLD_DEN
            > inner.buckets.len() * SLOTS_PER_BUCKET * GROW_THRESHOLD_NUM
        {
            inner.grow(&self.build_hasher);
            grew = true;
        }

        // Reserve the arena slot first, left empty, so a grow triggered by
        // placement below does not re-place this entry
        let idx = match inner.free.pop() {
            Some(i) => i,
            None => {
                inner.entries.push(None);
                (inner.entries.len() - 1) as u32
            }
        };
        if let (false, Some((bidx, s))) = (grew, first_free) {
            inner.buckets[bidx].slots[s] = Slot::pack(fp, idx);
            inner.claim(bidx, s);
        } else {
            // A rebuild invalidated scan's coordinates, or the chain filled past the
            // global threshold — walk (and grow) until the entry fits
            let mut bin = bin_of(digest, inner.mask);
            while !inner.place(bin, fp, idx) {
                inner.grow(&self.build_hasher);
                bin = bin_of(digest, inner.mask);
            }
        }
        inner.entries[idx as usize] = Some((key, value));
        inner.count += 1;
        None
    }
    /// Returns a clone of the value associated with `key`, if present
    pub fn get_cloned(&self, key: &K) -> Option<V>
    where
        V: Clone,
    {
        let digest = self.hash(key);
        let shard_idx = self.shard_index(digest)?;
        let inner = self.shards[shard_idx].read();
        let (_, _, idx) = inner.locate(bin_of(digest, inner.mask), fp_of(digest), key)?;
        inner.entries[idx as usize].as_ref().map(|(_, v)| v.clone())
    }

    /// Returns a reference to the value associated with `key`, valid while the returned
    /// guard is held
    ///
    /// NOTE: The guard holds the read lock of the whole shard, blocking writers on that
    /// shard until dropped. Use [`RwShardedDlhtMap::get_cloned`] to avoid holding it.
    pub fn get<'map>(&'map self, key: &K) -> Option<MappedRwLockReadGuard<'map, V>> {
        let digest = self.hash(key);
        let shard_idx = self.shard_index(digest)?;
        let idx = {
            let inner = self.shards[shard_idx].read();
            let (_, _, idx) = inner.locate(bin_of(digest, inner.mask), fp_of(digest), key)?;
            idx
        };
        let inner = self.shards[shard_idx].read();
        RwLockReadGuard::try_map(inner, |inner: &Inner<K, V>| {
            inner
                .entries
                .get(idx as usize)
                .and_then(|entry| entry.as_ref())
                .map(|(_, v)| v)
        })
        .ok()
    }

    /// Removes the entry for `key`, returning its value if it was present. The slot and
    /// arena position become reusable immediately.
    pub fn remove(&self, key: &K) -> Option<V> {
        let digest = self.hash(key);
        let shard_idx = self.shard_index(digest)?;
        let mut inner = self.shards[shard_idx].write();
        let bin = bin_of(digest, inner.mask);
        let (bidx, s, idx) = inner.locate(bin, fp_of(digest), key)?;

        inner.buckets[bidx].meta &= !(1 << s);
        let old = inner.entries[idx as usize].take();
        inner.free.push(idx);
        inner.count -= 1;
        old.map(|(_, v)| v)
    }

    /// Returns whether or not the map contains an entry for `key`
    pub fn contains_key(&self, key: &K) -> bool {
        let digest = self.hash(key);
        let Some(shard_idx) = self.shard_index(digest) else {
            return false;
        };
        let inner = self.shards[shard_idx].read();
        inner
            .locate(bin_of(digest, inner.mask), fp_of(digest), key)
            .is_some()
    }
}

impl<K, V> Default for RwShardedDlhtMap<K, V> {
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
        let map: RwShardedDlhtMap<u32, String> = RwShardedDlhtMap::new(8);
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
        let map: RwShardedDlhtMap<u32, u32> = RwShardedDlhtMap::new(4);

        assert_eq!(map.insert(7, 70), None);
        assert_eq!(map.len(), 1);
        assert_eq!(map.insert(7, 71), Some(70));
        assert_eq!(map.len(), 1);
        assert_eq!(map.get_cloned(&7), Some(71));
    }

    #[test]
    fn test_remove_reclaims_slot_and_invalidates_key() {
        let map: RwShardedDlhtMap<u32, u32> = RwShardedDlhtMap::with_capacity(2, 64);

        assert_eq!(map.remove(&1), None);
        map.insert(1, 10);
        map.insert(2, 20);
        assert_eq!(map.remove(&1), Some(10));

        assert_eq!(map.get_cloned(&1), None);
        assert!(!map.contains_key(&1));
        assert_eq!(map.remove(&1), None);
        assert_eq!(map.len(), 1);

        for i in 0..32u32 {
            map.insert(100 + i, i);
        }
        assert_eq!(map.len(), 33);
        assert_eq!(map.get_cloned(&2), Some(20));
        assert_eq!(map.get_cloned(&131), Some(31));
    }

    #[test]
    fn test_many_entries_across_shards() {
        let map = RwShardedDlhtMap::<u64, u64>::with_capacity(16, 10_000);

        for i in 0..10_000u64 {
            assert_eq!(map.insert(i, i * 2), None);
        }
        assert_eq!(map.len(), 10_000);
        for i in 0..10_000u64 {
            assert_eq!(map.get_cloned(&i), Some(i * 2));
        }
    }

    #[test]
    fn test_growth_preserves_entries() {
        let map = RwShardedDlhtMap::<usize, usize>::new(2);

        for i in 0..50_000 {
            assert_eq!(map.insert(i, i * 3), None);
        }
        assert_eq!(map.len(), 50_000);
        for i in 0..50_000 {
            assert_eq!(map.get_cloned(&i), Some(i * 3));
        }

        for i in (0..50_000).step_by(7) {
            assert_eq!(map.remove(&i), Some(i * 3));
        }
        assert_eq!(map.get_cloned(&0), None);
        assert_eq!(map.get_cloned(&1), Some(3));
    }

    #[test]
    fn test_churn_recycles_slots() {
        let map: RwShardedDlhtMap<usize, usize> = RwShardedDlhtMap::with_capacity(4, 128);

        for round in 0..10 {
            for v in 0..200 {
                map.insert(v + round * 1000, v);
            }
            assert_eq!(map.len(), 200);
            for v in 0..200 {
                assert_eq!(map.remove(&(v + round * 1000)), Some(v));
            }
            assert!(map.is_empty());
        }
    }

    #[test]
    fn test_get_and_guards() {
        let map: RwShardedDlhtMap<u32, Vec<u32>> = RwShardedDlhtMap::new(4);

        assert!(map.get(&1).is_none());

        map.insert(1, vec![1]);
        {
            let guard = map.get(&1).unwrap();
            assert_eq!(guard.as_slice(), &[1]);
        }
        assert_eq!(map.get_cloned(&1), Some(vec![1]));
    }

    #[test]
    fn test_concurrent_disjoint_inserts_and_removes() {
        const THREADS: usize = 8;
        const PER_THREAD: usize = 5_000;

        let map = Arc::new(RwShardedDlhtMap::<usize, usize>::with_capacity(
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
                (0..PER_THREAD)
                    .map(|i| m.get_cloned(&(t * PER_THREAD + i)).unwrap())
                    .collect::<Vec<_>>()
            }));
        }
        for h in handles {
            let values = h.join().unwrap();
            assert!(values.iter().all(|v| v % 3 == 0));
        }

        assert_eq!(map.len(), THREADS * PER_THREAD);

        let mut handles = vec![];
        for t in 0..THREADS {
            let m = Arc::clone(&map);
            handles.push(thread::spawn(move || {
                for i in 0..PER_THREAD {
                    if i % 2 == t % 2 {
                        assert!(m.remove(&(t * PER_THREAD + i)).is_some());
                    }
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }

        assert_eq!(map.len(), THREADS * PER_THREAD / 2);
    }

    #[test]
    fn test_default_has_shards() {
        let map: RwShardedDlhtMap<u8, u8> = RwShardedDlhtMap::default();
        assert_eq!(map.num_shards(), 32);
        assert!(map.is_empty());
    }

    #[test]
    fn test_chain_invariants_hold_under_churn() {
        let map = RwShardedDlhtMap::<usize, usize>::with_capacity(2, 4_096);

        let mut state = 0x1234_5678_u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let mut live = std::collections::HashSet::new();

        for step in 0..20_000usize {
            let k = (next() % 8_192) as usize;
            if live.contains(&k) && step % 3 == 0 {
                assert_eq!(map.remove(&k), Some(k * 2));
                live.remove(&k);
            } else {
                map.insert(k, k * 2);
                live.insert(k);
            }
            if step % 512 == 0 {
                map.check_invariants();
            }
        }

        assert_eq!(map.len(), live.len());
        for k in &live {
            assert_eq!(map.get_cloned(k), Some(k * 2));
        }
        map.check_invariants();
    }
}
