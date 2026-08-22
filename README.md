# DHARD - Data structure for writing shard that persists on disk

[![CI](https://github.com/m-epasta/dhard/actions/workflows/ci.yml/badge.svg)](https://github.com/m-epasta/dhard/actions/workflows/ci.yml)
Dhard provides data structures and traits for sharding data in memory, built so shards
can later be persisted on disk.
Unlike most of libraries, the business logic/implementation is up to you.
Why? Dhard is made to work with **any** type (note that to be able to read
from disk the type must be sized).

The library is shipped with 2 layers:

- **Sharding primitives**:  `ShardCollection` distributes items round-robin across
  cache-line-padded shards, each guarding its own items behind a `parking_lot` RwLock;
- **Concurrent data structures**: a read/delete-optimized map and a
  slot map, all built on the same sharding pattern.
- (unimplemented) IO persisting shards: primitives for writing and reading to/from
file shards

## Requirements

Rust edition 2024 and rustc version >= 1.88.0

## Quick start

```rust
use dhard::ShardCollection;

let collection: ShardCollection<u32> = ShardCollection::new(4);

let (shard_idx, item_idx) = collection.push(42).expect("collection has shards");
let item = collection
    .get_shard(shard_idx)
    .and_then(|shard| shard.get_cloned(item_idx));

assert_eq!(item, Some(42));
```

## Custom sharding logic

Implement `ShardExt` to define how a piece of data is divided into shards.
The number of shards is derived from a threshold of max items per shard, which
defaults to `ShardExt::THRESHOLD` and can be overridden per call via
`ShardExt::shard_with`.

```rust
use std::collections::HashMap;
use std::hash::Hash;

use dhard::{ShardCollection, ShardExt};

struct Chunked<K, V>(HashMap<K, V>);

impl<K, V> ShardExt<HashMap<K, V>> for Chunked<K, V>
where
    K: Clone + Eq + Hash,
    V: Clone,
{
    type Item = (K, V);

    fn shard_with(data: &HashMap<K, V>, threshold: usize) -> ShardCollection<(K, V)> {
        let num_shards = data.len().div_ceil(threshold.max(1)).max(1);
        let shards = ShardCollection::new(num_shards);
        for (k, v) in data {
            shards.push((k.clone(), v.clone()));
        }
        shards
    }
}

let map: HashMap<u32, u32> = (0..100).map(|i| (i, i)).collect();
let shards = Chunked::shard(&map);

assert_eq!(shards.len(), 100);
assert!(shards.num_shards() > 1);
```

## Concurrent data structures

- **`RwShardedDlhtMap`**:  read/delete specialist: fingerprints in cache-line-chained
  bins point into a side arena, hashing each key once. Fastest keyed lookups and
  removals, at the cost of roughly half-speed inserts.
- **`RwShardedSlotMap`**:  mints stable `SlotKey` handles at insert time
  (shard + generational slot, backed by `slotmap` arenas): O(1) lookups and removals
  without any hashing, provided deletions can present the stored handle rather than an
  arbitrary key.

| You need | Use |
| --- | --- |
| Keyed access, not heavy workload | `RwShardedDlhtMap` |
| Stable handles, O(1) remove, zero hashing | `RwShardedSlotMap` |
| Maximum raw append throughput | `ShardCollection` |
| Single-threaded keyed storage | `slotmap::SlotMap` |

To learn more about the library, you can check the [docs](https://docs.rs/dhard)

## Benchmarks

## LICENSE

Dhard is licensed as MIT or APACHE 2.0, you will find the licenses
at:

- [MIT LICENSE](./LICENSE-MIT)
- [APACHE 2.0 LICENSE](./LICENSE-APACHE)
