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

## When to shard?

Sharding your types exposes a tradeoff where you lose homogeneous, simple type and raw
performances (not in any case) in favor of independant "partitions" of your type. For
example, in a message broker (e.g: apache kafka) you'll have different topics which are
genuinely the same thing, but contains independant data. Here, the shards are the topics
and your type is the message broker.

So sharding is an abstaction that has a non negligble impact, make sure you really need
one when you need, this crate offers you a working shard system that you can implement
easily on your types. ALSO, dhard is designed for being thread-safe so you may not want
dhard because of the performance cost of the atomics counters.

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

- **`RwShardedSlotMap`**:  mints stable `SlotKey` handles at insert time
  (shard + generational slot, backed by `slotmap` arenas): O(1) lookups and removals
  without any hashing, provided deletions can present the stored handle rather than an
  arbitrary key.

| You need | Use |
| --- | --- |
| Stable handles, O(1) remove, zero hashing | `RwShardedSlotMap` |
| Maximum raw append throughput | `ShardCollection` |
| Single-threaded keyed storage | `slotmap::SlotMap` |
| Stable handles, O(1) remove, zero hashing, single-threaded | `ShardedSlotMap` |

To learn more about the library, you can check the [docs](https://docs.rs/dhard)

## Benchmarks

Single-threaded versus 8-thread contested throughput (10,000,000 ops per benchmark),
each structure measured where it shines: the bare `slotmap::SlotMap` and the
sync-less `ShardedSlotMap` single-threaded, the locked concurrent `RwShardedSlotMap`
across 8 contending threads. Values are ops/sec (higher is better).

| Structure | Mode | Writes (ops/s) | Reads (ops/s) | Removals (ops/s) |
| --- | --- | --- | --- | --- |
| `slotmap::SlotMap` | single-threaded | 237587603.90 | 291149667.35 | 211548513.75 |
| `ShardedSlotMap` | single-threaded | 27257313.39 | 235283349.97 | 136107894.58 |
| `RwShardedSlotMap` | 8-thread contended | 39707126.75 | 49206507.67 | 62471223.41 |

## LICENSE

Dhard is licensed as MIT or APACHE 2.0, you will find the licenses
at:

- [MIT LICENSE](./LICENSE-MIT)
- [APACHE 2.0 LICENSE](./LICENSE-APACHE)
