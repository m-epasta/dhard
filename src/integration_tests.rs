//! this test file contains a "real" usage of the library that can be taken as an example

use std::collections::HashMap;

use crate::{ShardCollection, ShardExt};

struct _RwShardedHashMap;

type RwShardedHashMap<T> = ShardCollection<_RwShardedHashMap, HashMap<T, T>>;

// Based on the size of the Hashamp we will divide it to have more likely maximum 32 entries per Shard
impl<T, V> ShardExt<T, V> for RwShardedHashMap<T> {
    fn shard(data: &T) -> ShardCollection<T, V> {
        todo!()
    }
}
