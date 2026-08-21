//! this test file contains a "real" usage of the library that can be taken as an example

use std::collections::HashMap;

use crate::{ShardCollection, ShardExt};

type RwShardedHashMap<T> = ShardCollection<HashMap<T, T>>;

impl<T> ShardExt<HashMap<T, T>> for RwShardedHashMap<T> {
    fn shard(data: &HashMap<T, T>) -> ShardCollection<HashMap<T, T>> {
        todo!()
    }
}
