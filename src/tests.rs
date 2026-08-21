use super::*;

#[derive(Debug, Default, Clone, PartialEq)]
struct TestData {
    id: u32,
    name: String,
}

type TestShard = Shard<TestData>;

fn make_item(id: u32, name: &str) -> TestData {
    TestData {
        id,
        name: name.to_string(),
    }
}

#[test]
fn test_new_is_empty() {
    let shard = TestShard::new();
    assert!(shard.is_empty());
    assert_eq!(shard.len(), 0);
}

#[test]
fn test_default_is_empty() {
    let shard = TestShard::default();
    assert!(shard.is_empty());
    assert_eq!(shard.len(), 0);
}

#[test]
fn test_push_single() {
    let shard = TestShard::new();
    shard.push(make_item(1, "a"));
    assert!(!shard.is_empty());
    assert_eq!(shard.len(), 1);
}

#[test]
fn test_push_multiple() {
    let shard = TestShard::new();
    shard.push(make_item(1, "a"));
    shard.push(make_item(2, "b"));
    shard.push(make_item(3, "c"));
    assert_eq!(shard.len(), 3);
}

#[test]
fn test_items_as_slice() {
    let shard = TestShard::new();
    shard.push(make_item(1, "a"));
    shard.push(make_item(2, "b"));

    let items = shard.items();
    let slice: &[TestData] = items.as_slice();
    assert_eq!(slice.len(), 2);
    assert_eq!(slice[0].id, 1);
    assert_eq!(slice[1].id, 2);
}

#[test]
fn test_get_cloned_valid() {
    let shard = TestShard::new();
    shard.push(make_item(42, "hello"));

    let item = shard.get_cloned(0).unwrap();
    assert_eq!(item.id, 42);
    assert_eq!(item.name, "hello");
}

#[test]
fn test_get_cloned_out_of_bounds() {
    let shard = TestShard::new();
    shard.push(make_item(1, "a"));

    assert!(shard.get_cloned(1).is_none());
    assert!(shard.get_cloned(100).is_none());
}

#[test]
fn test_get_cloned_returns_clone() {
    let shard = TestShard::new();
    shard.push(make_item(1, "original"));

    let mut item1 = shard.get_cloned(0).unwrap();
    let item2 = shard.get_cloned(0).unwrap();
    item1.name = "modified".to_string();

    assert_eq!(item1.name, "modified");
    assert_eq!(item2.name, "original");
}

#[test]
fn test_get_ref_valid() {
    let shard = TestShard::new();
    shard.push(make_item(10, "test"));

    let shard_ref = shard.get_ref(0).unwrap();
    let value = shard_ref.get_ref();
    assert_eq!(value.id, 10);
    assert_eq!(value.name, "test");
}

#[test]
fn test_get_ref_out_of_bounds() {
    let shard = TestShard::new();
    shard.push(make_item(1, "a"));

    assert!(shard.get_ref(1).is_none());
    assert!(shard.get_ref(usize::MAX).is_none());
}

#[test]
fn test_get_ref_multiple() {
    let shard = TestShard::new();
    shard.push(make_item(1, "a"));
    shard.push(make_item(2, "b"));
    shard.push(make_item(3, "c"));

    let r0 = shard.get_ref(0).unwrap();
    let r1 = shard.get_ref(1).unwrap();
    let r2 = shard.get_ref(2).unwrap();

    assert_eq!(r0.get_ref().id, 1);
    assert_eq!(r1.get_ref().id, 2);
    assert_eq!(r2.get_ref().id, 3);
}

#[test]
fn test_get_ref_first_and_last() {
    let shard = TestShard::new();
    shard.push(make_item(0, "first"));
    shard.push(make_item(1, "middle"));
    shard.push(make_item(2, "last"));

    let first = shard.get_ref(0).unwrap();
    let last = shard.get_ref(2).unwrap();

    assert_eq!(first.get_ref().name, "first");
    assert_eq!(last.get_ref().name, "last");
}

#[test]
fn test_get_mut_valid() {
    let shard = TestShard::new();
    shard.push(make_item(1, "before"));

    let mut shard_mut = shard.get_mut(0).unwrap();
    let value = shard_mut.get_mut_ref();
    value.id = 99;
    value.name = "after".to_string();
    drop(shard_mut);

    let item = shard.get_cloned(0).unwrap();
    assert_eq!(item.id, 99);
    assert_eq!(item.name, "after");
}

#[test]
fn test_get_mut_out_of_bounds() {
    let shard = TestShard::new();
    shard.push(make_item(1, "a"));

    assert!(shard.get_mut(1).is_none());
    assert!(shard.get_mut(usize::MAX).is_none());
}

#[test]
fn test_get_mut_multiple_items() {
    let shard = TestShard::new();
    shard.push(make_item(1, "a"));
    shard.push(make_item(2, "b"));
    shard.push(make_item(3, "c"));

    {
        let mut m = shard.get_mut(1).unwrap();
        m.get_mut_ref().name = "modified".to_string();
    }

    assert_eq!(shard.get_cloned(0).unwrap().name, "a");
    assert_eq!(shard.get_cloned(1).unwrap().name, "modified");
    assert_eq!(shard.get_cloned(2).unwrap().name, "c");
}

#[test]
fn test_empty_shard_operations() {
    let shard = TestShard::new();

    assert!(shard.get_cloned(0).is_none());
    assert!(shard.get_ref(0).is_none());
    assert!(shard.get_mut(0).is_none());
    assert_eq!(shard.len(), 0);
    assert!(shard.is_empty());
}

#[test]
fn test_shard_with_primitive_types() {
    let shard: Shard<i32> = Shard::new();
    shard.push(1);
    shard.push(2);
    shard.push(3);

    assert_eq!(shard.get_cloned(0).unwrap(), 1);
    assert_eq!(shard.get_cloned(1).unwrap(), 2);
    assert_eq!(shard.get_cloned(2).unwrap(), 3);
}

#[test]
fn test_shard_with_string() {
    let shard: Shard<String> = Shard::new();
    shard.push("hello".to_string());
    shard.push("world".to_string());

    assert_eq!(shard.get_cloned(0).unwrap(), "hello");
    assert_eq!(shard.get_cloned(1).unwrap(), "world");
}

#[test]
fn test_shard_with_option() {
    let shard: Shard<Option<i32>> = Shard::new();
    shard.push(Some(1));
    shard.push(None);
    shard.push(Some(3));

    assert_eq!(shard.get_cloned(0).unwrap(), Some(1));
    assert_eq!(shard.get_cloned(1).unwrap(), None);
    assert_eq!(shard.get_cloned(2).unwrap(), Some(3));
}

#[test]
fn test_push_after_get_ref() {
    let shard = TestShard::new();
    shard.push(make_item(1, "a"));

    let r = shard.get_ref(0).unwrap();
    let _val = r.get_ref();
    drop(r);

    shard.push(make_item(2, "b"));
    assert_eq!(shard.len(), 2);
}

#[test]
fn test_push_after_get_mut() {
    let shard = TestShard::new();
    shard.push(make_item(1, "a"));

    {
        let mut m = shard.get_mut(0).unwrap();
        m.get_mut_ref().id = 99;
    }

    shard.push(make_item(2, "b"));
    assert_eq!(shard.len(), 2);
    assert_eq!(shard.get_cloned(0).unwrap().id, 99);
}

#[test]
fn test_len_tracks_pushes() {
    let shard = TestShard::new();
    assert_eq!(shard.len(), 0);

    shard.push(make_item(1, "a"));
    assert_eq!(shard.len(), 1);

    shard.push(make_item(2, "b"));
    assert_eq!(shard.len(), 2);

    shard.push(make_item(3, "c"));
    assert_eq!(shard.len(), 3);
}

#[test]
fn test_is_empty_tracks_state() {
    let shard = TestShard::new();
    assert!(shard.is_empty());

    shard.push(make_item(1, "a"));
    assert!(!shard.is_empty());
}

#[test]
fn test_get_ref_boundary_index_zero() {
    let shard = TestShard::new();
    shard.push(make_item(42, "only"));

    let r = shard.get_ref(0).unwrap();
    assert_eq!(r.get_ref().id, 42);
}

#[test]
fn test_get_ref_boundary_index_at_len() {
    let shard = TestShard::new();
    shard.push(make_item(1, "a"));

    assert!(shard.get_ref(1).is_none());
}

#[test]
fn test_get_ref_boundary_index_past_len() {
    let shard = TestShard::new();
    shard.push(make_item(1, "a"));
    shard.push(make_item(2, "b"));

    assert!(shard.get_ref(2).is_none());
    assert!(shard.get_ref(1000).is_none());
}

#[test]
fn test_get_mut_boundary_index_zero() {
    let shard = TestShard::new();
    shard.push(make_item(42, "only"));

    let mut m = shard.get_mut(0).unwrap();
    m.get_mut_ref().id = 100;
    drop(m);

    assert_eq!(shard.get_cloned(0).unwrap().id, 100);
}

#[test]
fn test_get_mut_boundary_index_at_len() {
    let shard = TestShard::new();
    shard.push(make_item(1, "a"));

    assert!(shard.get_mut(1).is_none());
}

#[test]
fn test_concurrent_get_ref_and_push() {
    use std::sync::Arc;
    use std::thread;

    let shard = Arc::new(TestShard::new());
    let mut handles = vec![];

    for i in 0..10 {
        let s = Arc::clone(&shard);
        handles.push(thread::spawn(move || {
            s.push(make_item(i, &i.to_string()));
        }));
    }

    for h in handles {
        h.join().unwrap();
    }

    assert_eq!(shard.len(), 10);
}

#[test]
fn test_collection_new_creates_empty_shards() {
    let collection: ShardCollection<TestData> = ShardCollection::new(4);
    assert_eq!(collection.num_shards(), 4);
    assert!(collection.is_empty());
    assert_eq!(collection.len(), 0);
    for i in 0..4 {
        let shard = collection.get_shard(i).unwrap();
        assert!(shard.is_empty());
        assert_eq!(shard.len(), 0);
    }
}

#[test]
fn test_collection_new_zero_shards() {
    let collection: ShardCollection<u32> = ShardCollection::new(0);
    assert_eq!(collection.num_shards(), 0);
    assert!(collection.is_empty());
    assert_eq!(collection.len(), 0);
}

#[test]
fn test_collection_default_is_single_shard() {
    let collection = ShardCollection::<u32>::default();
    assert_eq!(collection.num_shards(), 1);
    assert!(collection.is_empty());
}

#[test]
fn test_collection_push_returns_indices() {
    let collection: ShardCollection<u32> = ShardCollection::new(3);
    let expected = [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (0, 2)];
    for (n, expected_idx) in expected.iter().enumerate() {
        assert_eq!(collection.push(n as u32), Some(*expected_idx));
    }
    for (n, &(s, i)) in expected.iter().enumerate() {
        assert_eq!(
            collection.get_shard(s).unwrap().get_cloned(i),
            Some(n as u32)
        );
    }
}

#[test]
fn test_collection_push_round_robin_distribution() {
    let collection: ShardCollection<u32> = ShardCollection::new(3);
    for i in 0..7 {
        collection.push(i);
    }
    let s0 = collection.get_shard(0).unwrap().items();
    let s1 = collection.get_shard(1).unwrap().items();
    let s2 = collection.get_shard(2).unwrap().items();
    assert_eq!(s0.as_slice(), &[0, 3, 6]);
    assert_eq!(s1.as_slice(), &[1, 4]);
    assert_eq!(s2.as_slice(), &[2, 5]);
}

#[test]
fn test_collection_push_on_empty_collection() {
    let collection: ShardCollection<u32> = ShardCollection::new(0);
    assert_eq!(collection.push(1), None);
    assert_eq!(collection.push(2), None);
    assert!(collection.is_empty());
}

#[test]
fn test_collection_len_sums_across_shards() {
    let collection: ShardCollection<u32> = ShardCollection::new(3);
    for i in 0..5 {
        collection.push(i);
    }
    assert_eq!(collection.len(), 5);
    assert_eq!(collection.get_shard(0).unwrap().len(), 2);
    assert_eq!(collection.get_shard(1).unwrap().len(), 2);
    assert_eq!(collection.get_shard(2).unwrap().len(), 1);
}

#[test]
fn test_collection_is_empty_tracks_all_shards() {
    let collection: ShardCollection<u32> = ShardCollection::new(2);
    assert!(collection.is_empty());
    collection.push(1);
    assert!(!collection.is_empty());
}

#[test]
fn test_collection_get_shard_bounds() {
    let collection: ShardCollection<u32> = ShardCollection::new(2);
    assert!(collection.get_shard(0).is_some());
    assert!(collection.get_shard(1).is_some());
    assert!(collection.get_shard(2).is_none());
    assert!(collection.get_shard(usize::MAX).is_none());
}

#[test]
fn test_collection_get_shard_mutation() {
    let collection: ShardCollection<TestData> = ShardCollection::new(2);
    collection.push(make_item(1, "a"));
    collection.push(make_item(2, "b"));

    {
        let shard = collection.get_shard(1).unwrap();
        let mut m = shard.get_mut(0).unwrap();
        m.get_mut_ref().name = "modified".to_string();
    }

    assert_eq!(
        collection.get_shard(0).unwrap().get_cloned(0).unwrap().name,
        "a"
    );
    assert_eq!(
        collection.get_shard(1).unwrap().get_cloned(0).unwrap().name,
        "modified"
    );
}

#[test]
fn test_collection_concurrent_pushes() {
    use std::sync::Arc;
    use std::thread;

    let collection = Arc::new(ShardCollection::<u32>::new(4));
    let mut handles = vec![];

    for _ in 0..8 {
        let c = Arc::clone(&collection);
        handles.push(thread::spawn(move || {
            for i in 0..125 {
                c.push(i);
            }
        }));
    }

    for h in handles {
        h.join().unwrap();
    }

    assert_eq!(collection.len(), 1000);
    for i in 0..4 {
        assert_eq!(collection.get_shard(i).unwrap().len(), 250);
    }
}

#[test]
fn test_release_lock_allows_relock() {
    let shard = TestShard::new();
    shard.push(make_item(1, "a"));

    let mut m = shard.get_mut(0).unwrap();
    m.release_lock();

    shard.push(make_item(2, "b"));
    assert_eq!(shard.len(), 2);

    let mut m2 = shard.get_mut(0).unwrap();
    m2.get_mut_ref().name = "modified".to_string();
    drop(m2);

    assert_eq!(shard.get_cloned(0).unwrap().name, "modified");
}

#[test]
#[should_panic(expected = "lock already released")]
fn test_get_mut_ref_after_release_panics() {
    let shard = TestShard::new();
    shard.push(make_item(1, "a"));

    let mut m = shard.get_mut(0).unwrap();
    m.release_lock();
    let _ = m.get_mut_ref();
}

#[test]
fn test_release_lock_twice_is_idempotent() {
    let shard = TestShard::new();
    shard.push(make_item(1, "a"));

    let mut m = shard.get_mut(0).unwrap();
    m.release_lock();
    m.release_lock();
    drop(m);

    assert_eq!(shard.len(), 1);
}

#[test]
fn test_get_mut_ref_repeated_calls() {
    let shard = TestShard::new();
    shard.push(make_item(1, "a"));

    let mut m = shard.get_mut(0).unwrap();
    m.get_mut_ref().id = 10;
    m.get_mut_ref().id += 5;
    assert_eq!(m.get_mut_ref().id, 15);
    drop(m);

    assert_eq!(shard.get_cloned(0).unwrap().id, 15);
}

#[test]
fn test_write_guard_blocks_concurrent_push() {
    use std::sync::{Arc, mpsc};
    use std::thread;
    use std::time::Duration;

    let shard = Arc::new(TestShard::new());
    shard.push(make_item(1, "a"));

    let guard = shard.get_mut(0).unwrap();
    let (tx, rx) = mpsc::channel();
    let s = Arc::clone(&shard);
    let pusher = thread::spawn(move || {
        s.push(make_item(2, "b"));
        tx.send(()).expect("receiver dropped");
    });

    assert!(rx.recv_timeout(Duration::from_millis(50)).is_err());
    drop(guard);
    pusher.join().unwrap();
    rx.recv_timeout(Duration::from_secs(5))
        .expect("push did not complete");
    assert_eq!(shard.len(), 2);
}

#[test]
fn test_concurrent_readers_do_not_block() {
    use std::sync::{Arc, Barrier};
    use std::thread;

    let shard = Arc::new(TestShard::new());
    for i in 0..10 {
        shard.push(make_item(i, &i.to_string()));
    }

    let barrier = Arc::new(Barrier::new(2));
    let mut handles = vec![];

    for _ in 0..2 {
        let s = Arc::clone(&shard);
        let b = Arc::clone(&barrier);
        handles.push(thread::spawn(move || {
            let items = s.items();
            assert_eq!(items.len(), 10);
            b.wait();
        }));
    }

    for h in handles {
        h.join().unwrap();
    }
}

#[test]
fn test_auto_traits() {
    fn assert_send_sync<T: Send + Sync>() {}
    fn assert_sync<T: Sync>() {}
    assert_send_sync::<Shard<TestData>>();
    assert_send_sync::<ShardCollection<TestData>>();
    assert_sync::<ShardRef<'static, TestData>>();
    assert_sync::<ShardMutRef<'static, TestData>>();
}

#[test]
fn test_concurrent_push_indices_unique() {
    use std::collections::HashSet;
    use std::sync::Arc;
    use std::thread;

    let collection = Arc::new(ShardCollection::<u32>::new(4));
    let mut handles = vec![];

    for _ in 0..8 {
        let c = Arc::clone(&collection);
        handles.push(thread::spawn(move || {
            (0..125).filter_map(|i| c.push(i)).collect::<Vec<_>>()
        }));
    }

    let all: HashSet<(usize, usize)> = handles
        .into_iter()
        .flat_map(|h| h.join().unwrap())
        .collect();

    assert_eq!(all.len(), 1000);
    assert_eq!(collection.len(), 1000);
}

#[test]
fn test_mixed_readers_and_writers_stress() {
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;
    use std::thread;

    const WRITERS: usize = 4;
    const PER_WRITER: usize = 250;
    const READERS: usize = 4;
    const TOTAL: usize = WRITERS * PER_WRITER;

    let collection = Arc::new(ShardCollection::<u64>::new(8));
    let done = Arc::new(AtomicBool::new(false));
    let mut writer_handles = vec![];
    let mut reader_handles = vec![];

    for w in 0..WRITERS {
        let c = Arc::clone(&collection);
        writer_handles.push(thread::spawn(move || {
            for i in 0..PER_WRITER as u64 {
                c.push(w as u64 * PER_WRITER as u64 + i);
            }
        }));
    }

    for _ in 0..READERS {
        let c = Arc::clone(&collection);
        let done = Arc::clone(&done);
        reader_handles.push(thread::spawn(move || {
            let mut last = 0;
            while !done.load(Ordering::Acquire) {
                let mut now = 0;
                for s in 0..c.num_shards() {
                    now += c.get_shard(s).unwrap().len();
                }
                assert!(now >= last);
                last = now;
            }
            let mut now = 0;
            for s in 0..c.num_shards() {
                now += c.get_shard(s).unwrap().len();
            }
            assert!(now >= last);
            assert_eq!(now, TOTAL);
        }));
    }

    for h in writer_handles {
        h.join().unwrap();
    }
    done.store(true, Ordering::Release);
    for h in reader_handles {
        h.join().unwrap();
    }

    assert_eq!(collection.len(), TOTAL);
}

#[test]
fn test_shard_ref_deref() {
    let shard = TestShard::new();
    shard.push(make_item(7, "seven"));

    let r = shard.get_ref(0).unwrap();
    assert_eq!(r.id, 7);
    let value: &TestData = &r;
    assert_eq!(value.name, "seven");
}

#[test]
fn test_shard_mut_ref_deref_mut() {
    let shard = TestShard::new();
    shard.push(make_item(1, "a"));

    let mut m = shard.get_mut(0).unwrap();
    m.name = "b".to_string();
    m.id += 41;
    drop(m);

    let item = shard.get_cloned(0).unwrap();
    assert_eq!(item.id, 42);
    assert_eq!(item.name, "b");
}

#[test]
fn test_shard_with_capacity() {
    let shard = Shard::<u32>::with_capacity(10);
    assert!(shard.is_empty());
    for i in 0..20u32 {
        shard.push(i);
    }
    assert_eq!(shard.len(), 20);
    assert_eq!(shard.get_cloned(19), Some(19));
}

#[test]
fn test_shard_extend() {
    let shard = TestShard::new();
    shard.extend((0..5).map(|i| make_item(i, "x")));
    assert_eq!(shard.len(), 5);
    assert_eq!(shard.get_cloned(4).unwrap().id, 4);

    shard.extend(Vec::<TestData>::new());
    assert_eq!(shard.len(), 5);
}

#[test]
fn test_collection_with_capacity_and_extend() {
    let collection = ShardCollection::<u32>::with_capacity(4, 8);
    assert_eq!(collection.num_shards(), 4);
    assert!(collection.is_empty());

    collection.extend(0..100u32);
    assert_eq!(collection.len(), 100);

    let reference = ShardCollection::<u32>::new(4);
    for i in 0..100u32 {
        reference.push(i);
    }
    for s in 0..4 {
        assert_eq!(
            collection.get_shard(s).unwrap().items().as_slice(),
            reference.get_shard(s).unwrap().items().as_slice()
        );
    }
}

#[test]
fn test_collection_extend_edges() {
    let empty: ShardCollection<u32> = ShardCollection::new(0);
    empty.extend(0..10);
    assert_eq!(empty.num_shards(), 0);

    let collection = ShardCollection::<u32>::new(2);
    collection.extend(Vec::<u32>::new());
    assert!(collection.is_empty());

    collection.extend(0..3);
    assert_eq!(collection.len(), 3);
}
