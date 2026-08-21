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
fn test_get_mut_first_and_last() {
    let shard = TestShard::new();
    shard.push(make_item(0, "first"));
    shard.push(make_item(1, "last"));

    {
        let mut first = shard.get_mut(0).unwrap();
        first.get_mut_ref().name = "changed_first".to_string();
    }
    {
        let mut last = shard.get_mut(1).unwrap();
        last.get_mut_ref().name = "changed_last".to_string();
    }

    assert_eq!(shard.get_cloned(0).unwrap().name, "changed_first");
    assert_eq!(shard.get_cloned(1).unwrap().name, "changed_last");
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
