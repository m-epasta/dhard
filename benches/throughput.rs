//! 10M-op throughput benchmarks, grouped by operation: writes, reads, removals.
//! Run with `cargo bench --bench throughput`.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use dashmap::DashMap;
use dhard::{RwShardedSlotMap, ShardCollection, ShardedSlotMap, SlotKey};
use slotmap::{DefaultKey, SlotMap as RawSlotMap};

const TOTAL_OPS: usize = 10_000_000;
const THREADS: usize = 8;
const HEAVY_OPS: usize = 50_000_000;

fn section(name: &str) {
    println!("\n=== {name} ===");
}

fn report(label: &str, elapsed: Duration) {
    report_ops(label, elapsed, TOTAL_OPS);
}

fn report_ops(label: &str, elapsed: Duration, ops: usize) {
    println!(
        "{label}: {:.5}ms ({:.2} ops/sec)",
        elapsed.as_secs_f64() * 1000.0,
        ops as f64 / elapsed.as_secs_f64()
    );
}

// ---------------------------------------------------------------------------
// Writes
// ---------------------------------------------------------------------------

fn bench_dashmap_insert_single_threaded() {
    let map: DashMap<usize, usize> = DashMap::with_capacity_and_shard_amount(TOTAL_OPS, 16);
    let start = Instant::now();
    for k in 0..TOTAL_OPS {
        map.insert(k, k);
    }
    report("DashMap[16] single-threaded insert", start.elapsed());
}

fn bench_dashmap_insert_contended() {
    let chunk = TOTAL_OPS / THREADS;
    let map: Arc<DashMap<usize, usize>> =
        Arc::new(DashMap::with_capacity_and_shard_amount(TOTAL_OPS, 16));
    let start = Instant::now();
    let handles: Vec<_> = (0..THREADS)
        .map(|t| {
            let m = Arc::clone(&map);
            thread::spawn(move || {
                for i in 0..chunk {
                    let k = t * chunk + i;
                    m.insert(k, k);
                }
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }
    report(
        &format!("DashMap[16] {THREADS}-thread contended insert"),
        start.elapsed(),
    );
}

fn bench_slotmap_insert_single_threaded() {
    let map = RwShardedSlotMap::<usize>::with_capacity(16, TOTAL_OPS);
    let start = Instant::now();
    for v in 0..TOTAL_OPS {
        let _ = map.insert(v);
    }
    report(
        "RwShardedSlotMap[16] single-threaded insert",
        start.elapsed(),
    );
}

fn bench_slotmap_insert_contended() {
    let chunk = TOTAL_OPS / THREADS;
    let map = Arc::new(RwShardedSlotMap::<usize>::with_capacity(16, TOTAL_OPS));
    let start = Instant::now();
    let handles: Vec<_> = (0..THREADS)
        .map(|_| {
            let m = Arc::clone(&map);
            thread::spawn(move || {
                for i in 0..chunk {
                    let _ = m.insert(i);
                }
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }
    report(
        &format!("RwShardedSlotMap[16] {THREADS}-thread contended insert"),
        start.elapsed(),
    );
}

fn bench_std_slotmap_insert_single_threaded() {
    let mut map = RawSlotMap::<DefaultKey, usize>::with_capacity(TOTAL_OPS);
    let start = Instant::now();
    for v in 0..TOTAL_OPS {
        map.insert(v);
    }
    report("SlotMap[1] single-threaded insert", start.elapsed());
}

fn bench_std_slotmap_insert_contended() {
    let chunk = TOTAL_OPS / THREADS;
    let map: Arc<Mutex<RawSlotMap<DefaultKey, usize>>> =
        Arc::new(Mutex::new(RawSlotMap::with_capacity(TOTAL_OPS)));
    let start = Instant::now();
    let handles: Vec<_> = (0..THREADS)
        .map(|_| {
            let m = Arc::clone(&map);
            thread::spawn(move || {
                for i in 0..chunk {
                    m.lock().unwrap().insert(i);
                }
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }
    report(
        &format!("SlotMap[1] {THREADS}-thread contended insert"),
        start.elapsed(),
    );
}

fn bench_collection_push_single_threaded() {
    let collection = ShardCollection::<usize>::with_capacity(16, TOTAL_OPS);
    let start = Instant::now();
    for k in 0..TOTAL_OPS {
        let _ = collection.push(k);
    }
    report("ShardCollection[16] single-threaded push", start.elapsed());
}

fn bench_collection_push_contended() {
    let chunk = TOTAL_OPS / THREADS;
    let collection = Arc::new(ShardCollection::<usize>::with_capacity(16, TOTAL_OPS));
    let start = Instant::now();
    let handles: Vec<_> = (0..THREADS)
        .map(|_| {
            let c = Arc::clone(&collection);
            thread::spawn(move || {
                for i in 0..chunk {
                    let _ = c.push(i);
                }
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }
    report(
        &format!("ShardCollection[16] {THREADS}-thread contended push"),
        start.elapsed(),
    );
}

fn bench_mutex_hashmap_insert_contended() {
    let chunk = TOTAL_OPS / THREADS;
    let map = Arc::new(Mutex::new(HashMap::with_capacity(TOTAL_OPS)));
    let start = Instant::now();
    let handles: Vec<_> = (0..THREADS)
        .map(|t| {
            let m = Arc::clone(&map);
            thread::spawn(move || {
                for i in 0..chunk {
                    let k = t * chunk + i;
                    m.lock().unwrap().insert(k, k);
                }
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }
    report(
        &format!("Mutex<HashMap>[1] {THREADS}-thread contended insert"),
        start.elapsed(),
    );
}

// ---------------------------------------------------------------------------
// Reads
// ---------------------------------------------------------------------------

fn bench_dashmap_get_contended() {
    let chunk = TOTAL_OPS / THREADS;
    let map: Arc<DashMap<usize, usize>> =
        Arc::new(DashMap::with_capacity_and_shard_amount(TOTAL_OPS, 16));
    (0..TOTAL_OPS).for_each(|k| {
        map.insert(k, k);
    });

    let start = Instant::now();
    let handles: Vec<_> = (0..THREADS)
        .map(|t| {
            let m = Arc::clone(&map);
            thread::spawn(move || {
                let mut hits = 0usize;
                for i in 0..chunk {
                    if m.contains_key(&(i * (t + 1))) {
                        hits += 1;
                    }
                }
                hits
            })
        })
        .collect();
    for h in handles {
        assert!(h.join().unwrap() > 0);
    }
    report(
        &format!("DashMap[16] {THREADS}-thread contended get"),
        start.elapsed(),
    );
}

fn bench_slotmap_get_contended() {
    let chunk = TOTAL_OPS / THREADS;
    let map = Arc::new(RwShardedSlotMap::<usize>::with_capacity(16, TOTAL_OPS));
    let keys: Vec<_> = (0..TOTAL_OPS).map(|v| map.insert(v)).collect();
    let keys = Arc::new(keys);

    let start = Instant::now();
    let handles: Vec<_> = (0..THREADS)
        .map(|t| {
            let m = Arc::clone(&map);
            let keys = Arc::clone(&keys);
            thread::spawn(move || {
                let mut hits = 0usize;
                for i in 0..chunk {
                    if m.contains(keys[i * (t + 1) % TOTAL_OPS]) {
                        hits += 1;
                    }
                }
                hits
            })
        })
        .collect();
    for h in handles {
        assert!(h.join().unwrap() > 0);
    }
    report(
        &format!("RwShardedSlotMap[16] {THREADS}-thread contended get"),
        start.elapsed(),
    );
}

fn bench_std_slotmap_get_single_threaded() {
    let mut map = RawSlotMap::<DefaultKey, usize>::with_capacity(TOTAL_OPS);
    let keys: Vec<_> = (0..TOTAL_OPS).map(|v| map.insert(v)).collect();

    let start = Instant::now();
    let hits = keys.iter().filter(|k| map.contains_key(**k)).count();
    assert_eq!(hits, TOTAL_OPS);
    report("SlotMap[1] single-threaded get", start.elapsed());
}

fn bench_std_slotmap_get_contended() {
    let chunk = TOTAL_OPS / THREADS;
    let mut map = RawSlotMap::<DefaultKey, usize>::with_capacity(TOTAL_OPS);
    let keys: Vec<_> = (0..TOTAL_OPS).map(|v| map.insert(v)).collect();
    let map = Arc::new(Mutex::new(map));
    let keys = Arc::new(keys);

    let start = Instant::now();
    let handles: Vec<_> = (0..THREADS)
        .map(|t| {
            let m = Arc::clone(&map);
            let keys = Arc::clone(&keys);
            thread::spawn(move || {
                let mut hits = 0usize;
                for i in 0..chunk {
                    if m.lock().unwrap().contains_key(keys[t * chunk + i]) {
                        hits += 1;
                    }
                }
                hits
            })
        })
        .collect();
    for h in handles {
        assert!(h.join().unwrap() > 0);
    }
    report(
        &format!("SlotMap[1] {THREADS}-thread contended get"),
        start.elapsed(),
    );
}

// ---------------------------------------------------------------------------
// Removals
// ---------------------------------------------------------------------------

fn bench_dashmap_remove_contended() {
    let chunk = TOTAL_OPS / THREADS;
    let map: Arc<DashMap<usize, usize>> =
        Arc::new(DashMap::with_capacity_and_shard_amount(TOTAL_OPS, 16));
    (0..TOTAL_OPS).for_each(|k| {
        map.insert(k, k);
    });

    let start = Instant::now();
    let handles: Vec<_> = (0..THREADS)
        .map(|t| {
            let m = Arc::clone(&map);
            thread::spawn(move || {
                for i in 0..chunk {
                    assert!(m.remove(&(t * chunk + i)).is_some());
                }
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }
    report(
        &format!("DashMap[16] {THREADS}-thread contended remove"),
        start.elapsed(),
    );
}

fn bench_slotmap_remove_contended() {
    let chunk = TOTAL_OPS / THREADS;
    let map = Arc::new(RwShardedSlotMap::<usize>::with_capacity(16, TOTAL_OPS));
    let keys: Vec<_> = (0..TOTAL_OPS).map(|v| map.insert(v)).collect();
    let keys = Arc::new(keys);

    let start = Instant::now();
    let handles: Vec<_> = (0..THREADS)
        .map(|t| {
            let m = Arc::clone(&map);
            let keys = Arc::clone(&keys);
            thread::spawn(move || {
                for i in 0..chunk {
                    assert!(m.remove(keys[t * chunk + i]).is_some());
                }
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }
    report(
        &format!("RwShardedSlotMap[16] {THREADS}-thread contended remove"),
        start.elapsed(),
    );
}

fn bench_std_slotmap_remove_single_threaded() {
    let mut map = RawSlotMap::<DefaultKey, usize>::with_capacity(TOTAL_OPS);
    let keys: Vec<_> = (0..TOTAL_OPS).map(|v| map.insert(v)).collect();

    let start = Instant::now();
    for k in keys {
        assert!(map.remove(k).is_some());
    }
    report("SlotMap[1] single-threaded remove", start.elapsed());
}

fn bench_std_slotmap_remove_contended() {
    let chunk = TOTAL_OPS / THREADS;
    let mut map = RawSlotMap::<DefaultKey, usize>::with_capacity(TOTAL_OPS);
    let keys: Vec<_> = (0..TOTAL_OPS).map(|v| map.insert(v)).collect();
    let map = Arc::new(Mutex::new(map));
    let keys = Arc::new(keys);

    let start = Instant::now();
    let handles: Vec<_> = (0..THREADS)
        .map(|t| {
            let m = Arc::clone(&map);
            let keys = Arc::clone(&keys);
            thread::spawn(move || {
                for i in 0..chunk {
                    assert!(m.lock().unwrap().remove(keys[t * chunk + i]).is_some());
                }
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }
    report(
        &format!("SlotMap[1] {THREADS}-thread contended remove"),
        start.elapsed(),
    );
}

// ---------------------------------------------------------------------------
// Churn

/// Sliding-window churn: each thread keeps a fixed-size ring of live keys and every
/// iteration inserts one value and removes the oldest. Exercises free-list recycling
/// under sustained mixed traffic; ops/sec counts inserts plus removals.
const CHURN_WINDOW: usize = 1024;

fn bench_slotmap_churn_contended() {
    let chunk = TOTAL_OPS / THREADS;
    let map = Arc::new(RwShardedSlotMap::<u64>::with_capacity(16, TOTAL_OPS));
    let start = Instant::now();
    let handles: Vec<_> = (0..THREADS)
        .map(|t| {
            let m = Arc::clone(&map);
            thread::spawn(move || {
                let mut ring: Vec<Option<SlotKey>> = vec![None; CHURN_WINDOW];
                let mut ops = 0usize;
                for i in 0..chunk {
                    let slot = i % CHURN_WINDOW;
                    if let Some(old) = ring[slot].take() {
                        assert!(m.remove(old).is_some());
                        ops += 1;
                    }
                    ring[slot] = Some(m.insert((t as u64) << 40 | i as u64));
                    ops += 1;
                }
                for key in ring.into_iter().flatten() {
                    assert!(m.remove(key).is_some());
                }
                ops
            })
        })
        .collect();
    let ops: usize = handles.into_iter().map(|h| h.join().unwrap()).sum();
    assert!(map.is_empty());
    report_ops(
        &format!("RwShardedSlotMap[16] {THREADS}-thread churn"),
        start.elapsed(),
        ops,
    );
}

fn bench_heavy_std_slotmap_insert_single_threaded() {
    let mut map = RawSlotMap::<DefaultKey, usize>::with_capacity(HEAVY_OPS);
    let start = Instant::now();
    for v in 0..HEAVY_OPS {
        map.insert(v);
    }
    report_ops(
        "SlotMap[1] single-threaded insert",
        start.elapsed(),
        HEAVY_OPS,
    );
}

fn bench_heavy_rwsharded_slotmap_insert_single_threaded() {
    let map = RwShardedSlotMap::<usize>::with_capacity(16, HEAVY_OPS);
    let start = Instant::now();
    for v in 0..HEAVY_OPS {
        map.insert(v);
    }
    report_ops(
        "RwShardedSlotMap[16] single-threaded insert",
        start.elapsed(),
        HEAVY_OPS,
    );
}

fn bench_heavy_std_slotmap_get_single_threaded() {
    let mut map = RawSlotMap::<DefaultKey, usize>::with_capacity(HEAVY_OPS);
    let keys: Vec<_> = (0..HEAVY_OPS).map(|v| map.insert(v)).collect();

    let start = Instant::now();
    let hits = keys.iter().filter(|k| map.contains_key(**k)).count();
    assert_eq!(hits, HEAVY_OPS);
    report_ops("SlotMap[1] single-threaded get", start.elapsed(), HEAVY_OPS);
}

fn bench_heavy_rwsharded_slotmap_get_single_threaded() {
    let map = RwShardedSlotMap::<usize>::with_capacity(16, HEAVY_OPS);
    let keys: Vec<_> = (0..HEAVY_OPS).map(|v| map.insert(v)).collect();

    let start = Instant::now();
    let hits = keys.iter().filter(|k| map.contains(**k)).count();
    assert_eq!(hits, HEAVY_OPS);
    report_ops(
        "RwShardedSlotMap[16] single-threaded get",
        start.elapsed(),
        HEAVY_OPS,
    );
}

fn bench_heavy_std_slotmap_remove_single_threaded() {
    let mut map = RawSlotMap::<DefaultKey, usize>::with_capacity(HEAVY_OPS);
    let keys: Vec<_> = (0..HEAVY_OPS).map(|v| map.insert(v)).collect();

    let start = Instant::now();
    for k in keys {
        assert!(map.remove(k).is_some());
    }
    report_ops(
        "SlotMap[1] single-threaded remove",
        start.elapsed(),
        HEAVY_OPS,
    );
}

fn bench_heavy_rwsharded_slotmap_remove_single_threaded() {
    let map = RwShardedSlotMap::<usize>::with_capacity(16, HEAVY_OPS);
    let keys: Vec<_> = (0..HEAVY_OPS).map(|v| map.insert(v)).collect();

    let start = Instant::now();
    for k in keys {
        assert!(map.remove(k).is_some());
    }
    report_ops(
        "RwShardedSlotMap[16] single-threaded remove",
        start.elapsed(),
        HEAVY_OPS,
    );
}

fn bench_sharded_slotmap_insert_single_threaded(shards: usize) {
    let mut map = ShardedSlotMap::<usize>::with_capacity(shards, TOTAL_OPS);
    let start = Instant::now();
    for v in 0..TOTAL_OPS {
        let _ = map.insert(v);
    }
    report(
        &format!("ShardedSlotMap[{shards}] single-threaded insert"),
        start.elapsed(),
    );
}

fn bench_sharded_slotmap_get_single_threaded(shards: usize) {
    let mut map = ShardedSlotMap::<usize>::with_capacity(shards, TOTAL_OPS);
    let keys: Vec<_> = (0..TOTAL_OPS).map(|v| map.insert(v)).collect();

    let start = Instant::now();
    let hits = keys.iter().filter(|k| map.get(**k).is_some()).count();
    assert_eq!(hits, TOTAL_OPS);
    report(
        &format!("ShardedSlotMap[{shards}] single-threaded get"),
        start.elapsed(),
    );
}

fn bench_sharded_slotmap_remove_single_threaded(shards: usize) {
    let mut map = ShardedSlotMap::<usize>::with_capacity(shards, TOTAL_OPS);
    let keys: Vec<_> = (0..TOTAL_OPS).map(|v| map.insert(v)).collect();

    let start = Instant::now();
    for k in keys {
        assert!(map.remove(k).is_some());
    }
    report(
        &format!("ShardedSlotMap[{shards}] single-threaded remove"),
        start.elapsed(),
    );
}

fn bench_heavy_sharded_slotmap_insert_single_threaded(shards: usize) {
    let mut map = ShardedSlotMap::<usize>::with_capacity(shards, HEAVY_OPS);
    let start = Instant::now();
    for v in 0..HEAVY_OPS {
        let _ = map.insert(v);
    }
    report_ops(
        &format!("ShardedSlotMap[{shards}] single-threaded insert"),
        start.elapsed(),
        HEAVY_OPS,
    );
}

fn bench_heavy_sharded_slotmap_get_single_threaded(shards: usize) {
    let mut map = ShardedSlotMap::<usize>::with_capacity(shards, HEAVY_OPS);
    let keys: Vec<_> = (0..HEAVY_OPS).map(|v| map.insert(v)).collect();

    let start = Instant::now();
    let hits = keys.iter().filter(|k| map.get(**k).is_some()).count();
    assert_eq!(hits, HEAVY_OPS);
    report_ops(
        &format!("ShardedSlotMap[{shards}] single-threaded get"),
        start.elapsed(),
        HEAVY_OPS,
    );
}

fn bench_heavy_sharded_slotmap_remove_single_threaded(shards: usize) {
    let mut map = ShardedSlotMap::<usize>::with_capacity(shards, HEAVY_OPS);
    let keys: Vec<_> = (0..HEAVY_OPS).map(|v| map.insert(v)).collect();

    let start = Instant::now();
    for k in keys {
        assert!(map.remove(k).is_some());
    }
    report_ops(
        &format!("ShardedSlotMap[{shards}] single-threaded remove"),
        start.elapsed(),
        HEAVY_OPS,
    );
}

fn main() {
    println!("{TOTAL_OPS} ops per benchmark");

    section("writes");
    bench_dashmap_insert_single_threaded();
    bench_dashmap_insert_contended();
    bench_slotmap_insert_single_threaded();
    bench_slotmap_insert_contended();
    bench_std_slotmap_insert_single_threaded();
    bench_std_slotmap_insert_contended();
    bench_sharded_slotmap_insert_single_threaded(1);
    bench_sharded_slotmap_insert_single_threaded(16);
    bench_collection_push_single_threaded();
    bench_collection_push_contended();
    bench_mutex_hashmap_insert_contended();

    section("reads");
    bench_dashmap_get_contended();
    bench_slotmap_get_contended();
    bench_std_slotmap_get_single_threaded();
    bench_std_slotmap_get_contended();
    bench_sharded_slotmap_get_single_threaded(1);
    bench_sharded_slotmap_get_single_threaded(16);

    section("removals");
    bench_dashmap_remove_contended();
    bench_slotmap_remove_contended();
    bench_std_slotmap_remove_single_threaded();
    bench_std_slotmap_remove_contended();
    bench_sharded_slotmap_remove_single_threaded(1);
    bench_sharded_slotmap_remove_single_threaded(16);

    section("churn");
    bench_slotmap_churn_contended();

    section("heavy writes (50M)");
    bench_heavy_std_slotmap_insert_single_threaded();
    bench_heavy_rwsharded_slotmap_insert_single_threaded();
    bench_heavy_sharded_slotmap_insert_single_threaded(1);
    bench_heavy_sharded_slotmap_insert_single_threaded(16);

    section("heavy reads (50M)");
    bench_heavy_std_slotmap_get_single_threaded();
    bench_heavy_rwsharded_slotmap_get_single_threaded();
    bench_heavy_sharded_slotmap_get_single_threaded(1);
    bench_heavy_sharded_slotmap_get_single_threaded(16);

    section("heavy removals (50M)");
    bench_heavy_std_slotmap_remove_single_threaded();
    bench_heavy_rwsharded_slotmap_remove_single_threaded();
    bench_heavy_sharded_slotmap_remove_single_threaded(1);
    bench_heavy_sharded_slotmap_remove_single_threaded(16);
}
