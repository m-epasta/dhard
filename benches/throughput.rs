//! 10M-op throughput benchmarks, grouped by operation: writes, reads, removals.
//! Run with `cargo bench --bench throughput`.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use dashmap::DashMap;
use dhard::{RwShardedDlhtMap, RwShardedHashMap, RwShardedSlotMap, ShardCollection};

const TOTAL_OPS: usize = 10_000_000;
const THREADS: usize = 8;

fn section(name: &str) {
    println!("\n=== {name} ===");
}

fn report(label: &str, elapsed: Duration) {
    println!(
        "{label}: {:.5}ms ({:.2} ops/sec)",
        elapsed.as_secs_f64() * 1000.0,
        TOTAL_OPS as f64 / elapsed.as_secs_f64()
    );
}

// ---------------------------------------------------------------------------
// Writes
// ---------------------------------------------------------------------------

fn bench_map_insert_single_threaded() {
    let map = RwShardedHashMap::<usize, usize>::with_capacity(16, TOTAL_OPS);
    let start = Instant::now();
    for k in 0..TOTAL_OPS {
        map.insert(k, k);
    }
    report("RwShardedHashMap[16] single-threaded insert", start.elapsed());
}

fn bench_map_insert_contended() {
    let chunk = TOTAL_OPS / THREADS;
    let map = Arc::new(RwShardedHashMap::<usize, usize>::with_capacity(
        16,
        TOTAL_OPS,
    ));
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
        &format!("RwShardedHashMap[16] {THREADS}-thread contended insert"),
        start.elapsed(),
    );
}

fn bench_dlht_insert_single_threaded() {
    let map = RwShardedDlhtMap::<usize, usize>::with_capacity(16, TOTAL_OPS);
    let start = Instant::now();
    for k in 0..TOTAL_OPS {
        map.insert(k, k);
    }
    report("RwShardedDlhtMap[16] single-threaded insert", start.elapsed());
}

fn bench_dlht_insert_contended() {
    let chunk = TOTAL_OPS / THREADS;
    let map = Arc::new(RwShardedDlhtMap::<usize, usize>::with_capacity(
        16,
        TOTAL_OPS,
    ));
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
        &format!("RwShardedDlhtMap[16] {THREADS}-thread contended insert"),
        start.elapsed(),
    );
}

fn bench_dashmap_insert_single_threaded() {
    let map: DashMap<usize, usize> =
        DashMap::with_capacity_and_shard_amount(TOTAL_OPS, 16);
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
    report("RwShardedSlotMap[16] single-threaded insert", start.elapsed());
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

fn bench_map_get_contended() {
    let chunk = TOTAL_OPS / THREADS;
    let map = Arc::new(RwShardedHashMap::<usize, usize>::with_capacity(
        16,
        TOTAL_OPS,
    ));
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
        &format!("RwShardedHashMap[16] {THREADS}-thread contended get"),
        start.elapsed(),
    );
}

fn bench_dlht_get_contended() {
    let chunk = TOTAL_OPS / THREADS;
    let map = Arc::new(RwShardedDlhtMap::<usize, usize>::with_capacity(
        16,
        TOTAL_OPS,
    ));
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
        &format!("RwShardedDlhtMap[16] {THREADS}-thread contended get"),
        start.elapsed(),
    );
}

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

// ---------------------------------------------------------------------------
// Removals
// ---------------------------------------------------------------------------

fn bench_map_remove_contended() {
    let chunk = TOTAL_OPS / THREADS;
    let map = Arc::new(RwShardedHashMap::<usize, usize>::with_capacity(
        16,
        TOTAL_OPS,
    ));
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
        &format!("RwShardedHashMap[16] {THREADS}-thread contended remove"),
        start.elapsed(),
    );
}

fn bench_dlht_remove_contended() {
    let chunk = TOTAL_OPS / THREADS;
    let map = Arc::new(RwShardedDlhtMap::<usize, usize>::with_capacity(
        16,
        TOTAL_OPS,
    ));
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
        &format!("RwShardedDlhtMap[16] {THREADS}-thread contended remove"),
        start.elapsed(),
    );
}

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

fn main() {
    println!("{TOTAL_OPS} ops per benchmark");

    section("writes");
    bench_map_insert_single_threaded();
    bench_map_insert_contended();
    bench_dlht_insert_single_threaded();
    bench_dlht_insert_contended();
    bench_dashmap_insert_single_threaded();
    bench_dashmap_insert_contended();
    bench_slotmap_insert_single_threaded();
    bench_slotmap_insert_contended();
    bench_collection_push_single_threaded();
    bench_collection_push_contended();
    bench_mutex_hashmap_insert_contended();

    section("reads");
    bench_map_get_contended();
    bench_dlht_get_contended();
    bench_dashmap_get_contended();
    bench_slotmap_get_contended();

    section("removals");
    bench_map_remove_contended();
    bench_dlht_remove_contended();
    bench_dashmap_remove_contended();
    bench_slotmap_remove_contended();
}
