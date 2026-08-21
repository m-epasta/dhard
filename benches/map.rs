use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::thread;

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use dhard::RwShardedHashMap;

const THREADS: usize = 8;
const ITEMS_PER_THREAD: usize = 20_000;
const LOOKUPS_PER_THREAD: usize = 200_000;

fn bench_map_contended_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("map_contended_insert");
    for shards in [4usize, 16] {
        group.bench_function(format!("rw_sharded_map_shards_{shards}"), |b| {
            b.iter(|| {
                let map = Arc::new(RwShardedHashMap::<usize, usize>::with_capacity(
                    shards,
                    THREADS * ITEMS_PER_THREAD,
                ));
                let handles: Vec<_> = (0..THREADS)
                    .map(|t| {
                        let m = Arc::clone(&map);
                        thread::spawn(move || {
                            for i in 0..ITEMS_PER_THREAD {
                                let k = t * ITEMS_PER_THREAD + i;
                                m.insert(k, k);
                            }
                        })
                    })
                    .collect();
                for h in handles {
                    h.join().unwrap();
                }
            });
        });
    }

    group.bench_function("mutex_hashmap", |b| {
        b.iter(|| {
            let map = Arc::new(Mutex::new(HashMap::with_capacity(
                THREADS * ITEMS_PER_THREAD,
            )));
            let handles: Vec<_> = (0..THREADS)
                .map(|t| {
                    let m = Arc::clone(&map);
                    thread::spawn(move || {
                        for i in 0..ITEMS_PER_THREAD {
                            let k = t * ITEMS_PER_THREAD + i;
                            m.lock().unwrap().insert(k, k);
                        }
                    })
                })
                .collect();
            for h in handles {
                h.join().unwrap();
            }
        });
    });
    group.finish();
}

fn bench_map_contended_get(c: &mut Criterion) {
    let mut group = c.benchmark_group("map_contended_get");
    let total_keys = THREADS * ITEMS_PER_THREAD;

    for shards in [4usize, 16] {
        group.bench_function(format!("rw_sharded_map_shards_{shards}"), |b| {
            b.iter_batched(
                || {
                    let map = Arc::new(RwShardedHashMap::<usize, usize>::with_capacity(
                        shards,
                        total_keys,
                    ));
                    (0..total_keys).for_each(|k| {
                        map.insert(k, k);
                    });
                    map
                },
                |map| {
                    let handles: Vec<_> = (0..THREADS)
                        .map(|t| {
                            let m = Arc::clone(&map);
                            thread::spawn(move || {
                                let step = t + 1;
                                let mut hits = 0usize;
                                for i in 0..LOOKUPS_PER_THREAD {
                                    if m.get_cloned(&((i * step) % total_keys)).is_some() {
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
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.bench_function("mutex_hashmap", |b| {
        b.iter_batched(
            || {
                let map =
                    Arc::new(Mutex::new(HashMap::with_capacity(total_keys)));
                (0..total_keys).for_each(|k| {
                    map.lock().unwrap().insert(k, k);
                });
                map
            },
            |map| {
                let handles: Vec<_> = (0..THREADS)
                    .map(|t| {
                        let m = Arc::clone(&map);
                        thread::spawn(move || {
                            let step = t + 1;
                            let mut hits = 0usize;
                            for i in 0..LOOKUPS_PER_THREAD {
                                if m.lock().unwrap().contains_key(&((i * step) % total_keys)) {
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
            },
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_map_contended_insert,
    bench_map_contended_get
);
criterion_main!(benches);
