use std::sync::Arc;
use std::thread;

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use dhard::{Shard, ShardCollection};

const ITEMS_PER_THREAD: u64 = 10_000;

fn bench_collection_contended_push(c: &mut Criterion) {
    let mut group = c.benchmark_group("collection_contended_push");
    for shards in [1usize, 4, 16] {
        for threads in [1usize, 2, 8] {
            group.bench_function(format!("shards_{shards}_threads_{threads}"), |b| {
                b.iter(|| {
                    let collection = Arc::new(ShardCollection::<u64>::new(shards));
                    let handles: Vec<_> = (0..threads)
                        .map(|_| {
                            let c = Arc::clone(&collection);
                            thread::spawn(move || {
                                for i in 0..ITEMS_PER_THREAD {
                                    c.push(i);
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
    }
    group.finish();
}

fn bench_shard_contended_push(c: &mut Criterion) {
    let mut group = c.benchmark_group("shard_contended_push");
    for threads in [1usize, 2, 8] {
        group.bench_function(format!("threads_{threads}"), |b| {
            b.iter(|| {
                let shard = Arc::new(Shard::<u64>::new());
                let handles: Vec<_> = (0..threads)
                    .map(|_| {
                        let s = Arc::clone(&shard);
                        thread::spawn(move || {
                            for i in 0..ITEMS_PER_THREAD {
                                s.push(i);
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
    group.finish();
}

fn bench_collection_bulk_load(c: &mut Criterion) {
    let mut group = c.benchmark_group("collection_bulk_load");
    for shards in [4usize, 16] {
        group.bench_function(format!("extend_shards_{shards}"), |b| {
            b.iter_batched(
                || ShardCollection::<u64>::with_capacity(shards, 100_000 / shards),
                |collection| collection.extend(0..100_000u64),
                BatchSize::SmallInput,
            );
        });
        group.bench_function(format!("push_loop_shards_{shards}"), |b| {
            b.iter_batched(
                || ShardCollection::<u64>::with_capacity(shards, 100_000 / shards),
                |collection| {
                    for i in 0..100_000u64 {
                        collection.push(i);
                    }
                },
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_collection_contended_push,
    bench_shard_contended_push,
    bench_collection_bulk_load,
);
criterion_main!(benches);
