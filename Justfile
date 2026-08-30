test:
    cargo test -- --no-capture

cl:
    cargo clippy -- -D warnings

# Displays benches in README
docgen:
    ./scripts/gen_benchmark_table.sh
