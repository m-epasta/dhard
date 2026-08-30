#!/usr/bin/env bash
# Regenerates the comparison table under the "## Benchmarks" heading of README.md
# from `cargo bench --bench throughput`.
#
# Each structure is reported at the thread-count where it shines:
#   - slotmap::SlotMap     single-threaded   (bare arena)
#   - ShardedSlotMap       single-threaded   (sync-less)
#   - RwShardedSlotMap     8-thread contended (locked, concurrent)
#
# Uses the first (10M-op) occurrence of each label.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readme="$repo_root/README.md"
bench_out="$(mktemp)"

# Label -> (display name, mode). First occurrence is the 10M run.
declare -a rows=(
    "slotmap::SlotMap|SlotMap[1] single-threaded insert|SlotMap[1] single-threaded get|SlotMap[1] single-threaded remove|single-threaded"
    "ShardedSlotMap|ShardedSlotMap[16] single-threaded insert|ShardedSlotMap[16] single-threaded get|ShardedSlotMap[16] single-threaded remove|single-threaded"
    "RwShardedSlotMap|RwShardedSlotMap[16] 8-thread contended insert|RwShardedSlotMap[16] 8-thread contended get|RwShardedSlotMap[16] 8-thread contended remove|8-thread contended"
)

fetch_ops() {
    local label="$1"
    local line
    # First matching line (head -n 1) => the 10M-op section.
    line="$(grep -F -m1 -- "$label: " "$bench_out")" || {
        echo "error: benchmark label not found: $label" >&2
        rm -f "$bench_out"
        exit 1
    }
    sed -E 's/.*\(([0-9]+\.[0-9]+) ops\/sec\).*/\1/' <<<"$line"
}

cleanup() {
    rm -f "$bench_out"
}
trap cleanup EXIT

echo "running 'cargo bench --bench throughput' (this may take a while)..."
cargo bench --bench throughput --manifest-path "$repo_root/Cargo.toml" 2>&1 >"$bench_out"

# Render the table body (one line per row).
body_lines=()
while IFS='|' read -r name wlabel rlabel dlabel mode; do
    w="$(fetch_ops "$wlabel")"
    r="$(fetch_ops "$rlabel")"
    d="$(fetch_ops "$dlabel")"
    body_lines+=("| \`$name\` | $mode | $w | $r | $d |")
done < <(printf '%s\n' "${rows[@]}")

# Content placed between "## Benchmarks" and the next "## " heading. It does
# NOT include the heading itself (that is kept from the README).
table="Single-threaded versus 8-thread contested throughput (10,000,000 ops per benchmark),
each structure measured where it shines: the bare \`slotmap::SlotMap\` and the
sync-less \`ShardedSlotMap\` single-threaded, the locked concurrent \`RwShardedSlotMap\`
across 8 contending threads. Values are ops/sec (higher is better).

| Structure | Mode | Writes (ops/s) | Reads (ops/s) | Removals (ops/s) |
|---|---|---|---|---|
$(printf '%s\n' "${body_lines[@]}")
"

awk -v table="$table" '
  /^## Benchmarks$/ { in_bench = 1; print; next }
  /^## / && !/^## Benchmarks$/ {
    if (in_bench) { printf "\n"; print table; in_bench = 0 }
    print
    next
  }
  { if (!in_bench) print }
' "$readme" >"$readme.tmp"

mv "$readme.tmp" "$readme"

echo "README.md updated."
