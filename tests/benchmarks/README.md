# Performance benchmarks

These benchmarks are **opt-in** — the default `pytest` invocation skips
them (configured via `--ignore=tests/benchmarks` in `pyproject.toml`).
They exist to:

1. Establish a baseline for backtest throughput so future PRs can detect
   regressions or measure optimizations.
2. Surface the slowest paths (currently: `self.data` slicing inside
   `Strategy.on_bar_close`).

## Running

Install dev extras (`pip install -e ".[dev]"`) — this includes
`pytest-benchmark` — and then:

```bash
pytest tests/benchmarks/ \
  --benchmark-columns=mean,min,stddev,ops \
  --benchmark-sort=mean
```

To save a run for comparison:

```bash
pytest tests/benchmarks/ --benchmark-autosave
# later, after a change:
pytest tests/benchmarks/ --benchmark-compare
```

## Reference baseline

Recorded on Apple M-class hardware, qtrade 0.4.0:

| Benchmark                                  | Mean (ms) | Throughput      |
|--------------------------------------------|-----------|-----------------|
| `process_bar` × 1k bars, no orders         | ~35       | ~28k bars/s     |
| `process_bar` × 10k bars, no orders        | ~380      | ~26k bars/s     |
| `process_bar` × 1k bars, one open trade    | ~36       | ~28k bars/s     |
| `process_bar` × 10k bars, one open trade   | ~380      | ~26k bars/s     |
| Backtest (SMA crossover) × 1k bars         | ~135      | ~7k bars/s      |
| Backtest (SMA crossover) × 10k bars        | ~1400     | ~7k bars/s      |
| Backtest 4-asset portfolio × 5k bars       | ~3500     | ~1.4k bars/s/asset |
| `place_orders` × 100 (alternating buy/sell)| ~3.6      | ~28k orders/s   |

### Reading the numbers

- **Pure broker overhead** is ~35 µs per bar with zero or one trade and
  scales linearly with bar count.
- **Strategy logic** (the SMA crossover above) raises per-bar time to
  ~135 µs. The dominant cost is the `Strategy.data` property, which
  rebuilds a sliced DataFrame on every access.
- **Multi-asset overhead** is roughly proportional to the number of
  assets — there are no obvious scaling pathologies, just N× more bar
  work.

### Where to optimize first if it ever matters

1. Cache `Strategy.data` per bar instead of re-slicing on every property
   access.
2. Replace the per-bar `df.loc[ts, 'Close']` lookups in
   `Broker.{available_margin, unrealized_pnl, ...}` with positional
   indexing into a NumPy array.

These haven't been done because nothing currently demands them — most
realistic backtests (5–10 years daily ≈ 1k–2k bars) run in seconds, and
the optimization would add complexity for little user-visible benefit.
