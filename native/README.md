# levelrl-native

The parts of LevelRL that were too slow in Python, written in Rust: Maia's board encoding and legal-move mask, a scanner for Lichess PGN dumps, and the tool that sorts sampled positions into rating bands. The Python package loads it through a C ABI and falls back to pure Python when it isn't built.

```sh
cd native
cargo build --release
```

From Rust it can be used as a path dependency, `levelrl-native = { path = "native" }`.

```rust
use levelrl_native::{maia_features, NativeBoard};

let fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
let encoded = maia_features(fen)?;
assert_eq!(encoded.tokens.len(), 768);

let mut board = NativeBoard::from_fen(fen)?;
board.push_uci("e2e4")?;
```

## Tools

`scan` reads a PGN dump from standard input and samples positions from rated games.

```sh
zstd -dc games.pgn.zst | target/release/scan \
  --month 2026-03 --keep-bp 100 --seed 0 --out data/scan/2026-03
```

`band` takes the scan output and writes one gzip file of positions per rating band.

```sh
target/release/band --root . --months "2025-09 2026-03" \
  --bands "1100 1300 1500" --n-cal 0 --out data/positions
```

`fix_moves` converts older position files to standard UCI. Each tool prints its options with `--help`.

## Tests

```sh
cargo test
cargo clippy --all-targets -- -D warnings
```

The Python tests that compare this crate with the Python implementation need the release build first.

```sh
cargo build --release
cd ..
python -m pytest tests/test_native.py tests/test_data.py
```

The board encoding is ported from [CSSLab/maia3](https://github.com/CSSLab/maia3), so this crate is AGPL-3.0.
