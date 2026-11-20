use crate::model::PositionRow;
use crate::rng::Rng;
use crate::{Error, Result};
use pgn_reader::{BufferedReader, RawComment, RawHeader, SanPlus, Skip, Visitor};
use shakmaty::{CastlingMode, Chess, Position};
use std::collections::HashMap;
use std::fs::{create_dir_all, File};
use std::io::{BufWriter, Read, Write};
use std::path::PathBuf;
use std::time::Instant;

/// Sampling and output settings for [`scan_pgn`].
#[derive(Debug, Clone)]
pub struct ScanOptions {
    pub month: String,
    pub keep_bp: u32,
    pub seed: u64,
    pub out: PathBuf,
}

impl Default for ScanOptions {
    fn default() -> Self {
        Self {
            month: "unknown".to_string(),
            keep_bp: 100,
            seed: 0,
            out: PathBuf::from("data/scan/unknown"),
        }
    }
}

/// Games counted, rows written, and per-player human-game counts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScanSummary {
    pub games: usize,
    pub rows: usize,
    pub counts: HashMap<String, usize>,
}

struct Scan {
    month: String,
    rng: Rng,
    keep_bp: u32,
    seen: usize,
    started: Instant,
    rows: usize,
    counts: HashMap<String, usize>,
    site: String,
    event: String,
    white: String,
    black: String,
    white_elo: Option<i32>,
    black_elo: Option<i32>,
    white_title: String,
    black_title: String,
    tc: String,
    board: Chess,
    moves: Vec<String>,
    clocks: Vec<Option<f64>>,
    sampled: bool,
}

impl Scan {
    fn new(month: String, keep_bp: u32, seed: u64) -> Self {
        Self {
            month,
            rng: Rng(seed),
            keep_bp,
            seen: 0,
            started: Instant::now(),
            rows: 0,
            counts: HashMap::new(),
            site: String::new(),
            event: String::new(),
            white: String::new(),
            black: String::new(),
            white_elo: None,
            black_elo: None,
            white_title: String::new(),
            black_title: String::new(),
            tc: String::new(),
            board: Chess::default(),
            moves: Vec::new(),
            clocks: Vec::new(),
            sampled: false,
        }
    }

    fn human(&self) -> bool {
        self.event.contains("Rated") && self.white_title != "BOT" && self.black_title != "BOT"
    }
}

impl Visitor for Scan {
    type Result = Option<PositionRow>;

    fn begin_game(&mut self) {
        self.seen += 1;
        if self.seen.is_multiple_of(2_000_000) {
            let rate = self.seen as f64 / self.started.elapsed().as_secs_f64().max(0.001);
            eprintln!(
                "{}: {}M games, {} rows, {:.0} games/s, rust",
                self.month,
                self.seen / 1_000_000,
                self.rows,
                rate
            );
        }
        self.site.clear();
        self.event.clear();
        self.white.clear();
        self.black.clear();
        self.white_elo = None;
        self.black_elo = None;
        self.white_title.clear();
        self.black_title.clear();
        self.tc.clear();
        self.board = Chess::default();
        self.moves.clear();
        self.clocks.clear();
        self.sampled = false;
    }

    fn header(&mut self, name: &[u8], value: RawHeader<'_>) {
        let value = String::from_utf8_lossy(value.as_bytes()).into_owned();
        match name {
            b"Site" => self.site = value,
            b"Event" => self.event = value,
            b"White" => self.white = value,
            b"Black" => self.black = value,
            b"WhiteElo" => self.white_elo = value.parse().ok(),
            b"BlackElo" => self.black_elo = value.parse().ok(),
            b"WhiteTitle" => self.white_title = value,
            b"BlackTitle" => self.black_title = value,
            b"TimeControl" => self.tc = value,
            _ => {}
        }
    }

    fn end_headers(&mut self) -> Skip {
        if self.human() {
            *self.counts.entry(self.white.clone()).or_default() += 1;
            *self.counts.entry(self.black.clone()).or_default() += 1;
        }
        self.sampled = self.human()
            && self.white_elo.is_some()
            && self.black_elo.is_some()
            && self.rng.unit() < self.keep_bp as f64 / 10_000.0;
        Skip(!self.sampled)
    }

    fn san(&mut self, san_plus: SanPlus) {
        if !self.sampled {
            return;
        }
        if let Ok(m) = san_plus.san.to_move(&self.board) {
            self.moves
                .push(m.to_uci(CastlingMode::Standard).to_string());
            self.clocks.push(None);
            self.board.play_unchecked(&m);
        } else {
            self.sampled = false;
        }
    }

    fn comment(&mut self, comment: RawComment<'_>) {
        let text = String::from_utf8_lossy(comment.as_bytes());
        if let Some(start) = text.find("[%clk ") {
            let rest = &text[start + 6..];
            if let Some(end) = rest.find(']') {
                let parts: Vec<_> = rest[..end].split(':').collect();
                if parts.len() == 3 {
                    if let (Ok(h), Ok(m), Ok(s)) = (
                        parts[0].parse::<f64>(),
                        parts[1].parse::<f64>(),
                        parts[2].parse::<f64>(),
                    ) {
                        if let Some(clock) = self.clocks.last_mut() {
                            *clock = Some(h * 3600.0 + m * 60.0 + s);
                        }
                    }
                }
            }
        }
    }

    fn begin_variation(&mut self) -> Skip {
        Skip(true)
    }

    fn end_game(&mut self) -> Self::Result {
        if !self.sampled {
            return None;
        }
        let mut usable = Vec::new();
        for i in 0..self.moves.len() {
            if i >= 10 {
                if self
                    .clocks
                    .get(i)
                    .and_then(|clock| *clock)
                    .is_some_and(|clock| clock < 30.0)
                {
                    break;
                }
                usable.push(i);
            }
        }
        if usable.is_empty() {
            return None;
        }
        let i = usable[(self.rng.next() as usize) % usable.len()];
        let white_to_move = i % 2 == 0;
        let (elo, oppo_elo) = if white_to_move {
            (self.white_elo, self.black_elo)
        } else {
            (self.black_elo, self.white_elo)
        };
        let (Some(elo), Some(oppo_elo)) = (elo, oppo_elo) else {
            return None;
        };
        self.rows += 1;
        Some(PositionRow {
            site: self.site.clone(),
            moves: self.moves[..i].to_vec(),
            move_uci: self.moves[i].clone(),
            elo,
            oppo_elo,
            white: self.white.clone(),
            black: self.black.clone(),
            tc: self.tc.clone(),
            month: self.month.clone(),
        })
    }
}

/// Streams PGN games into `rows.jsonl`, `counts.json`, and `done` under `options.out`.
pub fn scan_pgn<R: Read>(input: R, options: ScanOptions) -> Result<ScanSummary> {
    if options.keep_bp > 10_000 {
        return Err(Error::InvalidArgument(format!(
            "keep-bp must be between 0 and 10000, got {}",
            options.keep_bp
        )));
    }
    create_dir_all(&options.out)?;
    let mut scanner = Scan::new(options.month, options.keep_bp, options.seed);
    let mut reader = BufferedReader::new(input);
    let mut rows = BufWriter::new(File::create(options.out.join("rows.jsonl"))?);
    while let Some(row) = reader.read_game(&mut scanner)? {
        if let Some(row) = row {
            serde_json::to_writer(&mut rows, &row)?;
            rows.write_all(b"\n")?;
        }
    }
    rows.flush()?;
    let mut counts = BufWriter::new(File::create(options.out.join("counts.json"))?);
    serde_json::to_writer(&mut counts, &scanner.counts)?;
    counts.flush()?;
    let total = scanner
        .counts
        .values()
        .try_fold(0usize, |total, count| total.checked_add(*count))
        .ok_or_else(|| Error::InvalidData("player count overflow".to_string()))?;
    let games = total / 2;
    let mut done = BufWriter::new(File::create(options.out.join("done"))?);
    serde_json::to_writer(
        &mut done,
        &serde_json::json!({
            "games": games,
            "rows": scanner.rows,
            "keep": scanner.keep_bp as f64 / 10_000.0,
            "engine": "rust",
        }),
    )?;
    done.flush()?;
    Ok(ScanSummary {
        games,
        rows: scanner.rows,
        counts: scanner.counts,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn empty_input_produces_valid_empty_outputs() {
        let out = std::env::temp_dir().join(format!("levelrl-native-scan-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&out);
        let summary = scan_pgn(
            Cursor::new(Vec::<u8>::new()),
            ScanOptions {
                out: out.clone(),
                ..ScanOptions::default()
            },
        )
        .unwrap();
        assert_eq!(summary.games, 0);
        assert_eq!(summary.rows, 0);
        assert_eq!(
            std::fs::read_to_string(out.join("counts.json")).unwrap(),
            "{}"
        );
        assert!(out.join("done").exists());
        std::fs::remove_dir_all(out).unwrap();
    }

    #[test]
    fn excessive_sampling_rate_is_an_error() {
        let error = scan_pgn(
            Cursor::new(Vec::<u8>::new()),
            ScanOptions {
                keep_bp: 10_001,
                ..ScanOptions::default()
            },
        )
        .unwrap_err();
        assert!(matches!(error, Error::InvalidArgument(_)));
    }
}
