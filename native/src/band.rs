use crate::model::PositionRow;
use crate::rng::Rng;
use crate::{Error, Result};
use flate2::{write::GzEncoder, Compression};
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::fs::{create_dir_all, read_to_string, rename, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Input selection and output settings for [`build_bands`].
#[derive(Debug, Clone)]
pub struct BandOptions {
    pub months: Vec<String>,
    pub bands: Vec<i32>,
    pub seed: u64,
    pub out: PathBuf,
    pub root: PathBuf,
    pub n_cal: Option<usize>,
    pub n_fid: Option<usize>,
}

impl Default for BandOptions {
    fn default() -> Self {
        Self {
            months: Vec::new(),
            bands: Vec::new(),
            seed: 0,
            out: PathBuf::from("data/positions"),
            root: PathBuf::from("."),
            n_cal: None,
            n_fid: None,
        }
    }
}

/// Number of rows written for each requested rating band.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BandSummary {
    pub rows: HashMap<i32, usize>,
}

fn shuffle<T>(rows: &mut [T], rng: &mut Rng) {
    for i in (1..rows.len()).rev() {
        let j = (rng.next() as usize) % (i + 1);
        rows.swap(i, j);
    }
}

/// Applies the Lichess blitz rule: base plus 40 increments is 180 to 479 seconds.
pub fn is_blitz(tc: &str) -> bool {
    let mut parts = tc.split('+');
    let base = parts.next().unwrap_or("").parse::<u64>();
    let inc = parts.next().unwrap_or("").parse::<u64>();
    parts.next().is_none()
        && base
            .ok()
            .zip(inc.ok())
            .and_then(|(base, inc)| inc.checked_mul(40).and_then(|inc| base.checked_add(inc)))
            .is_some_and(|seconds| (180..480).contains(&seconds))
}

fn read_rows(path: &Path) -> Result<Vec<PositionRow>> {
    let file = File::open(path)?;
    BufReader::new(file)
        .lines()
        .enumerate()
        .map(|(index, line)| {
            let line = line?;
            serde_json::from_str(&line).map_err(|error| {
                Error::InvalidData(format!("{}:{}: {error}", path.display(), index + 1))
            })
        })
        .collect()
}

fn write_rows(path: &Path, rows: &[PositionRow]) -> Result<()> {
    let tmp = path.with_extension("tmp");
    let mut writer = GzEncoder::new(BufWriter::new(File::create(&tmp)?), Compression::default());
    for row in rows {
        serde_json::to_writer(&mut writer, row)?;
        writer.write_all(b"\n")?;
    }
    writer.finish()?;
    rename(tmp, path)?;
    Ok(())
}

fn read_manifest(path: &Path) -> Result<Value> {
    match read_to_string(path) {
        Ok(text) => {
            let value: Value = serde_json::from_str(&text)
                .map_err(|error| Error::InvalidData(format!("{}: {error}", path.display())))?;
            if !value.is_object() {
                return Err(Error::InvalidData(format!(
                    "{} must contain a JSON object",
                    path.display()
                )));
            }
            Ok(value)
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(json!({})),
        Err(error) => Err(Error::Io(error)),
    }
}

/// Builds rating-band files from native scan outputs and updates the manifest.
pub fn build_bands(options: BandOptions) -> Result<BandSummary> {
    if options.months.is_empty() {
        return Err(Error::InvalidArgument(
            "--months requires at least one month".to_string(),
        ));
    }
    if options.bands.is_empty() {
        return Err(Error::InvalidArgument(
            "--bands requires at least one rating".to_string(),
        ));
    }
    let started = Instant::now();
    let mut counts: HashMap<String, usize> = HashMap::new();
    let mut by_month = Vec::new();
    for month in &options.months {
        let dir = options.root.join("data/scan").join(month);
        let counts_path = dir.join("counts.json");
        let text = read_to_string(&counts_path)?;
        let month_counts: HashMap<String, usize> = serde_json::from_str(&text)
            .map_err(|error| Error::InvalidData(format!("{}: {error}", counts_path.display())))?;
        for (name, n) in month_counts {
            let total = counts.entry(name).or_insert(0);
            *total = total.checked_add(n).ok_or_else(|| {
                Error::InvalidData(format!(
                    "player count overflow in {}",
                    counts_path.display()
                ))
            })?;
        }
        by_month.push(read_rows(&dir.join("rows.jsonl"))?);
    }
    let n_cal = options
        .n_cal
        .unwrap_or_else(|| 20_000usize.div_ceil(options.months.len()));
    let n_fid = options
        .n_fid
        .unwrap_or_else(|| 50_000usize.div_ceil(options.months.len()));
    let mut rng = Rng(options.seed);
    let mut fid: HashMap<i32, Vec<PositionRow>> = options
        .bands
        .iter()
        .map(|band| (*band, Vec::new()))
        .collect();
    for rows in &mut by_month {
        shuffle(rows, &mut rng);
        let used: HashSet<String> = rows
            .iter()
            .take(n_cal)
            .map(|row| row.site.clone())
            .collect();
        for row in rows.iter().skip(n_cal) {
            if used.contains(&row.site) || !is_blitz(&row.tc) {
                continue;
            }
            let band = options
                .bands
                .iter()
                .find(|band| (i64::from(row.elo) - i64::from(**band)).abs() <= 50);
            if let Some(band) = band {
                if counts.get(&row.white).copied().unwrap_or(0) >= 50
                    && counts.get(&row.black).copied().unwrap_or(0) >= 50
                {
                    if let Some(dst) = fid.get_mut(band) {
                        if dst.len() < n_fid {
                            dst.push(row.clone());
                        }
                    }
                }
            }
        }
    }
    create_dir_all(&options.out)?;
    let manifest_path = options.out.join("manifest.json");
    let old = read_manifest(&manifest_path)?;
    let mut manifest = old.clone();
    manifest["cutoff"] = json!("2025-07");
    let mut all_months: Vec<String> = old
        .get("months")
        .and_then(Value::as_array)
        .map(|values| {
            values
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default();
    for month in &options.months {
        if !all_months.contains(month) {
            all_months.push(month.clone());
        }
    }
    all_months.sort();
    manifest["months"] = json!(all_months);
    manifest["seed"] = json!(options.seed);
    let mut all_bands: Vec<i32> = old
        .get("bands")
        .and_then(Value::as_array)
        .map(|values| {
            values
                .iter()
                .filter_map(Value::as_i64)
                .map(|band| band as i32)
                .collect()
        })
        .unwrap_or_default();
    for band in &options.bands {
        if !all_bands.contains(band) {
            all_bands.push(*band);
        }
    }
    all_bands.sort();
    manifest["bands"] = json!(all_bands);
    manifest["half"] = json!(50);
    manifest["min_games"] = json!(50);
    manifest["skip_plies"] = json!(10);
    manifest["min_clock"] = json!(30.0);
    manifest["keep"] = json!(0.01);
    manifest["speed"] = json!("blitz");
    manifest["engine"] = json!("rust");
    let mut fid_counts = old.get("fid").cloned().unwrap_or_else(|| json!({}));
    if let Some(object) = fid_counts.as_object_mut() {
        for (band, rows) in &fid {
            object.insert(band.to_string(), json!(rows.len()));
        }
    }
    manifest["fid"] = fid_counts;
    let mut band_months = old.get("band_months").cloned().unwrap_or_else(|| json!({}));
    let band_months_object = band_months.as_object_mut().ok_or_else(|| {
        Error::InvalidData(format!(
            "{}: band_months must contain a JSON object",
            manifest_path.display()
        ))
    })?;
    for band in &options.bands {
        band_months_object.insert(band.to_string(), json!(options.months));
    }
    manifest["band_months"] = band_months;
    let summary = BandSummary {
        rows: fid.iter().map(|(band, rows)| (*band, rows.len())).collect(),
    };
    for (band, rows) in fid {
        write_rows(&options.out.join(format!("fid_{band}.jsonl.gz")), &rows)?;
    }
    std::fs::write(&manifest_path, serde_json::to_string_pretty(&manifest)?)?;
    eprintln!(
        "band build: {} targets, {:.2}s",
        options.bands.len(),
        started.elapsed().as_secs_f64()
    );
    Ok(summary)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blitz_rejects_malformed_and_extreme_time_controls() {
        assert!(is_blitz("180+0"));
        assert!(is_blitz("120+2"));
        assert!(!is_blitz("18446744073709551615+18446744073709551615"));
        assert!(!is_blitz("180+0+0"));
    }

    #[test]
    fn empty_selection_is_an_error() {
        assert!(matches!(
            build_bands(BandOptions::default()),
            Err(Error::InvalidArgument(_))
        ));
    }

    #[test]
    fn malformed_rows_are_errors_instead_of_panics() {
        let root = std::env::temp_dir().join(format!("levelrl-native-band-{}", std::process::id()));
        let scan = root.join("data/scan/2026-03");
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&scan).unwrap();
        std::fs::write(scan.join("counts.json"), "{}").unwrap();
        std::fs::write(scan.join("rows.jsonl"), "not json\n").unwrap();
        let error = build_bands(BandOptions {
            months: vec!["2026-03".to_string()],
            bands: vec![1500],
            root: root.clone(),
            ..BandOptions::default()
        })
        .unwrap_err();
        assert!(matches!(error, Error::InvalidData(_)));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn incremental_builds_preserve_other_band_months() {
        let root =
            std::env::temp_dir().join(format!("levelrl-native-band-merge-{}", std::process::id()));
        let scan = root.join("data/scan/2026-03");
        let out = root.join("data/positions");
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&scan).unwrap();
        std::fs::create_dir_all(&out).unwrap();
        std::fs::write(scan.join("counts.json"), "{}").unwrap();
        std::fs::write(scan.join("rows.jsonl"), "").unwrap();
        std::fs::write(
            out.join("manifest.json"),
            r#"{"band_months":{"1100":["2025-09"]}}"#,
        )
        .unwrap();

        build_bands(BandOptions {
            months: vec!["2026-03".to_string()],
            bands: vec![700],
            root: root.clone(),
            out: out.clone(),
            ..BandOptions::default()
        })
        .unwrap();

        let manifest: Value =
            serde_json::from_str(&std::fs::read_to_string(out.join("manifest.json")).unwrap())
                .unwrap();
        assert_eq!(manifest["band_months"]["1100"], json!(["2025-09"]));
        assert_eq!(manifest["band_months"]["700"], json!(["2026-03"]));
        std::fs::remove_dir_all(root).unwrap();
    }
}
