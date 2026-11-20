use crate::model::PositionRow;
use crate::{Error, Result};
use flate2::{read::GzDecoder, write::GzEncoder, Compression};
use shakmaty::{san::SanPlus, uci::UciMove, CastlingMode, Chess, Move, Position};
use std::fs::{create_dir_all, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

/// Files and rows written by [`convert_position_directory`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConversionSummary {
    pub files: usize,
    pub rows: usize,
}

/// Converts a legal SAN, UCI, or legacy coordinate move to standard UCI.
pub fn move_to_uci(board: &Chess, text: &str) -> Result<String> {
    Ok(parse_move(board, text)?
        .to_uci(CastlingMode::Standard)
        .to_string())
}

/// Replaces every move in a position row with standard UCI, validating the sequence.
pub fn standardize_position_row(row: &mut PositionRow) -> Result<()> {
    let mut board = Chess::default();
    for move_text in &mut row.moves {
        let m = parse_move(&board, move_text)?;
        *move_text = m.to_uci(CastlingMode::Standard).to_string();
        board.play_unchecked(&m);
    }
    row.move_uci = move_to_uci(&board, &row.move_uci)?;
    Ok(())
}

/// Parses a JSON position row and returns it with standard UCI moves.
pub fn convert_position_row(input: &str) -> Result<String> {
    let mut row: PositionRow = serde_json::from_str(input)?;
    standardize_position_row(&mut row)?;
    Ok(serde_json::to_string(&row)?)
}

/// Converts every `.gz` JSON-lines file in a directory, preserving file names.
pub fn convert_position_directory(input: &Path, output: &Path) -> Result<ConversionSummary> {
    create_dir_all(output)?;
    let mut summary = ConversionSummary { files: 0, rows: 0 };
    for entry in std::fs::read_dir(input)? {
        let path = entry?.path();
        if path.extension().and_then(|name| name.to_str()) != Some("gz") {
            continue;
        }
        let name = path.file_name().ok_or_else(|| {
            Error::InvalidData(format!(
                "position path has no file name: {}",
                path.display()
            ))
        })?;
        let reader = BufReader::new(GzDecoder::new(File::open(&path)?));
        let mut writer = GzEncoder::new(
            BufWriter::new(File::create(output.join(name))?),
            Compression::default(),
        );
        for (index, line) in reader.lines().enumerate() {
            let line = line?;
            let converted = convert_position_row(&line).map_err(|error| {
                Error::InvalidData(format!("{}:{}: {error}", path.display(), index + 1))
            })?;
            writer.write_all(converted.as_bytes())?;
            writer.write_all(b"\n")?;
            summary.rows += 1;
        }
        writer.finish()?;
        summary.files += 1;
    }
    Ok(summary)
}

fn parse_move(board: &Chess, text: &str) -> Result<Move> {
    if let Ok(m) = text.parse::<SanPlus>() {
        if let Ok(m) = m.san.to_move(board) {
            return Ok(m);
        }
    }
    let mut normalized = text.replace(['-', '='], "");
    if normalized.chars().next().is_some_and(char::is_uppercase) {
        normalized.remove(0);
    }
    if let Some(last) = normalized.chars().last().filter(|last| last.is_uppercase()) {
        normalized.truncate(normalized.len() - last.len_utf8());
        normalized.extend(last.to_lowercase());
    }
    normalized
        .parse::<UciMove>()
        .map_err(|_| Error::InvalidMove(text.to_string()))?
        .to_move(board)
        .map_err(|_| Error::IllegalMove(text.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converts_san_coordinate_and_legacy_promotion_moves() {
        let board = Chess::default();
        assert_eq!(move_to_uci(&board, "e4").unwrap(), "e2e4");
        assert_eq!(move_to_uci(&board, "e2-e4").unwrap(), "e2e4");
        let promotion: Chess = "4k3/P7/8/8/8/8/8/4K3 w - - 0 1"
            .parse::<shakmaty::fen::Fen>()
            .unwrap()
            .into_position(CastlingMode::Standard)
            .unwrap();
        assert_eq!(move_to_uci(&promotion, "a7-a8=Q").unwrap(), "a7a8q");
    }

    #[test]
    fn malformed_and_illegal_moves_are_structured_errors() {
        let board = Chess::default();
        assert!(matches!(
            move_to_uci(&board, "É"),
            Err(Error::InvalidMove(_))
        ));
        assert!(matches!(
            move_to_uci(&board, "e2e5"),
            Err(Error::IllegalMove(_))
        ));
        assert!(matches!(
            convert_position_row("not json"),
            Err(Error::Json(_))
        ));
    }

    #[test]
    fn converts_a_complete_position_row() {
        let input = r#"{"site":"game","moves":["e4","e7-e5"],"move":"Nf3","elo":1500,"oppo_elo":1480,"white":"a","black":"b","tc":"180+0","month":"2026-03"}"#;
        let row: PositionRow = serde_json::from_str(&convert_position_row(input).unwrap()).unwrap();
        assert_eq!(row.moves, ["e2e4", "e7e5"]);
        assert_eq!(row.move_uci, "g1f3");
    }
}
