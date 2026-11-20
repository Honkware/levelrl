use serde::{Deserialize, Serialize};

/// One selected position and the move played by the side to move.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct PositionRow {
    pub site: String,
    pub moves: Vec<String>,
    #[serde(rename = "move")]
    pub move_uci: String,
    pub elo: i32,
    pub oppo_elo: i32,
    pub white: String,
    pub black: String,
    pub tc: String,
    pub month: String,
}
