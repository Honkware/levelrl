use std::fmt;
use std::io;

/// An error from feature extraction, data conversion, scanning, or band construction.
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    Io(io::Error),
    Json(serde_json::Error),
    InvalidArgument(String),
    InvalidFen(String),
    InvalidMove(String),
    IllegalMove(String),
    InvalidData(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(f, "I/O error: {error}"),
            Self::Json(error) => write!(f, "JSON error: {error}"),
            Self::InvalidArgument(message) => write!(f, "invalid argument: {message}"),
            Self::InvalidFen(fen) => write!(f, "invalid FEN: {fen}"),
            Self::InvalidMove(text) => write!(f, "invalid SAN/UCI move: {text}"),
            Self::IllegalMove(text) => write!(f, "illegal move: {text}"),
            Self::InvalidData(message) => write!(f, "invalid data: {message}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Json(error) => Some(error),
            _ => None,
        }
    }
}

impl From<io::Error> for Error {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<serde_json::Error> for Error {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error)
    }
}

/// The result type used by the crate's safe public API.
pub type Result<T> = std::result::Result<T, Error>;
