use levelrl_native::cli::CliArgs;
use levelrl_native::{scan_pgn, Result, ScanOptions};
use std::env;
use std::io;
use std::path::PathBuf;
use std::process::ExitCode;

fn run() -> Result<()> {
    if env::args_os().any(|arg| arg == "--help" || arg == "-h") {
        println!("Usage: scan [--month YYYY-MM] [--keep-bp 0..10000] [--seed N] [--out DIRECTORY] < games.pgn");
        return Ok(());
    }
    let args = CliArgs::parse()?.finish(&["--month", "--keep-bp", "--seed", "--out"])?;
    let month = args.value("--month")?.unwrap_or("unknown").to_string();
    let options = ScanOptions {
        out: args
            .value("--out")?
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("data/scan").join(&month)),
        month,
        keep_bp: args.parse_value("--keep-bp", 100)?,
        seed: args.parse_value("--seed", 0)?,
    };
    scan_pgn(io::stdin().lock(), options)?;
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("scan: {error}");
            ExitCode::FAILURE
        }
    }
}
