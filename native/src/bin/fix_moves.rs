use levelrl_native::cli::CliArgs;
use levelrl_native::{convert_position_directory, Result};
use std::env;
use std::path::PathBuf;
use std::process::ExitCode;

fn run() -> Result<()> {
    if env::args_os().any(|arg| arg == "--help" || arg == "-h") {
        println!("Usage: fix_moves [--in DIRECTORY] [--out DIRECTORY]");
        return Ok(());
    }
    let args = CliArgs::parse()?.finish(&["--in", "--out"])?;
    let input = PathBuf::from(args.value("--in")?.unwrap_or("data/positions_pre_uci_fix"));
    let output = PathBuf::from(args.value("--out")?.unwrap_or("data/positions_uci"));
    let summary = convert_position_directory(&input, &output)?;
    println!("validated and converted {} positions", summary.rows);
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("fix_moves: {error}");
            ExitCode::FAILURE
        }
    }
}
