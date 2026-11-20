use levelrl_native::cli::CliArgs;
use levelrl_native::{build_bands, BandOptions, Error, Result};
use std::env;
use std::path::PathBuf;
use std::process::ExitCode;

fn limit(value: Option<&str>, name: &str) -> Result<Option<usize>> {
    match value {
        None | Some("-1") => Ok(None),
        Some(value) => value.parse::<usize>().map(Some).map_err(|error| {
            Error::InvalidArgument(format!(
                "{name} expects a nonnegative integer or -1, got `{value}`: {error}"
            ))
        }),
    }
}

fn run() -> Result<()> {
    if env::args_os().any(|arg| arg == "--help" || arg == "-h") {
        println!(
            "Usage: band --months 'YYYY-MM [...]' --bands 'ELO [...]' [--root DIRECTORY] [--out DIRECTORY] [--seed N] [--n-cal N|-1] [--n-fid N|-1]"
        );
        return Ok(());
    }
    let args = CliArgs::parse()?.finish(&[
        "--months", "--bands", "--seed", "--out", "--root", "--n-cal", "--n-fid",
    ])?;
    let bands = args
        .values("--bands")?
        .iter()
        .map(|value| {
            value.parse::<i32>().map_err(|error| {
                Error::InvalidArgument(format!(
                    "--bands contains invalid rating `{value}`: {error}"
                ))
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let options = BandOptions {
        months: args.values("--months")?,
        bands,
        seed: args.parse_value("--seed", 0)?,
        out: PathBuf::from(args.value("--out")?.unwrap_or("data/positions")),
        root: PathBuf::from(args.value("--root")?.unwrap_or(".")),
        n_cal: limit(args.value("--n-cal")?, "--n-cal")?,
        n_fid: limit(args.value("--n-fid")?, "--n-fid")?,
    };
    build_bands(options)?;
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("band: {error}");
            ExitCode::FAILURE
        }
    }
}
