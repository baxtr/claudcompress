use clap::{Parser, Subcommand};
use std::fs;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "claudcompress", about = "QICM quantum compression")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Compress a text file
    Compress {
        /// Input file
        file: PathBuf,
        /// Output file (default: <file>.cqz)
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Number of threads (default: auto-detect)
        #[arg(short, long, default_value_t = 0)]
        threads: usize,
    },
    /// Decompress a .cqz file
    Decompress {
        /// Input file
        file: PathBuf,
        /// Output file (default: strip .cqz extension)
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Number of threads (default: auto-detect)
        #[arg(short, long, default_value_t = 0)]
        threads: usize,
    },
    /// Show compression ratio without writing output
    Ratio {
        /// Input file
        file: PathBuf,
        /// Number of threads (default: auto-detect)
        #[arg(short, long, default_value_t = 0)]
        threads: usize,
    },
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Compress { file, output, threads } => {
            let text = fs::read_to_string(&file).unwrap_or_else(|e| {
                eprintln!("Error reading {}: {e}", file.display());
                std::process::exit(1);
            });
            let compressed = claudcompress::quantum_compress_threads(&text, threads);
            let out_path = output.unwrap_or_else(|| {
                let mut p = file.clone();
                let name = format!("{}.cqz", p.file_name().unwrap().to_string_lossy());
                p.set_file_name(name);
                p
            });
            fs::write(&out_path, &compressed).unwrap_or_else(|e| {
                eprintln!("Error writing {}: {e}", out_path.display());
                std::process::exit(1);
            });
            eprintln!("  Written to {}", out_path.display());
        }
        Commands::Decompress { file, output, threads } => {
            let data = fs::read(&file).unwrap_or_else(|e| {
                eprintln!("Error reading {}: {e}", file.display());
                std::process::exit(1);
            });
            let text = claudcompress::quantum_decompress_threads(&data, threads).unwrap_or_else(|e| {
                eprintln!("Error: {e}");
                std::process::exit(1);
            });
            let out_path = output.unwrap_or_else(|| {
                let s = file.to_string_lossy();
                if let Some(stripped) = s.strip_suffix(".cqz") {
                    PathBuf::from(stripped)
                } else {
                    let mut p = file.clone();
                    let name = format!("{}.txt", p.file_name().unwrap().to_string_lossy());
                    p.set_file_name(name);
                    p
                }
            });
            fs::write(&out_path, &text).unwrap_or_else(|e| {
                eprintln!("Error writing {}: {e}", out_path.display());
                std::process::exit(1);
            });
            eprintln!("  Written to {}", out_path.display());
        }
        Commands::Ratio { file, threads } => {
            let text = fs::read_to_string(&file).unwrap_or_else(|e| {
                eprintln!("Error reading {}: {e}", file.display());
                std::process::exit(1);
            });
            let _ = claudcompress::quantum_compress_threads(&text, threads);
        }
    }
}
