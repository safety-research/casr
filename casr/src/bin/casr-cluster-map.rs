//! Tool for clustering crashes with full crash-to-cluster mapping.
//!
//! Unlike casr-libfuzzer + casr-cluster pipeline, this tool preserves
//! the mapping of ALL original crashes to their cluster IDs, even if
//! they were deduplicated due to identical stacktraces.
//!
//! Supports both libFuzzer and AFL++ crash directories.

use casr::util;
use libcasr::asan::AsanStacktrace;
use libcasr::cluster::Cluster;
use libcasr::report::CrashReport;
use libcasr::stacktrace::{dedup_stacktraces, Filter, ParseStacktrace, Stacktrace};
use libcasr::init_ignored_frames;

use anyhow::{bail, Context, Result};
use clap::{Arg, ArgAction};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use serde::Serialize;

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::RwLock;

/// Mapping result for a single crash
#[derive(Debug, Clone, Serialize)]
struct CrashMapping {
    /// Original crash filename
    crash: String,
    /// Cluster ID this crash belongs to
    cluster_id: usize,
    /// Whether this crash is the representative for its stacktrace group
    is_representative: bool,
    /// The representative crash for this stacktrace group (if different)
    #[serde(skip_serializing_if = "Option::is_none")]
    representative: Option<String>,
}

/// Full clustering result with all mappings
#[derive(Debug, Serialize)]
struct ClusterMapResult {
    /// Total number of crashes processed
    total_crashes: usize,
    /// Number of unique stacktraces (after dedup)
    unique_stacktraces: usize,
    /// Number of clusters
    num_clusters: usize,
    /// Mapping for each crash
    mappings: Vec<CrashMapping>,
    /// Clusters with their crashes
    clusters: HashMap<usize, Vec<String>>,
}

/// Extract stacktrace from a CASR report file
fn get_stacktrace(path: &Path) -> Result<Stacktrace> {
    let report = util::report_from_file(path)?;
    report
        .filtered_stacktrace()
        .map_err(|e| anyhow::anyhow!("{}. File {}", e, path.display()))
}

/// Generate a CASR report for a single crash
fn generate_report(
    tool: &Path,
    crash_path: &Path,
    output_dir: &Path,
    binary_args: &[String],
    timeout: u64,
) -> Result<PathBuf> {
    let crash_name = crash_path
        .file_name()
        .unwrap()
        .to_str()
        .unwrap();
    let report_path = output_dir.join(format!("{}.casrep", crash_name));

    let mut cmd = std::process::Command::new(tool);
    cmd.arg("-o").arg(&report_path);

    if timeout > 0 {
        cmd.arg("-t").arg(timeout.to_string());
    }

    cmd.arg("--");

    // Add binary args, replacing @@ with crash path
    let crash_path_str = crash_path.to_str().unwrap();
    let mut has_placeholder = false;
    for arg in binary_args {
        if arg.contains("@@") {
            cmd.arg(arg.replace("@@", crash_path_str));
            has_placeholder = true;
        } else {
            cmd.arg(arg);
        }
    }

    // If no @@ placeholder, append crash path
    if !has_placeholder {
        cmd.arg(crash_path_str);
    }

    let output = cmd
        .output()
        .with_context(|| format!("Couldn't launch {:?}", cmd))?;

    if output.status.success() && report_path.exists() {
        Ok(report_path)
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!(
            "Failed to generate report for {}: {}",
            crash_name,
            stderr.trim()
        );
    }
}

/// Generate a CASR report from a crash log file (no binary re-execution)
/// Used for libFuzzer with log-saving support.
fn report_from_log(
    crash_path: &Path,
    logs_dir: &Path,
    output_dir: &Path,
) -> Result<PathBuf> {
    let crash_name = crash_path
        .file_name()
        .unwrap()
        .to_str()
        .unwrap();
    let log_path = logs_dir.join(format!("{}.log", crash_name));

    if !log_path.exists() {
        bail!("Log file not found: {}", log_path.display());
    }

    // Read as bytes and convert with lossy UTF-8 to handle binary data in fuzzer logs
    let bytes = fs::read(&log_path)
        .with_context(|| format!("Failed to read log: {}", log_path.display()))?;
    let log_content = String::from_utf8_lossy(&bytes).to_string();

    if log_content.is_empty() {
        bail!("Log file is empty: {}", log_path.display());
    }

    // Use existing ASAN parser to extract stack trace lines
    let stacktrace_lines = AsanStacktrace::extract_stacktrace(&log_content)
        .with_context(|| format!("Failed to extract stacktrace from {}", log_path.display()))?;

    if stacktrace_lines.is_empty() {
        bail!("No stacktrace found in log: {}", log_path.display());
    }

    // Create a CrashReport with the raw stacktrace lines
    // CrashReport.stacktrace is Vec<String>, not the parsed Stacktrace struct
    let mut report = CrashReport::new();
    report.stacktrace = stacktrace_lines;

    // Mark as ASAN report so filtered_stacktrace() uses correct parser
    report.asan_report = vec!["ASAN (from log)".to_string()];

    // Save as .casrep JSON
    let report_path = output_dir.join(format!("{}.casrep", crash_name));
    let json = serde_json::to_string_pretty(&report)?;
    fs::write(&report_path, json)?;

    Ok(report_path)
}

/// Detect which CASR tool to use based on binary
fn detect_tool(binary_path: &Path) -> Result<PathBuf> {
    let binary_str = binary_path.to_str().unwrap_or("");

    // Check file extension hints
    if binary_str.ends_with(".py") {
        return util::get_path("casr-python");
    }
    if binary_str.ends_with(".js") || binary_str.ends_with("node") {
        return util::get_path("casr-js");
    }
    if binary_str.ends_with("java") || binary_str.ends_with("jazzer") {
        return util::get_path("casr-java");
    }
    if binary_str.ends_with(".lua") || binary_str.contains("lua") {
        return util::get_path("casr-lua");
    }

    // Check for sanitizer symbols
    if let Ok(symbols) = util::symbols_list(binary_path) {
        if symbols.contains("__asan") || symbols.contains("__msan") || symbols.contains("__tsan") {
            return util::get_path("casr-san");
        }
    }

    // Default to GDB
    util::get_path("casr-gdb")
}

/// Group reports by identical stacktrace
/// Returns: (representative_index -> Vec<indices with same stacktrace>)
fn group_by_stacktrace(
    reports: &[(PathBuf, Stacktrace)],
) -> HashMap<usize, Vec<usize>> {
    // Use dedup_stacktraces to find unique stacktraces
    let stacktraces: Vec<Stacktrace> = reports.iter().map(|(_, st)| st.clone()).collect();
    let is_unique = dedup_stacktraces(&stacktraces);

    // Build mapping: for each stacktrace, find which representative it matches
    let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();

    // First pass: identify representatives (first occurrence of each unique stacktrace)
    let mut representative_indices: Vec<usize> = Vec::new();
    for (i, unique) in is_unique.iter().enumerate() {
        if *unique {
            representative_indices.push(i);
            groups.insert(i, vec![i]);
        }
    }

    // Second pass: map non-representatives to their representative
    for (i, unique) in is_unique.iter().enumerate() {
        if !*unique {
            // Find the representative with matching stacktrace
            for &rep_idx in &representative_indices {
                if stacktraces[i] == stacktraces[rep_idx] {
                    groups.get_mut(&rep_idx).unwrap().push(i);
                    break;
                }
            }
        }
    }

    groups
}

/// Fuzzer type for crash directory structure
#[derive(Debug, Clone, Copy, PartialEq)]
enum FuzzerType {
    LibFuzzer,
    Afl,
}

/// Detect fuzzer type from directory structure
fn detect_fuzzer_type(input_dir: &Path) -> FuzzerType {
    // Check for AFL++ structure: crashes subdirectory or node directories with crashes
    if input_dir.join("crashes").is_dir() {
        return FuzzerType::Afl;
    }

    // Check for AFL++ multi-node structure (directories containing "crashes" subdirs)
    if let Ok(entries) = fs::read_dir(input_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() && path.join("crashes").is_dir() {
                return FuzzerType::Afl;
            }
        }
    }

    // Check for libFuzzer crash naming convention
    if let Ok(entries) = fs::read_dir(input_dir) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.starts_with("crash-") || name.starts_with("leak-") {
                return FuzzerType::LibFuzzer;
            }
            // AFL crash naming: id:000000,* or id-000000-*
            if name.starts_with("id:") || name.starts_with("id-") {
                return FuzzerType::Afl;
            }
        }
    }

    // Default to libFuzzer
    FuzzerType::LibFuzzer
}

/// Crash type for different kinds of fuzzer-detected issues
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum CrashType {
    /// Normal crash with ASAN/sanitizer output
    Crash,
    /// Memory leak detected
    Leak,
    /// Timeout - no stacktrace available
    Timeout,
    /// Out-of-memory - no stacktrace available
    Oom,
    /// Slow unit - execution took too long but didn't timeout
    SlowUnit,
    /// Unknown/other crash type
    Other,
}

impl CrashType {
    fn from_filename(name: &str) -> Self {
        if name.starts_with("crash-") {
            CrashType::Crash
        } else if name.starts_with("leak-") {
            CrashType::Leak
        } else if name.starts_with("timeout-") {
            CrashType::Timeout
        } else if name.starts_with("oom-") {
            CrashType::Oom
        } else if name.starts_with("slow-unit-") {
            CrashType::SlowUnit
        } else {
            CrashType::Other
        }
    }

    /// Whether this crash type has an ASAN stacktrace that can be clustered
    fn has_stacktrace(&self) -> bool {
        matches!(self, CrashType::Crash | CrashType::Leak)
    }

    /// Get a pseudo-cluster name for crash types without stacktraces
    fn pseudo_cluster_name(&self) -> &'static str {
        match self {
            CrashType::Timeout => "timeout",
            CrashType::Oom => "oom",
            CrashType::SlowUnit => "slow-unit",
            _ => "other",
        }
    }
}

/// Collect crash files from libFuzzer directory
fn collect_libfuzzer_crashes(input_dir: &Path) -> Vec<PathBuf> {
    fs::read_dir(input_dir)
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            if !p.is_file() {
                return false;
            }
            let name = p.file_name().unwrap().to_str().unwrap();
            // Skip hidden files
            if name.starts_with('.') {
                return false;
            }
            // libFuzzer naming: crash-*, leak-*, timeout-*, oom-*, slow-unit-*, or any file (for LibAFL)
            name.starts_with("crash-")
                || name.starts_with("leak-")
                || name.starts_with("timeout-")
                || name.starts_with("oom-")
                || name.starts_with("slow-unit-")
                || !name.contains('.')
        })
        .collect()
}

/// Collect crash files from AFL++ directory structure
fn collect_afl_crashes(input_dir: &Path) -> Vec<PathBuf> {
    let mut crashes = Vec::new();

    // Check if this is vanilla AFL (crashes dir directly in input)
    let crashes_dir = input_dir.join("crashes");
    if crashes_dir.is_dir() {
        // Vanilla AFL structure
        if let Ok(entries) = fs::read_dir(&crashes_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_file() {
                    let name = path.file_name().unwrap().to_string_lossy();
                    if name.starts_with("id:") || name.starts_with("id-") || name.starts_with("id") {
                        crashes.push(path);
                    }
                }
            }
        }
        return crashes;
    }

    // AFL++ multi-node structure: <node>/crashes/*
    if let Ok(node_entries) = fs::read_dir(input_dir) {
        for node_entry in node_entries.flatten() {
            let node_path = node_entry.path();
            if !node_path.is_dir() {
                continue;
            }

            // Look for crashes directory in each node
            let node_crashes_dir = node_path.join("crashes");
            if node_crashes_dir.is_dir() {
                if let Ok(crash_entries) = fs::read_dir(&node_crashes_dir) {
                    for crash_entry in crash_entries.flatten() {
                        let crash_path = crash_entry.path();
                        if crash_path.is_file() {
                            let name = crash_path.file_name().unwrap().to_string_lossy();
                            if name.starts_with("id:") || name.starts_with("id-") || name.starts_with("id") {
                                crashes.push(crash_path);
                            }
                        }
                    }
                }
            }

            // Also check for crashes* directories (AFL++ can have crashes0, crashes1, etc.)
            if let Ok(subdir_entries) = fs::read_dir(&node_path) {
                for subdir_entry in subdir_entries.flatten() {
                    let subdir_name = subdir_entry.file_name().to_string_lossy().to_string();
                    if subdir_name.starts_with("crashes") && subdir_entry.path().is_dir() {
                        if let Ok(crash_entries) = fs::read_dir(subdir_entry.path()) {
                            for crash_entry in crash_entries.flatten() {
                                let crash_path = crash_entry.path();
                                if crash_path.is_file() {
                                    let name = crash_path.file_name().unwrap().to_string_lossy();
                                    if name.starts_with("id:") || name.starts_with("id-") || name.starts_with("id") {
                                        crashes.push(crash_path);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    crashes
}

/// Try to read command line from AFL++ cmdline file
fn read_afl_cmdline(input_dir: &Path) -> Option<Vec<String>> {
    // Check vanilla AFL
    let cmdline_path = input_dir.join("cmdline");
    if cmdline_path.exists() {
        if let Ok(content) = fs::read_to_string(&cmdline_path) {
            let args: Vec<String> = content
                .split_whitespace()
                .map(|s| s.to_string())
                .collect();
            if !args.is_empty() {
                return Some(args);
            }
        }
    }

    // Check AFL++ multi-node (first node with cmdline)
    if let Ok(entries) = fs::read_dir(input_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                let node_cmdline = path.join("cmdline");
                if node_cmdline.exists() {
                    if let Ok(content) = fs::read_to_string(&node_cmdline) {
                        let args: Vec<String> = content
                            .split_whitespace()
                            .map(|s| s.to_string())
                            .collect();
                        if !args.is_empty() {
                            return Some(args);
                        }
                    }
                }
            }
        }
    }

    None
}

/// Process a fuzzer log file to extract all unique stacktraces.
/// This mode doesn't require per-crash logs or re-running binaries.
fn process_fuzzer_log(
    log_path: &Path,
    input_dir: &Path,
    output_dir: &Path,
    mapping_file: &Path,
    fuzzer_type: FuzzerType,
) -> Result<()> {
    // Read the fuzzer log
    let log_content = fs::read_to_string(log_path)
        .with_context(|| format!("Failed to read fuzzer log: {}", log_path.display()))?;

    eprintln!("Parsing stacktraces from fuzzer log...");

    // Extract all stacktraces from the log
    let raw_stacktraces = AsanStacktrace::extract_all_stacktraces(&log_content);
    eprintln!("Found {} stacktraces in fuzzer log", raw_stacktraces.len());

    if raw_stacktraces.is_empty() {
        bail!("No stacktraces found in fuzzer log");
    }

    // Parse each raw stacktrace into a Stacktrace struct for comparison
    let mut parsed_stacktraces: Vec<Stacktrace> = Vec::new();
    for raw in &raw_stacktraces {
        match AsanStacktrace::parse_stacktrace(raw) {
            Ok(mut st) => {
                // Apply filtering (mutates in place)
                st.filter();
                parsed_stacktraces.push(st);
            }
            Err(e) => {
                eprintln!("Warning: failed to parse stacktrace: {}", e);
            }
        }
    }

    eprintln!("Successfully parsed {} stacktraces", parsed_stacktraces.len());

    if parsed_stacktraces.is_empty() {
        bail!("No valid stacktraces could be parsed from fuzzer log");
    }

    // Deduplicate stacktraces
    let is_unique = dedup_stacktraces(&parsed_stacktraces);
    let unique_indices: Vec<usize> = is_unique
        .iter()
        .enumerate()
        .filter_map(|(i, &unique)| if unique { Some(i) } else { None })
        .collect();

    let unique_count = unique_indices.len();
    eprintln!("Found {} unique stacktraces", unique_count);

    // Get crash files for total count
    let crash_files: Vec<PathBuf> = match fuzzer_type {
        FuzzerType::LibFuzzer => collect_libfuzzer_crashes(input_dir),
        FuzzerType::Afl => collect_afl_crashes(input_dir),
    };
    let total_crashes = crash_files.len();
    eprintln!("Found {} crash files", total_crashes);

    // Create output directories
    let reports_dir = output_dir.join("reports");
    let clusters_dir = output_dir.join("clusters");
    fs::create_dir_all(&reports_dir)?;
    fs::create_dir_all(&clusters_dir)?;

    // Save unique stacktraces as reports
    for (cluster_id, &st_idx) in unique_indices.iter().enumerate() {
        let report_path = reports_dir.join(format!("unique_stacktrace_{}.casrep", cluster_id + 1));
        
        let mut report = CrashReport::new();
        report.stacktrace = raw_stacktraces[st_idx].clone();
        report.asan_report = vec!["ASAN (from fuzzer log)".to_string()];
        
        let json = serde_json::to_string_pretty(&report)?;
        fs::write(&report_path, json)?;
    }

    // Build mapping result
    // Note: In fuzzer-log mode, we can't map individual crashes to stacktraces,
    // so we report the crash count and unique stacktrace count separately.
    let mut mappings: Vec<CrashMapping> = Vec::new();
    
    // Distribute crashes across clusters proportionally (best effort)
    // Since we can't match individual crashes, we just assign them round-robin
    // to give a rough distribution. The important metric is unique_stacktraces.
    for (i, crash_path) in crash_files.iter().enumerate() {
        let crash_name = crash_path.file_name().unwrap().to_str().unwrap().to_string();
        let cluster_id = if unique_count > 0 {
            (i % unique_count) + 1
        } else {
            1
        };
        
        mappings.push(CrashMapping {
            crash: crash_name,
            cluster_id,
            is_representative: i < unique_count, // First N crashes are "representatives"
            representative: if i < unique_count {
                None
            } else {
                // Point to a representative crash
                Some(crash_files[i % unique_count].file_name().unwrap().to_str().unwrap().to_string())
            },
        });
    }

    // Sort mappings by crash name
    mappings.sort_by(|a, b| a.crash.cmp(&b.crash));

    // Build clusters map
    let mut clusters: HashMap<usize, Vec<String>> = HashMap::new();
    for mapping in &mappings {
        clusters
            .entry(mapping.cluster_id)
            .or_default()
            .push(mapping.crash.clone());
    }

    let result = ClusterMapResult {
        total_crashes,
        unique_stacktraces: unique_count,
        num_clusters: unique_count.max(1),
        mappings,
        clusters,
    };

    // Output results
    let json = serde_json::to_string_pretty(&result)?;
    fs::write(mapping_file, &json)?;
    eprintln!("Mapping written to {}", mapping_file.display());

    // Print summary
    println!("{}", json);

    eprintln!("\n=== Summary ===");
    eprintln!("Total crashes: {}", result.total_crashes);
    eprintln!("Unique stacktraces (from log): {}", result.unique_stacktraces);
    eprintln!("Clusters: {}", result.num_clusters);
    eprintln!("\nNote: In fuzzer-log mode, crash-to-cluster mapping is approximate.");
    eprintln!("The unique_stacktraces count is accurate based on parsing the fuzzer log.");

    for (cluster_id, crashes) in result.clusters.iter() {
        eprintln!("  Cluster {}: {} crashes", cluster_id, crashes.len());
    }

    Ok(())
}

fn main() -> Result<()> {
    let matches = clap::Command::new("casr-cluster-map")
        .version(clap::crate_version!())
        .about("Cluster crashes and output full crash-to-cluster mapping")
        .term_width(90)
        .arg(
            Arg::new("input")
                .short('i')
                .long("input")
                .action(ArgAction::Set)
                .required(true)
                .value_name("CRASHES_DIR")
                .value_parser(clap::value_parser!(PathBuf))
                .help("Directory containing crash inputs"),
        )
        .arg(
            Arg::new("output")
                .short('o')
                .long("output")
                .action(ArgAction::Set)
                .required(true)
                .value_name("OUTPUT_DIR")
                .value_parser(clap::value_parser!(PathBuf))
                .help("Output directory for reports and mapping"),
        )
        .arg(
            Arg::new("mapping")
                .short('m')
                .long("mapping")
                .action(ArgAction::Set)
                .value_name("MAPPING_FILE")
                .value_parser(clap::value_parser!(PathBuf))
                .help("Output file for crash-to-cluster mapping (JSON). Defaults to OUTPUT_DIR/mapping.json"),
        )
        .arg(
            Arg::new("timeout")
                .short('t')
                .long("timeout")
                .action(ArgAction::Set)
                .default_value("30")
                .value_name("SECONDS")
                .value_parser(clap::value_parser!(u64))
                .help("Timeout (in seconds) for target execution"),
        )
        .arg(
            Arg::new("jobs")
                .short('j')
                .long("jobs")
                .action(ArgAction::Set)
                .value_name("N")
                .value_parser(clap::value_parser!(u32).range(1..))
                .help("Number of parallel jobs [default: half of CPU cores]"),
        )
        .arg(
            Arg::new("tool")
                .long("tool")
                .action(ArgAction::Set)
                .value_name("TOOL")
                .value_parser(["casr-san", "casr-gdb", "casr-python", "casr-java", "casr-js", "casr-lua", "casr-csharp"])
                .help("Force specific CASR tool (auto-detect if not specified)"),
        )
        .arg(
            Arg::new("fuzzer")
                .long("fuzzer")
                .action(ArgAction::Set)
                .value_name("FUZZER")
                .value_parser(["auto", "libfuzzer", "afl"])
                .default_value("auto")
                .help("Fuzzer type: 'libfuzzer' for libFuzzer/LibAFL, 'afl' for AFL++, 'auto' to detect"),
        )
        .arg(
            Arg::new("ignore")
                .long("ignore")
                .action(ArgAction::Set)
                .value_parser(clap::value_parser!(PathBuf))
                .value_name("FILE")
                .help("File with regexes for functions/paths to ignore in stacktraces"),
        )
        .arg(
            Arg::new("use-logs")
                .long("use-logs")
                .action(ArgAction::SetTrue)
                .help("Use crash log files instead of re-running binaries. \
                       Expects logs in {input}/logs/{crash_name}.log. \
                       Only for libFuzzer with log-saving support."),
        )
        .arg(
            Arg::new("fuzzer-log")
                .long("fuzzer-log")
                .action(ArgAction::Set)
                .value_name("LOG_FILE")
                .value_parser(clap::value_parser!(PathBuf))
                .help("Parse ALL stacktraces directly from the main fuzzer log file. \
                       This extracts all unique ASAN stacktraces from the log without \
                       needing per-crash logs or re-running binaries. Use this when \
                       per-crash logs are not available (e.g., fork mode with ignore_crashes)."),
        )
        .arg(
            Arg::new("ARGS")
                .action(ArgAction::Set)
                .num_args(1..)
                .last(true)
                .required(false)
                .help("Target binary and arguments. Use @@ as placeholder for crash input. \
                       For AFL++, can be omitted to read from cmdline file."),
        )
        .get_matches();

    // Initialize stacktrace filters
    init_ignored_frames!("cpp", "rust", "python", "go", "java", "js", "csharp");

    // Parse arguments
    let input_dir = matches.get_one::<PathBuf>("input").unwrap();
    let output_dir = matches.get_one::<PathBuf>("output").unwrap();
    let timeout = *matches.get_one::<u64>("timeout").unwrap();

    let mapping_file = matches
        .get_one::<PathBuf>("mapping")
        .cloned()
        .unwrap_or_else(|| output_dir.join("mapping.json"));

    let jobs = matches
        .get_one::<u32>("jobs")
        .map(|j| *j as usize)
        .unwrap_or_else(|| std::cmp::max(1, num_cpus::get() / 2));

    // Determine fuzzer type
    let fuzzer_type_str = matches.get_one::<String>("fuzzer").unwrap();
    let fuzzer_type = match fuzzer_type_str.as_str() {
        "libfuzzer" => FuzzerType::LibFuzzer,
        "afl" => FuzzerType::Afl,
        _ => detect_fuzzer_type(input_dir),
    };
    eprintln!("Fuzzer type: {:?}", fuzzer_type);

    // Get binary args from command line or AFL cmdline file
    let binary_args: Vec<String> = if let Some(args) = matches.get_many::<String>("ARGS") {
        args.map(|s| s.to_string()).collect()
    } else if fuzzer_type == FuzzerType::Afl {
        // Try to read from AFL cmdline file
        match read_afl_cmdline(input_dir) {
            Some(args) => {
                eprintln!("Read command line from AFL cmdline file");
                args
            }
            None => {
                bail!("No ARGS provided and couldn't read cmdline file from AFL directory");
            }
        }
    } else {
        bail!("No target binary specified (use -- ./binary @@)");
    };

    if binary_args.is_empty() {
        bail!("No target binary specified");
    }

    // Load custom ignore patterns
    if let Some(ignore_path) = matches.get_one::<PathBuf>("ignore") {
        util::add_custom_ignored_frames(ignore_path)?;
    }

    // Check for fuzzer-log mode (parse all stacktraces from main log)
    let fuzzer_log_path = matches.get_one::<PathBuf>("fuzzer-log");

    if let Some(log_path) = fuzzer_log_path {
        eprintln!("Mode: fuzzer-log (extracting all stacktraces from main log)");
        return process_fuzzer_log(
            log_path,
            input_dir,
            output_dir,
            &mapping_file,
            fuzzer_type,
        );
    }

    // Check for log-based mode (no binary re-runs)
    let use_logs = matches.get_flag("use-logs");
    let logs_dir = input_dir.join("logs");

    if use_logs {
        eprintln!("Mode: log-based (no binary re-runs)");
        if !logs_dir.exists() {
            bail!(
                "Logs directory not found at {}. Was libFuzzer built with log-saving support?",
                logs_dir.display()
            );
        }
    }

    // Detect or use specified tool (only needed for binary-based mode)
    let tool = if use_logs {
        // Dummy path - not used in log mode
        PathBuf::new()
    } else if let Some(tool_name) = matches.get_one::<String>("tool") {
        util::get_path(tool_name)?
    } else {
        detect_tool(Path::new(&binary_args[0]))?
    };

    if !use_logs {
        eprintln!("Using tool: {}", tool.display());
    }

    // Create output directories
    let reports_dir = output_dir.join("reports");
    let clusters_dir = output_dir.join("clusters");
    fs::create_dir_all(&reports_dir)?;
    fs::create_dir_all(&clusters_dir)?;

    // Get all crash files based on fuzzer type
    let all_crash_files: Vec<PathBuf> = match fuzzer_type {
        FuzzerType::LibFuzzer => collect_libfuzzer_crashes(input_dir),
        FuzzerType::Afl => collect_afl_crashes(input_dir),
    };

    let total_crashes = all_crash_files.len();
    eprintln!("Found {} crash files", total_crashes);

    if total_crashes == 0 {
        let hint = match fuzzer_type {
            FuzzerType::LibFuzzer => "Expected crash-*, leak-*, timeout-*, or oom-* files in the input directory",
            FuzzerType::Afl => "Expected AFL++ directory structure with crashes/id* files",
        };
        bail!("No crash files found in {}. {}", input_dir.display(), hint);
    }

    // Separate crash files by type:
    // - Crashes with stacktraces (crash-*, leak-*) go through normal CASR processing
    // - Crashes without stacktraces (timeout-*, oom-*, slow-unit-*) get pseudo-clusters
    let mut stacktrace_crashes: Vec<PathBuf> = Vec::new();
    let mut pseudo_cluster_crashes: HashMap<CrashType, Vec<PathBuf>> = HashMap::new();

    for crash_path in &all_crash_files {
        let name = crash_path.file_name().unwrap().to_str().unwrap();
        let crash_type = CrashType::from_filename(name);

        if crash_type.has_stacktrace() {
            stacktrace_crashes.push(crash_path.clone());
        } else {
            pseudo_cluster_crashes
                .entry(crash_type)
                .or_default()
                .push(crash_path.clone());
        }
    }

    eprintln!(
        "  {} crashes/leaks (with stacktraces)",
        stacktrace_crashes.len()
    );
    for (crash_type, crashes) in &pseudo_cluster_crashes {
        eprintln!(
            "  {} {} (pseudo-cluster)",
            crashes.len(),
            crash_type.pseudo_cluster_name()
        );
    }

    // Step 1: Generate reports for crashes WITH stacktraces in parallel
    eprintln!("Generating reports with {} jobs...", jobs);

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(jobs)
        .build()
        .unwrap();

    let successful_reports: RwLock<Vec<(PathBuf, PathBuf)>> = RwLock::new(Vec::new()); // (crash_path, report_path)
    let failed: RwLock<Vec<(PathBuf, String)>> = RwLock::new(Vec::new());

    // Only process crashes that have stacktraces
    pool.install(|| {
        stacktrace_crashes.par_iter().for_each(|crash_path| {
            let result = if use_logs {
                // Log-based mode: parse stacktrace from log file (no binary re-run)
                report_from_log(crash_path, &logs_dir, &reports_dir)
            } else {
                // Binary-based mode: re-run binary to generate crash report
                generate_report(&tool, crash_path, &reports_dir, &binary_args, timeout)
            };

            match result {
                Ok(report_path) => {
                    successful_reports
                        .write()
                        .unwrap()
                        .push((crash_path.clone(), report_path));
                }
                Err(e) => {
                    // In log mode, warn about missing/invalid logs but don't fail
                    if use_logs {
                        eprintln!(
                            "WARNING: Skipping {} - {}",
                            crash_path.file_name().unwrap().to_str().unwrap(),
                            e
                        );
                    }
                    failed
                        .write()
                        .unwrap()
                        .push((crash_path.clone(), e.to_string()));
                }
            }
        });
    });

    let successful_reports = successful_reports.into_inner().unwrap();
    let failed = failed.into_inner().unwrap();

    eprintln!(
        "Generated {} reports, {} failed",
        successful_reports.len(),
        failed.len()
    );

    if !failed.is_empty() {
        eprintln!("Failed crashes:");
        for (path, err) in failed.iter().take(5) {
            eprintln!("  {}: {}", path.display(), err);
        }
        if failed.len() > 5 {
            eprintln!("  ... and {} more", failed.len() - 5);
        }
    }

    if successful_reports.is_empty() {
        bail!("No reports generated successfully");
    }

    // Step 2: Parse stacktraces from reports
    eprintln!("Parsing stacktraces...");
    let mut reports_with_stacktraces: Vec<(PathBuf, PathBuf, Stacktrace)> = Vec::new(); // (crash, report, stacktrace)

    for (crash_path, report_path) in &successful_reports {
        match get_stacktrace(report_path) {
            Ok(st) => {
                reports_with_stacktraces.push((crash_path.clone(), report_path.clone(), st));
            }
            Err(e) => {
                eprintln!("Warning: couldn't parse {}: {}", report_path.display(), e);
            }
        }
    }

    eprintln!("Parsed {} stacktraces", reports_with_stacktraces.len());

    if reports_with_stacktraces.len() < 2 {
        eprintln!("Less than 2 valid stacktrace reports, creating single stacktrace cluster");
        
        let mut mappings: Vec<CrashMapping> = reports_with_stacktraces
            .iter()
            .map(|(crash, _, _)| CrashMapping {
                crash: crash.file_name().unwrap().to_str().unwrap().to_string(),
                cluster_id: 1,
                is_representative: true,
                representative: None,
            })
            .collect();
        
        let mut cluster_to_crashes: HashMap<usize, Vec<String>> = HashMap::new();
        if !reports_with_stacktraces.is_empty() {
            cluster_to_crashes.insert(
                1,
                reports_with_stacktraces
                    .iter()
                    .map(|(c, _, _)| c.file_name().unwrap().to_str().unwrap().to_string())
                    .collect(),
            );
        }
        
        // Add pseudo-clusters for timeout, oom, etc.
        let base_clusters = if reports_with_stacktraces.is_empty() { 0 } else { 1 };
        let mut next_cluster_id = base_clusters + 1;
        let mut pseudo_cluster_count = 0;
        
        for (crash_type, crashes) in &pseudo_cluster_crashes {
            if crashes.is_empty() {
                continue;
            }
            
            let cluster_id = next_cluster_id;
            next_cluster_id += 1;
            pseudo_cluster_count += 1;
            
            eprintln!(
                "Adding pseudo-cluster {} for {} {} events",
                cluster_id,
                crashes.len(),
                crash_type.pseudo_cluster_name()
            );
            
            let mut is_first = true;
            let mut representative_name: Option<String> = None;
            
            for crash_path in crashes {
                let crash_name = crash_path.file_name().unwrap().to_str().unwrap().to_string();
                
                if is_first {
                    representative_name = Some(crash_name.clone());
                    mappings.push(CrashMapping {
                        crash: crash_name.clone(),
                        cluster_id,
                        is_representative: true,
                        representative: None,
                    });
                    is_first = false;
                } else {
                    mappings.push(CrashMapping {
                        crash: crash_name.clone(),
                        cluster_id,
                        is_representative: false,
                        representative: representative_name.clone(),
                    });
                }
                
                cluster_to_crashes
                    .entry(cluster_id)
                    .or_default()
                    .push(crash_name);
            }
        }
        
        mappings.sort_by(|a, b| a.crash.cmp(&b.crash));
        
        let final_num_clusters = base_clusters + pseudo_cluster_count;
        
        let result = ClusterMapResult {
            total_crashes,
            unique_stacktraces: reports_with_stacktraces.len(),
            num_clusters: final_num_clusters,
            mappings,
            clusters: cluster_to_crashes,
        };

        let json = serde_json::to_string_pretty(&result)?;
        fs::write(&mapping_file, &json)?;
        eprintln!("Mapping written to {}", mapping_file.display());
        println!("{}", json);
        
        eprintln!("\n=== Summary ===");
        eprintln!("Total crashes: {}", result.total_crashes);
        eprintln!("Unique stacktraces: {}", result.unique_stacktraces);
        eprintln!("Clusters: {} ({} from stacktraces, {} pseudo-clusters)", 
                  final_num_clusters, base_clusters, pseudo_cluster_count);
        
        return Ok(());
    }

    // Step 3: Group by identical stacktrace (deduplication)
    eprintln!("Grouping by stacktrace...");
    let reports_for_grouping: Vec<(PathBuf, Stacktrace)> = reports_with_stacktraces
        .iter()
        .map(|(_, report, st)| (report.clone(), st.clone()))
        .collect();

    let groups = group_by_stacktrace(&reports_for_grouping);
    let unique_stacktraces = groups.len();
    eprintln!(
        "Found {} unique stacktraces from {} reports",
        unique_stacktraces,
        reports_with_stacktraces.len()
    );

    // Handle case where all crashes have the same stacktrace (1 unique)
    if unique_stacktraces < 2 {
        eprintln!("Only {} unique stacktrace(s), creating single stacktrace cluster", unique_stacktraces);

        // All stacktrace crashes go to cluster 1
        let mut mappings: Vec<CrashMapping> = Vec::new();
        let mut cluster_to_crashes: HashMap<usize, Vec<String>> = HashMap::new();
        let mut is_first = true;
        let mut representative_name: Option<String> = None;

        for (crash_path, _, _) in &reports_with_stacktraces {
            let crash_name = crash_path.file_name().unwrap().to_str().unwrap().to_string();

            if is_first {
                representative_name = Some(crash_name.clone());
                mappings.push(CrashMapping {
                    crash: crash_name.clone(),
                    cluster_id: 1,
                    is_representative: true,
                    representative: None,
                });
                is_first = false;
            } else {
                mappings.push(CrashMapping {
                    crash: crash_name.clone(),
                    cluster_id: 1,
                    is_representative: false,
                    representative: representative_name.clone(),
                });
            }
            
            cluster_to_crashes
                .entry(1)
                .or_default()
                .push(crash_name);
        }

        // Add pseudo-clusters for timeout, oom, etc.
        let base_clusters = unique_stacktraces;
        let mut next_cluster_id = base_clusters + 1;
        let mut pseudo_cluster_count = 0;

        for (crash_type, crashes) in &pseudo_cluster_crashes {
            if crashes.is_empty() {
                continue;
            }

            let cluster_id = next_cluster_id;
            next_cluster_id += 1;
            pseudo_cluster_count += 1;

            eprintln!(
                "Adding pseudo-cluster {} for {} {} events",
                cluster_id,
                crashes.len(),
                crash_type.pseudo_cluster_name()
            );

            let mut is_first = true;
            let mut representative_name: Option<String> = None;

            for crash_path in crashes {
                let crash_name = crash_path.file_name().unwrap().to_str().unwrap().to_string();

                if is_first {
                    representative_name = Some(crash_name.clone());
                    mappings.push(CrashMapping {
                        crash: crash_name.clone(),
                        cluster_id,
                        is_representative: true,
                        representative: None,
                    });
                    is_first = false;
                } else {
                    mappings.push(CrashMapping {
                        crash: crash_name.clone(),
                        cluster_id,
                        is_representative: false,
                        representative: representative_name.clone(),
                    });
                }

                cluster_to_crashes
                    .entry(cluster_id)
                    .or_default()
                    .push(crash_name);
            }
        }

        let final_num_clusters = base_clusters + pseudo_cluster_count;

        // Sort mappings by crash name for consistent output
        mappings.sort_by(|a, b| a.crash.cmp(&b.crash));

        let result = ClusterMapResult {
            total_crashes,
            unique_stacktraces,
            num_clusters: final_num_clusters,
            mappings,
            clusters: cluster_to_crashes,
        };

        let json = serde_json::to_string_pretty(&result)?;
        fs::write(&mapping_file, &json)?;
        eprintln!("Mapping written to {}", mapping_file.display());
        println!("{}", json);

        eprintln!("\n=== Summary ===");
        eprintln!("Total crashes: {}", result.total_crashes);
        eprintln!("Unique stacktraces: {}", result.unique_stacktraces);
        eprintln!("Clusters: {} ({} from stacktraces, {} pseudo-clusters)", 
                  final_num_clusters, base_clusters, pseudo_cluster_count);

        return Ok(());
    }

    // Step 4: Copy only representative reports to a temp dir for clustering
    let unique_reports_dir = output_dir.join("unique_reports");
    fs::create_dir_all(&unique_reports_dir)?;

    let representative_indices: Vec<usize> = groups.keys().copied().collect();
    for &rep_idx in &representative_indices {
        let (_, report_path, _) = &reports_with_stacktraces[rep_idx];
        let dest = unique_reports_dir.join(report_path.file_name().unwrap());
        fs::copy(report_path, dest)?;
    }

    // Step 5: Cluster the unique representatives
    eprintln!("Clustering {} unique reports...", unique_stacktraces);

    let casreps = util::get_reports(&unique_reports_dir)?;
    let (casreps_info, _bad) = util::reports_from_paths(&casreps, jobs);

    if casreps_info.len() < 2 {
        bail!("Not enough valid reports for clustering");
    }

    let (clusters, _, _) = Cluster::cluster_reports(&casreps_info, 0, false)?;

    // Build representative report path -> cluster ID mapping
    let mut rep_report_to_cluster: HashMap<String, usize> = HashMap::new();
    for cluster in clusters.values() {
        for path in cluster.paths() {
            let filename = path.file_name().unwrap().to_str().unwrap().to_string();
            rep_report_to_cluster.insert(filename, cluster.number);
        }
    }

    // Save clusters
    util::save_clusters(&clusters, &clusters_dir)?;

    let num_clusters = clusters.len();
    eprintln!("Created {} clusters", num_clusters);

    // Step 6: Build full mapping (all crashes -> cluster via representative)
    eprintln!("Building full mapping...");

    let mut mappings: Vec<CrashMapping> = Vec::new();
    let mut cluster_to_crashes: HashMap<usize, Vec<String>> = HashMap::new();

    for (&rep_idx, members) in &groups {
        let (_, rep_report_path, _) = &reports_with_stacktraces[rep_idx];
        let rep_report_name = rep_report_path.file_name().unwrap().to_str().unwrap();
        let rep_crash_name = reports_with_stacktraces[rep_idx]
            .0
            .file_name()
            .unwrap()
            .to_str()
            .unwrap()
            .to_string();

        // Get cluster ID for this representative
        let cluster_id = *rep_report_to_cluster
            .get(rep_report_name)
            .unwrap_or(&1);

        for &member_idx in members {
            let crash_name = reports_with_stacktraces[member_idx]
                .0
                .file_name()
                .unwrap()
                .to_str()
                .unwrap()
                .to_string();

            let is_representative = member_idx == rep_idx;

            mappings.push(CrashMapping {
                crash: crash_name.clone(),
                cluster_id,
                is_representative,
                representative: if is_representative {
                    None
                } else {
                    Some(rep_crash_name.clone())
                },
            });

            cluster_to_crashes
                .entry(cluster_id)
                .or_default()
                .push(crash_name);
        }
    }

    // Step 7: Add pseudo-clusters for timeout, oom, slow-unit crashes
    let mut next_cluster_id = num_clusters + 1;
    let mut pseudo_cluster_count = 0;

    for (crash_type, crashes) in &pseudo_cluster_crashes {
        if crashes.is_empty() {
            continue;
        }

        let cluster_id = next_cluster_id;
        next_cluster_id += 1;
        pseudo_cluster_count += 1;

        eprintln!(
            "Adding pseudo-cluster {} for {} {} events",
            cluster_id,
            crashes.len(),
            crash_type.pseudo_cluster_name()
        );

        let mut is_first = true;
        let mut representative_name: Option<String> = None;

        for crash_path in crashes {
            let crash_name = crash_path.file_name().unwrap().to_str().unwrap().to_string();

            if is_first {
                representative_name = Some(crash_name.clone());
                mappings.push(CrashMapping {
                    crash: crash_name.clone(),
                    cluster_id,
                    is_representative: true,
                    representative: None,
                });
                is_first = false;
            } else {
                mappings.push(CrashMapping {
                    crash: crash_name.clone(),
                    cluster_id,
                    is_representative: false,
                    representative: representative_name.clone(),
                });
            }

            cluster_to_crashes
                .entry(cluster_id)
                .or_default()
                .push(crash_name);
        }
    }

    let final_num_clusters = num_clusters + pseudo_cluster_count;

    // Sort mappings by crash name for consistent output
    mappings.sort_by(|a, b| a.crash.cmp(&b.crash));

    let result = ClusterMapResult {
        total_crashes,
        unique_stacktraces,
        num_clusters: final_num_clusters,
        mappings,
        clusters: cluster_to_crashes,
    };

    // Output results
    let json = serde_json::to_string_pretty(&result)?;
    fs::write(&mapping_file, &json)?;
    eprintln!("Mapping written to {}", mapping_file.display());

    // Print summary
    println!("{}", json);

    eprintln!("\n=== Summary ===");
    eprintln!("Total crashes: {}", result.total_crashes);
    eprintln!("Unique stacktraces: {}", result.unique_stacktraces);
    eprintln!("Clusters: {} ({} from stacktraces, {} pseudo-clusters)", 
              final_num_clusters, num_clusters, pseudo_cluster_count);
    for (cluster_id, crashes) in result.clusters.iter() {
        eprintln!("  Cluster {}: {} crashes", cluster_id, crashes.len());
    }

    Ok(())
}
