//! Tool for clustering crashes with full crash-to-cluster mapping.
//!
//! Unlike casr-libfuzzer + casr-cluster pipeline, this tool preserves
//! the mapping of ALL original crashes to their cluster IDs, even if
//! they were deduplicated due to identical stacktraces.

use casr::util;
use libcasr::cluster::Cluster;
use libcasr::stacktrace::{dedup_stacktraces, Filter, Stacktrace};
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
                .value_parser(["casr-san", "casr-gdb", "casr-python", "casr-java", "casr-js", "casr-lua"])
                .help("Force specific CASR tool (auto-detect if not specified)"),
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
            Arg::new("ARGS")
                .action(ArgAction::Set)
                .num_args(1..)
                .last(true)
                .required(true)
                .help("Target binary and arguments. Use @@ as placeholder for crash input."),
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

    let binary_args: Vec<String> = matches
        .get_many::<String>("ARGS")
        .unwrap()
        .map(|s| s.to_string())
        .collect();

    if binary_args.is_empty() {
        bail!("No target binary specified");
    }

    // Load custom ignore patterns
    if let Some(ignore_path) = matches.get_one::<PathBuf>("ignore") {
        util::add_custom_ignored_frames(ignore_path)?;
    }

    // Detect or use specified tool
    let tool = if let Some(tool_name) = matches.get_one::<String>("tool") {
        util::get_path(tool_name)?
    } else {
        detect_tool(Path::new(&binary_args[0]))?
    };
    eprintln!("Using tool: {}", tool.display());

    // Create output directories
    let reports_dir = output_dir.join("reports");
    let clusters_dir = output_dir.join("clusters");
    fs::create_dir_all(&reports_dir)?;
    fs::create_dir_all(&clusters_dir)?;

    // Get all crash files
    let crash_files: Vec<PathBuf> = fs::read_dir(input_dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_file() && !p.file_name().unwrap().to_str().unwrap().starts_with('.'))
        .collect();

    let total_crashes = crash_files.len();
    eprintln!("Found {} crash files", total_crashes);

    if total_crashes == 0 {
        bail!("No crash files found in {}", input_dir.display());
    }

    // Step 1: Generate reports for all crashes in parallel
    eprintln!("Generating reports with {} jobs...", jobs);

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(jobs)
        .build()
        .unwrap();

    let successful_reports: RwLock<Vec<(PathBuf, PathBuf)>> = RwLock::new(Vec::new()); // (crash_path, report_path)
    let failed: RwLock<Vec<(PathBuf, String)>> = RwLock::new(Vec::new());

    pool.install(|| {
        crash_files.par_iter().for_each(|crash_path| {
            match generate_report(&tool, crash_path, &reports_dir, &binary_args, timeout) {
                Ok(report_path) => {
                    successful_reports
                        .write()
                        .unwrap()
                        .push((crash_path.clone(), report_path));
                }
                Err(e) => {
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
        eprintln!("Less than 2 valid reports, creating single cluster");
        let result = ClusterMapResult {
            total_crashes,
            unique_stacktraces: reports_with_stacktraces.len(),
            num_clusters: 1,
            mappings: reports_with_stacktraces
                .iter()
                .map(|(crash, _, _)| CrashMapping {
                    crash: crash.file_name().unwrap().to_str().unwrap().to_string(),
                    cluster_id: 1,
                    is_representative: true,
                    representative: None,
                })
                .collect(),
            clusters: [(
                1,
                reports_with_stacktraces
                    .iter()
                    .map(|(c, _, _)| c.file_name().unwrap().to_str().unwrap().to_string())
                    .collect(),
            )]
            .into(),
        };

        let json = serde_json::to_string_pretty(&result)?;
        fs::write(&mapping_file, &json)?;
        println!("{}", json);
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
        eprintln!("Only {} unique stacktrace(s), creating single cluster", unique_stacktraces);

        // All crashes go to cluster 1
        let mut mappings: Vec<CrashMapping> = Vec::new();
        let mut is_first = true;
        let mut representative_name: Option<String> = None;

        for (crash_path, _, _) in &reports_with_stacktraces {
            let crash_name = crash_path.file_name().unwrap().to_str().unwrap().to_string();

            if is_first {
                representative_name = Some(crash_name.clone());
                mappings.push(CrashMapping {
                    crash: crash_name,
                    cluster_id: 1,
                    is_representative: true,
                    representative: None,
                });
                is_first = false;
            } else {
                mappings.push(CrashMapping {
                    crash: crash_name,
                    cluster_id: 1,
                    is_representative: false,
                    representative: representative_name.clone(),
                });
            }
        }

        // Sort mappings by crash name for consistent output
        mappings.sort_by(|a, b| a.crash.cmp(&b.crash));

        let result = ClusterMapResult {
            total_crashes,
            unique_stacktraces,
            num_clusters: 1,
            mappings: mappings.clone(),
            clusters: [(
                1,
                mappings.iter().map(|m| m.crash.clone()).collect(),
            )]
            .into(),
        };

        let json = serde_json::to_string_pretty(&result)?;
        fs::write(&mapping_file, &json)?;
        eprintln!("Mapping written to {}", mapping_file.display());
        println!("{}", json);

        eprintln!("\n=== Summary ===");
        eprintln!("Total crashes: {}", result.total_crashes);
        eprintln!("Unique stacktraces: {}", result.unique_stacktraces);
        eprintln!("Clusters: {}", result.num_clusters);
        eprintln!("  Cluster 1: {} crashes", result.mappings.len());

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

    // Sort mappings by crash name for consistent output
    mappings.sort_by(|a, b| a.crash.cmp(&b.crash));

    let result = ClusterMapResult {
        total_crashes,
        unique_stacktraces,
        num_clusters,
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
    eprintln!("Clusters: {}", result.num_clusters);
    for (cluster_id, crashes) in result.clusters.iter() {
        eprintln!("  Cluster {}: {} crashes", cluster_id, crashes.len());
    }

    Ok(())
}
