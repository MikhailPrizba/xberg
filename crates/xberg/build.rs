// Inference capabilities are intentionally absent from the compact fork.
fn main() {
    println!("cargo::rustc-check-cfg=cfg(coverage)");
    if let Ok(manifest_dir) = std::env::var("CARGO_MANIFEST_DIR") {
        emit_rerun_directives(&manifest_dir);
    }
    println!("cargo::rustc-env=XBERG_BUILD_ID={}", build_id());
}

/// Best-effort identifier for *this* build, embedded at compile time via `env!("XBERG_BUILD_ID")`.
///
/// Exists so two separately built binaries at the same crate version and cache schema version
/// get different on-disk cache entries instead of silently serving each other's results (a real
/// defect: see the module doc on `cache::version`). Resolved in this order:
///
/// 1. The `XBERG_BUILD_ID` environment variable, if set to a non-empty value. This is the
///    escape hatch for the two cases the git-based derivation below cannot reach on its own:
///    a Docker build (`.dockerignore` excludes `.git` from every build context in this repo, so
///    `git` has nothing to read there) and a crates.io/docs.rs build from a published source
///    tarball (no `.git` at all). A release pipeline can pass `--build-arg XBERG_BUILD_ID=<sha>`
///    / set the env var to the commit it packaged; a developer running two arms of an A/B
///    measurement from the same dirty worktree can export distinct values by hand to force
///    distinct cache entries instead of remembering to wipe the cache between arms.
/// 2. The current commit SHA (`git rev-parse HEAD`) plus, when the working tree has uncommitted
///    changes, a hash of the tracked diff against `HEAD` and the list of untracked paths. This
///    covers ordinary local development and any CI job that runs the build from a real git
///    checkout: two builds from different commits, or from the same commit with different
///    uncommitted edits, get different ids. Requires `git` on `PATH` and `CARGO_MANIFEST_DIR`
///    to sit inside a git checkout; any failure (missing binary, not a repository, non-UTF-8
///    output) falls through to the next case rather than failing the build.
/// 3. `""` when neither is available. `cache_version_tag` (crates/xberg/src/cache/version.rs)
///    then folds in an empty build id, collapsing this component of the tag back to what it
///    always was before `XBERG_BUILD_ID` existed. For a build from a published crates.io/docs.rs
///    tarball this is not a gap, it is correct: crates.io tarballs are immutable per version, so
///    two such builds of the same version share identical source and there is no "build" left
///    for an id to distinguish.
///
/// Cargo does NOT track environment variables for a build script unless the script says so, and
/// emitting no `rerun-if-*` at all only gets the default "rerun when a file in the package
/// changed" heuristic -- which never notices `XBERG_BUILD_ID`. Measured: two builds with
/// different `XBERG_BUILD_ID` values produced the identical tag `2d0ec67f`, i.e. the override was
/// inert. `emit_rerun_directives` below therefore declares the env var and the git refs
/// explicitly. Note that emitting any `rerun-if-*` instruction REPLACES the default heuristic,
/// so the source tree has to be declared alongside them or an ordinary code edit stops
/// re-running the script.
///
/// Never touches the network and never fails the build; worst case it returns `""`.
/// Declare everything `build_id` reads, because emitting any `rerun-if-*` instruction turns off
/// Cargo's default "any file in the package changed" heuristic.
///
/// `src` and `Cargo.toml` restore that default. `XBERG_BUILD_ID` is the override. `.git/HEAD` and
/// `.git/index` cover committing, switching branch and staging; an unstaged edit is already
/// covered because it changes a file under `src`.
fn emit_rerun_directives(manifest_dir: &str) {
    println!("cargo::rerun-if-env-changed=XBERG_BUILD_ID");
    println!("cargo::rerun-if-changed={manifest_dir}/src");
    println!("cargo::rerun-if-changed={manifest_dir}/Cargo.toml");

    let Some(git_dir) = git_stdout(manifest_dir, &["rev-parse", "--git-dir"]) else {
        return;
    };
    let git_dir = git_dir.trim();
    if git_dir.is_empty() {
        return;
    }
    let base = if std::path::Path::new(git_dir).is_absolute() {
        git_dir.to_string()
    } else {
        format!("{manifest_dir}/{git_dir}")
    };
    println!("cargo::rerun-if-changed={base}/HEAD");
    println!("cargo::rerun-if-changed={base}/index");
}

fn build_id() -> String {
    if let Ok(id) = std::env::var("XBERG_BUILD_ID")
        && !id.is_empty()
    {
        return id;
    }

    let Ok(manifest_dir) = std::env::var("CARGO_MANIFEST_DIR") else {
        return String::new();
    };

    let Some(sha) = git_stdout(&manifest_dir, &["rev-parse", "HEAD"]) else {
        return String::new();
    };
    let sha = sha.trim();

    // Tracked, uncommitted changes (staged and unstaged) against HEAD.
    let diff = git_stdout(&manifest_dir, &["diff", "HEAD"]).unwrap_or_default();
    // New files git doesn't know about yet: a diff against HEAD can't see these, but an A/B
    // measurement that adds a file without committing it needs a distinct id all the same.
    let untracked = git_stdout(&manifest_dir, &["ls-files", "--others", "--exclude-standard"]).unwrap_or_default();

    if diff.is_empty() && untracked.is_empty() {
        return sha.to_string();
    }

    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    diff.hash(&mut hasher);
    untracked.hash(&mut hasher);
    format!("{sha}-dirty-{:016x}", hasher.finish())
}

/// Run `git <args>` in `dir`, returning its stdout as UTF-8 on success.
///
/// Returns `None` on any failure -- `git` missing, `dir` not inside a repository, a non-zero
/// exit, or non-UTF-8 output -- so callers can fall through to the next id source.
fn git_stdout(dir: &str, args: &[&str]) -> Option<String> {
    let output = std::process::Command::new("git")
        .args(args)
        .current_dir(dir)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8(output.stdout).ok()
}
