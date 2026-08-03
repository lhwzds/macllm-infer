use std::{
    env,
    path::{Path, PathBuf},
    process::{Command, Output},
};

const UNKNOWN: &str = "unknown";

fn main() {
    let repository =
        env::var_os("CARGO_MANIFEST_DIR").map_or_else(|| PathBuf::from("."), PathBuf::from);

    let profile = env::var("PROFILE").unwrap_or_else(|_| UNKNOWN.to_owned());
    let opt_level = env::var("OPT_LEVEL").unwrap_or_else(|_| UNKNOWN.to_owned());
    println!("cargo:rustc-env=MACLLM_BUILD_PROFILE={profile}");
    println!("cargo:rustc-env=MACLLM_BUILD_OPT_LEVEL={opt_level}");

    emit_git_inputs(&repository);

    let (commit, dirty) = collect_git_identity(&repository).map_or_else(
        || (UNKNOWN.to_owned(), UNKNOWN.to_owned()),
        |identity| (identity.commit, identity.dirty.to_string()),
    );
    println!("cargo:rustc-env=MACLLM_BUILD_GIT_COMMIT={commit}");
    println!("cargo:rustc-env=MACLLM_BUILD_GIT_DIRTY={dirty}");
}

#[derive(Debug, PartialEq, Eq)]
struct GitIdentity {
    commit: String,
    dirty: bool,
}

fn collect_git_identity(repository: &Path) -> Option<GitIdentity> {
    let first_commit = git_text(repository, &["rev-parse", "--verify", "HEAD"])?;
    let first_status = git_output(
        repository,
        &["status", "--porcelain=v1", "--untracked-files=all"],
    )?
    .stdout;
    let second_commit = git_text(repository, &["rev-parse", "--verify", "HEAD"])?;
    let second_status = git_output(
        repository,
        &["status", "--porcelain=v1", "--untracked-files=all"],
    )?
    .stdout;

    if first_commit != second_commit
        || first_status != second_status
        || !is_lower_hex(&first_commit, 40)
    {
        return None;
    }

    Some(GitIdentity {
        commit: first_commit,
        dirty: !first_status.is_empty(),
    })
}

fn emit_git_inputs(repository: &Path) {
    println!(
        "cargo:rerun-if-changed={}",
        repository.join("build.rs").display()
    );

    if let Some(output) = git_output(
        repository,
        &[
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
        ],
    ) {
        for relative in output.stdout.split(|byte| *byte == 0) {
            if relative.is_empty() {
                continue;
            }
            let relative = String::from_utf8_lossy(relative);
            if relative.contains('\n') || relative.contains('\r') {
                continue;
            }
            println!(
                "cargo:rerun-if-changed={}",
                repository.join(relative.as_ref()).display()
            );
        }
    }

    // Commit and staging changes can leave every worktree file timestamp
    // untouched. These Git metadata paths make such identity changes rebuild
    // the embedded values as well.
    for git_path in ["HEAD", "index", "logs/HEAD"] {
        if let Some(path) = git_text(repository, &["rev-parse", "--git-path", git_path]) {
            let path = PathBuf::from(path);
            let path = if path.is_absolute() {
                path
            } else {
                repository.join(path)
            };
            println!("cargo:rerun-if-changed={}", path.display());
        }
    }
}

fn git_text(repository: &Path, arguments: &[&str]) -> Option<String> {
    let output = git_output(repository, arguments)?;
    let value = std::str::from_utf8(&output.stdout).ok()?.trim();
    (!value.is_empty()).then(|| value.to_owned())
}

fn git_output(repository: &Path, arguments: &[&str]) -> Option<Output> {
    let output = Command::new("git")
        .arg("-C")
        .arg(repository)
        .args(arguments)
        .output()
        .ok()?;
    output.status.success().then_some(output)
}

fn is_lower_hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}
