//! Module layering guard.
//!
//! The crate is being split into a Cargo workspace. The chat group now lives
//! in the `macllm-chat` crate, so Cargo itself enforces the boundary between
//! the chat contracts and the tensor runtime: the guard only has to check
//! that the crate manifest lists no forbidden dependency.
//!
//! The remaining rules are still a simple scan of the source text:
//!
//! * it removes the `#[cfg(test)]` regions, because test-only code may reach
//!   into modules that the production code must not depend on. The whole
//!   annotated item is removed, whether the item ends with a semicolon
//!   (`use ...;`, `mod helpers;`) or with its closing brace (`fn ... { ... }`,
//!   `mod tests { ... }`), so that production code behind a test-only import
//!   is still scanned. Only a real attribute counts: a `#[cfg(test)]` that is
//!   written inside a comment or inside a string literal is copied verbatim
//!   and never removes the code behind it;
//! * it looks for the forbidden module paths both as `crate::<module>` and
//!   inside `use crate:: { ... }` groups, where the module name appears
//!   without the `crate::` prefix. The import paths are parsed instead of
//!   matched as plain text, so neither whitespace around the group brace nor
//!   a nested group hides a violation. A group entry may rename the imported
//!   item (`{models as adapter}`) or carry a comment
//!   (`{/* ... */ models::Foo}`); both are parsed, so neither hides the
//!   module that the entry names;
//! * it fails with the offending file and path so the violation is easy to
//!   locate.
//!
//! The rules are:
//!
//! * the chat crate (chat, tool calls, grammar, OpenAI wire types, agent
//!   prompts, output filters, the logits-constraint and finish-reason
//!   vocabulary and the chat templates) must not depend on the tensor
//!   runtime, the argument parser, the HTTP server or the root crate;
//! * the server must not reach into the model adapters;
//! * the benchmark harness must not reach into the server;
//! * the backend selectors must not depend on the argument parser.

use std::path::{Path, PathBuf};

/// The manifest of the crate that owns the chat group.
const CHAT_CRATE_MANIFEST: &str = "crates/macllm-chat/Cargo.toml";

/// Dependencies the chat crate must not list.
///
/// The chat group is protocol-independent, so it may not reach the tensor
/// runtime, the argument parser, the HTTP server or the root crate.
const CHAT_CRATE_FORBIDDEN: [&str; 6] = [
    "candle-core",
    "candle-nn",
    "axum",
    "tokio",
    "clap",
    "macllm-infer",
];

fn repository_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Whether the character may appear inside an identifier.
fn is_identifier_char(character: char) -> bool {
    character.is_alphanumeric() || character == '_'
}

/// Whether the source spells `keyword` at `cursor` as a whole word.
fn starts_with_keyword(source: &[char], cursor: usize, keyword: &str) -> bool {
    let keyword: Vec<char> = keyword.chars().collect();
    if cursor + keyword.len() > source.len() {
        return false;
    }
    if source[cursor..cursor + keyword.len()] != keyword[..] {
        return false;
    }
    if cursor > 0 && is_identifier_char(source[cursor - 1]) {
        return false;
    }
    let after = cursor + keyword.len();
    if after < source.len() && is_identifier_char(source[after]) {
        return false;
    }
    true
}

/// Skip whitespace, line comments and block comments.
fn skip_trivia(source: &[char], mut cursor: usize) -> usize {
    loop {
        while cursor < source.len() && source[cursor].is_whitespace() {
            cursor += 1;
        }
        if cursor + 1 < source.len() && source[cursor] == '/' && source[cursor + 1] == '/' {
            while cursor < source.len() && source[cursor] != '\n' {
                cursor += 1;
            }
            continue;
        }
        if cursor + 1 < source.len() && source[cursor] == '/' && source[cursor + 1] == '*' {
            cursor += 2;
            while cursor + 1 < source.len() && !(source[cursor] == '*' && source[cursor + 1] == '/')
            {
                cursor += 1;
            }
            cursor = (cursor + 2).min(source.len());
            continue;
        }
        return cursor;
    }
}

/// The end of the comment that starts at `cursor`, if there is one.
///
/// A line comment ends at the end of its line, a block comment at its
/// closing `*/`. Anything else (a division, for instance) is not a comment.
fn comment_end(source: &[char], cursor: usize) -> Option<usize> {
    if source.get(cursor) != Some(&'/') {
        return None;
    }
    match source.get(cursor + 1) {
        Some('/') => {
            let mut index = cursor + 2;
            while index < source.len() && source[index] != '\n' {
                index += 1;
            }
            Some(index)
        }
        Some('*') => {
            let mut index = cursor + 2;
            while index + 1 < source.len() && !(source[index] == '*' && source[index + 1] == '/') {
                index += 1;
            }
            Some((index + 2).min(source.len()))
        }
        _ => None,
    }
}

/// Skip a string literal that starts at `cursor`.
fn skip_string(source: &[char], cursor: usize) -> usize {
    let mut index = cursor + 1;
    while index < source.len() {
        match source[index] {
            '\\' => index += 2,
            '"' => return index + 1,
            _ => index += 1,
        }
    }
    source.len()
}

/// Skip a character literal that starts at `cursor`.
///
/// A lifetime such as `'a` only looks like a character literal, so it is
/// left alone when it is not closed right away.
fn skip_char_literal(source: &[char], cursor: usize) -> usize {
    if cursor + 2 < source.len() && source[cursor + 1] == '\\' {
        let mut index = cursor + 2;
        while index < source.len() && source[index] != '\'' {
            index += 1;
        }
        return (index + 1).min(source.len());
    }
    if cursor + 2 < source.len() && source[cursor + 2] == '\'' {
        return cursor + 3;
    }
    cursor + 1
}

/// Skip a raw string literal that starts at `cursor`.
///
/// A raw string keeps its backslashes, so it ends at the quote that is
/// followed by as many hashes as the opening delimiter had.
fn skip_raw_string(source: &[char], cursor: usize) -> Option<usize> {
    let mut index = cursor;
    if source.get(index) == Some(&'b') {
        index += 1;
    }
    if source.get(index) != Some(&'r') {
        return None;
    }
    index += 1;
    let hash_start = index;
    while source.get(index) == Some(&'#') {
        index += 1;
    }
    let hashes = index - hash_start;
    if source.get(index) != Some(&'"') {
        return None;
    }
    index += 1;
    while index < source.len() {
        if source[index] == '"' {
            let mut closing = index + 1;
            let mut found = 0usize;
            while found < hashes && source.get(closing) == Some(&'#') {
                closing += 1;
                found += 1;
            }
            if found == hashes {
                return Some(closing);
            }
        }
        index += 1;
    }
    None
}

/// Skip the literal that starts at `cursor`.
///
/// Strings, raw strings and character literals are skipped as a whole, so
/// that their content cannot be mistaken for code.
fn skip_literal(source: &[char], cursor: usize) -> usize {
    match source.get(cursor) {
        Some('"') => skip_string(source, cursor),
        Some('\'') => skip_char_literal(source, cursor),
        Some('r' | 'b') => skip_raw_string(source, cursor).unwrap_or(cursor + 1),
        _ => cursor + 1,
    }
}

/// Skip a bracketed group that starts at `cursor`.
///
/// Returns the index just after the matching closing bracket.
fn skip_bracketed(source: &[char], cursor: usize) -> Option<usize> {
    let open = *source.get(cursor)?;
    let close = match open {
        '(' => ')',
        '[' => ']',
        '{' => '}',
        _ => return None,
    };
    let mut depth = 0usize;
    let mut index = cursor;
    while index < source.len() {
        let character = source[index];
        match character {
            '"' | '\'' | 'r' | 'b' => {
                index = skip_literal(source, index);
                continue;
            }
            '/' if index + 1 < source.len() && source[index + 1] == '/' => {
                while index < source.len() && source[index] != '\n' {
                    index += 1;
                }
                continue;
            }
            '/' if index + 1 < source.len() && source[index + 1] == '*' => {
                index += 2;
                while index + 1 < source.len()
                    && !(source[index] == '*' && source[index + 1] == '/')
                {
                    index += 1;
                }
                index = (index + 2).min(source.len());
                continue;
            }
            _ => {}
        }
        if character == open {
            depth += 1;
        } else if character == close {
            depth -= 1;
            if depth == 0 {
                return Some(index + 1);
            }
        }
        index += 1;
    }
    None
}

/// Detect a `#[cfg(test)]` attribute at `cursor`.
///
/// Returns the index just after the attribute.
fn cfg_test_attribute_end(source: &[char], cursor: usize) -> Option<usize> {
    if source.get(cursor) != Some(&'#') || source.get(cursor + 1) != Some(&'[') {
        return None;
    }
    let mut index = skip_trivia(source, cursor + 2);
    let name_start = index;
    while index < source.len() && is_identifier_char(source[index]) {
        index += 1;
    }
    let name: String = source[name_start..index].iter().collect();
    if name != "cfg" {
        return None;
    }
    index = skip_trivia(source, index);
    if source.get(index) != Some(&'(') {
        return None;
    }
    let inner_start = index + 1;
    let inner_end = skip_bracketed(source, index)?;
    let inner: String = source[inner_start..inner_end - 1].iter().collect();
    let inner: String = inner
        .chars()
        .filter(|character| !character.is_whitespace())
        .collect();
    if inner != "test" {
        return None;
    }
    let close = skip_trivia(source, inner_end);
    if source.get(close) != Some(&']') {
        return None;
    }
    Some(close + 1)
}

/// Skip the attributes that follow a `#[cfg(test)]` attribute.
fn skip_attributes(source: &[char], mut cursor: usize) -> usize {
    loop {
        cursor = skip_trivia(source, cursor);
        if source.get(cursor) == Some(&'#') && source.get(cursor + 1) == Some(&'[') {
            match skip_bracketed(source, cursor + 1) {
                Some(end) => {
                    cursor = end;
                    continue;
                }
                None => return cursor,
            }
        }
        return cursor;
    }
}

/// Skip the item that a `#[cfg(test)]` attribute guards.
///
/// The item ends with a semicolon (`use ...;`, `mod helpers;`) or with its
/// closing brace (`fn ... { ... }`, `mod tests { ... }`). Both forms are
/// skipped completely, so production code behind a test-only import stays in
/// the scan.
fn skip_annotated_item(source: &[char], cursor: usize) -> usize {
    let mut index = cursor;
    let mut depth = 0usize;
    while index < source.len() {
        let character = source[index];
        match character {
            '"' | '\'' | 'r' | 'b' => {
                index = skip_literal(source, index);
            }
            '/' if index + 1 < source.len() && source[index + 1] == '/' => {
                while index < source.len() && source[index] != '\n' {
                    index += 1;
                }
            }
            '/' if index + 1 < source.len() && source[index + 1] == '*' => {
                index += 2;
                while index + 1 < source.len()
                    && !(source[index] == '*' && source[index + 1] == '/')
                {
                    index += 1;
                }
                index = (index + 2).min(source.len());
            }
            '{' | '[' | '(' => {
                depth += 1;
                index += 1;
            }
            ']' | ')' => {
                depth = depth.saturating_sub(1);
                index += 1;
            }
            '}' => {
                if depth == 0 {
                    return index;
                }
                depth -= 1;
                index += 1;
                if depth == 0 {
                    // An `else` branch continues the guarded item.
                    let next = skip_trivia(source, index);
                    if starts_with_keyword(source, next, "else") {
                        index = next + 4;
                    } else {
                        return index;
                    }
                }
            }
            ';' if depth == 0 => return index + 1,
            _ => {
                index += 1;
            }
        }
    }
    index
}

/// Remove every `#[cfg(test)]` region from the source.
///
/// The test module is the only place where test-only code may reach into
/// modules that the production code must not depend on, so its body is
/// removed from the scan. Everything else stays in the scan, including code
/// that follows the test module.
fn strip_test_regions(source: &str) -> String {
    let characters: Vec<char> = source.chars().collect();
    let mut kept = String::with_capacity(source.len());
    let mut cursor = 0usize;
    while cursor < characters.len() {
        // A `#[cfg(test)]` that is written inside a comment or inside a
        // string literal is not an attribute, so the comment and the literal
        // are copied verbatim and the attribute is not looked for inside
        // them. Otherwise a commented-out attribute would hide the
        // production code that follows it.
        if let Some(end) = comment_end(&characters, cursor) {
            kept.extend(&characters[cursor..end]);
            cursor = end;
            continue;
        }
        if matches!(characters[cursor], '"' | '\'' | 'r' | 'b') {
            let end = skip_literal(&characters, cursor);
            kept.extend(&characters[cursor..end]);
            cursor = end;
            continue;
        }
        if let Some(after_attribute) = cfg_test_attribute_end(&characters, cursor) {
            let item_start = skip_attributes(&characters, after_attribute);
            let item_end = skip_annotated_item(&characters, item_start);
            cursor = item_end.max(after_attribute);
            // Keep the two sides apart so that a match cannot span the hole.
            kept.push('\n');
            continue;
        }
        kept.push(characters[cursor]);
        cursor += 1;
    }
    kept
}

/// Split a `use` group into its top-level entries.
///
/// A comma inside a nested group, inside a string or inside a comment does
/// not end an entry.
fn split_use_entries(group: &[char]) -> Vec<String> {
    let mut entries = Vec::new();
    let mut current = String::new();
    let mut depth = 0usize;
    let mut index = 0usize;
    while index < group.len() {
        let character = group[index];
        match character {
            '"' => {
                let end = skip_string(group, index);
                current.extend(&group[index..end]);
                index = end;
                continue;
            }
            '/' => {
                let Some(end) = comment_end(group, index) else {
                    current.push(character);
                    index += 1;
                    continue;
                };
                current.extend(&group[index..end]);
                index = end;
                continue;
            }
            '{' | '[' | '(' => {
                depth += 1;
                current.push(character);
            }
            '}' | ']' | ')' => {
                depth = depth.saturating_sub(1);
                current.push(character);
            }
            ',' if depth == 0 => {
                entries.push(current.trim().to_owned());
                current.clear();
            }
            _ => current.push(character),
        }
        index += 1;
    }
    if !current.trim().is_empty() {
        entries.push(current.trim().to_owned());
    }
    entries
}

/// The first path segment of a `use` entry, which names the module.
///
/// The entry may start with a comment and may rename the imported item
/// (`models as adapter`); neither changes the module that the entry names.
/// A raw identifier (`r#models`) names the same module as `models`.
fn first_path_segment(entry: &[char]) -> Option<String> {
    let mut cursor = skip_trivia(entry, 0);
    if cursor + 1 < entry.len() && entry[cursor] == 'r' && entry[cursor + 1] == '#' {
        cursor += 2;
    }
    let start = cursor;
    while cursor < entry.len() && is_identifier_char(entry[cursor]) {
        cursor += 1;
    }
    if cursor == start {
        return None;
    }
    Some(entry[start..cursor].iter().collect())
}

/// Collect the modules imported by a `crate::...` path.
///
/// The path may name the module directly (`use crate::models::Foo;`) or open
/// a group (`use crate::{models::Foo, engine::Bar};`), with or without
/// whitespace around the group brace.
fn collect_crate_path(source: &[char], cursor: usize, modules: &mut Vec<String>) {
    if !starts_with_keyword(source, cursor, "crate") {
        return;
    }
    let mut index = skip_trivia(source, cursor + 5);
    if source.get(index) != Some(&':') || source.get(index + 1) != Some(&':') {
        return;
    }
    index = skip_trivia(source, index + 2);
    if source.get(index) == Some(&'{') {
        let Some(end) = skip_bracketed(source, index) else {
            return;
        };
        for entry in split_use_entries(&source[index + 1..end - 1]) {
            let entry: Vec<char> = entry.chars().collect();
            if let Some(module) = first_path_segment(&entry) {
                modules.push(module);
            }
        }
        return;
    }
    if let Some(module) = first_path_segment(&source[index..]) {
        modules.push(module);
    }
}

/// Collect the module names imported through `use crate::...` paths.
///
/// Inside such a group the module name appears without the `crate::`
/// prefix, so the plain `crate::<module>` scan would miss it.
fn use_crate_modules(source: &str) -> Vec<String> {
    let characters: Vec<char> = source.chars().collect();
    let mut modules = Vec::new();
    let mut cursor = 0usize;
    while cursor < characters.len() {
        let character = characters[cursor];
        match character {
            '"' | '\'' | 'r' | 'b' => {
                cursor = skip_literal(&characters, cursor);
                continue;
            }
            '/' if cursor + 1 < characters.len() && characters[cursor + 1] == '/' => {
                while cursor < characters.len() && characters[cursor] != '\n' {
                    cursor += 1;
                }
                continue;
            }
            '/' if cursor + 1 < characters.len() && characters[cursor + 1] == '*' => {
                cursor += 2;
                while cursor + 1 < characters.len()
                    && !(characters[cursor] == '*' && characters[cursor + 1] == '/')
                {
                    cursor += 1;
                }
                cursor = (cursor + 2).min(characters.len());
                continue;
            }
            _ => {}
        }
        if starts_with_keyword(&characters, cursor, "use") {
            let path = skip_trivia(&characters, cursor + 3);
            collect_crate_path(&characters, path, &mut modules);
            cursor += 3;
            continue;
        }
        cursor += 1;
    }
    modules
}

/// Report every forbidden module the source depends on.
fn forbidden_modules_in_source(source: &str, forbidden: &[&str]) -> Vec<String> {
    let stripped = strip_test_regions(source);
    let grouped = use_crate_modules(&stripped);
    let mut found = Vec::new();
    for module in forbidden {
        if stripped.contains(&format!("crate::{module}")) {
            found.push(format!("crate::{module}"));
        }
        if grouped.iter().any(|imported| imported == module) {
            found.push(format!("use crate::{{{module}::..}}"));
        }
    }
    found
}

/// Read a source file and report every forbidden module it depends on.
fn forbidden_modules(relative: &str, forbidden: &[&str]) -> Vec<String> {
    let path = repository_root().join(relative);
    let source = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()));
    forbidden_modules_in_source(&source, forbidden)
}

/// The dependency names a manifest declares in its `[dependencies]` tables.
///
/// Only the dependency tables are read: the package metadata and the
/// repository URL may mention a crate without depending on it.
fn declared_dependencies(manifest: &str) -> Vec<String> {
    let mut names = Vec::new();
    let mut in_dependencies = false;
    for line in manifest.lines() {
        let line = line.split('#').next().unwrap_or_default().trim();
        if line.is_empty() {
            continue;
        }
        if line.starts_with('[') {
            in_dependencies = line == "[dependencies]" || line.starts_with("[dependencies.");
            continue;
        }
        if !in_dependencies {
            continue;
        }
        if let Some((name, _)) = line.split_once('=') {
            let name = name.trim().trim_matches('"');
            if !name.is_empty() {
                names.push(name.to_owned());
            }
        }
    }
    names
}

/// Read a source file and drop every `#[cfg(test)]` region.
fn non_test_source(relative: &str) -> String {
    let path = repository_root().join(relative);
    let source = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()));
    strip_test_regions(&source)
}

#[test]
fn chat_crate_does_not_depend_on_the_runtime_or_the_tensors() {
    let manifest = repository_root().join(CHAT_CRATE_MANIFEST);
    let source = std::fs::read_to_string(&manifest)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", manifest.display()));
    let declared = declared_dependencies(&source);
    let mut violations = Vec::new();
    for dependency in CHAT_CRATE_FORBIDDEN {
        if declared.iter().any(|name| name == dependency) {
            violations.push(dependency);
        }
    }
    assert!(
        violations.is_empty(),
        "the chat crate must not depend on the tensor runtime, the argument parser, the server or the root crate:\n{}",
        violations.join("\n")
    );
}

#[test]
fn server_does_not_depend_on_the_model_adapters() {
    let violations = forbidden_modules("src/server.rs", &["models"]);
    assert!(
        violations.is_empty(),
        "the server must not depend on the model adapters:\n{}",
        violations.join("\n")
    );
}

#[test]
fn benchmark_does_not_depend_on_the_server() {
    let directory = repository_root().join("src/benchmark");
    let mut files: Vec<PathBuf> = std::fs::read_dir(&directory)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", directory.display()))
        .map(|entry| entry.expect("benchmark entry should be readable").path())
        .filter(|path| path.extension().is_some_and(|extension| extension == "rs"))
        .collect();
    files.sort();

    let mut violations = Vec::new();
    for file in files {
        let relative = file
            .strip_prefix(repository_root())
            .expect("benchmark file should live in the repository")
            .to_string_lossy()
            .replace('\\', "/");
        for violation in forbidden_modules(&relative, &["server"]) {
            violations.push(format!("{relative} references {violation}"));
        }
    }
    assert!(
        violations.is_empty(),
        "the benchmark harness must not depend on the server:\n{}",
        violations.join("\n")
    );
}

#[test]
fn backend_selectors_do_not_depend_on_the_argument_parser() {
    let source = non_test_source("src/backend.rs");
    assert!(
        !source.contains("clap"),
        "the backend selectors must not depend on the argument parser"
    );
}

#[test]
fn every_scanned_file_exists() {
    assert!(Path::new(&repository_root().join(CHAT_CRATE_MANIFEST)).is_file());
    assert!(Path::new(&repository_root().join("src/server.rs")).is_file());
}

#[test]
fn a_test_only_import_does_not_hide_the_production_code_behind_it() {
    let source = "\
#[cfg(test)]
use crate::models::Adapter;

fn production() {
    let _ = crate::engine::EngineConfig::default();
}
";
    let violations = forbidden_modules_in_source(source, &["engine"]);
    assert!(
        !violations.is_empty(),
        "the production code behind a test-only import must still be scanned"
    );
    assert!(
        forbidden_modules_in_source(source, &["models"]).is_empty(),
        "the test-only import itself must not be reported"
    );
}

#[test]
fn a_test_only_module_does_not_hide_the_production_code_behind_it() {
    let source = "\
#[cfg(test)]
mod tests {
    use crate::models::Adapter;

    #[test]
    fn works() {
        let _ = crate::models::Adapter::new();
    }
}

fn production() {
    let _ = crate::engine::EngineConfig::default();
}
";
    let violations = forbidden_modules_in_source(source, &["engine"]);
    assert!(
        !violations.is_empty(),
        "the production code behind a test module must still be scanned"
    );
    assert!(
        forbidden_modules_in_source(source, &["models"]).is_empty(),
        "the test module itself must not be reported"
    );
}

#[test]
fn a_grouped_import_is_seen_with_and_without_whitespace() {
    for source in [
        "use crate::{models::Adapter};\n",
        "use crate:: {models::Adapter};\n",
        "use crate::{\n    models::Adapter,\n};\n",
        "use crate:: {models::Adapter, engine::Engine};\n",
    ] {
        let violations = forbidden_modules_in_source(source, &["models"]);
        assert!(
            !violations.is_empty(),
            "a grouped import must be seen in `{source}`"
        );
    }
    let violations = forbidden_modules_in_source(
        "use crate:: {models::Adapter, engine::Engine};\n",
        &["engine"],
    );
    assert!(
        !violations.is_empty(),
        "every entry of a grouped import must be seen"
    );
}

#[test]
fn a_plain_import_is_seen() {
    let violations = forbidden_modules_in_source("use crate::models::Adapter;\n", &["models"]);
    assert!(!violations.is_empty(), "a plain import must be seen");
}

#[test]
fn an_unrelated_import_is_not_reported() {
    let violations = forbidden_modules_in_source(
        "use crate::chat::{ChatMessage, ChatRole};\nuse crate::finish::FinishReason;\n",
        &["models", "engine"],
    );
    assert!(
        violations.is_empty(),
        "an allowed import must not be reported: {violations:?}"
    );
}

#[test]
fn a_commented_out_test_attribute_does_not_hide_the_code_behind_it() {
    let source = "\
// #[cfg(test)]
use crate::models::Adapter;

fn production() {
    let _ = crate::engine::EngineConfig::default();
}
";
    let violations = forbidden_modules_in_source(source, &["models"]);
    assert!(
        !violations.is_empty(),
        "a commented-out attribute must not hide the import behind it"
    );
    let violations = forbidden_modules_in_source(source, &["engine"]);
    assert!(
        !violations.is_empty(),
        "a commented-out attribute must not hide the production code behind it"
    );
}

#[test]
fn a_test_attribute_inside_a_block_comment_does_not_hide_the_code_behind_it() {
    let source = "\
/* #[cfg(test)] */
use crate::models::Adapter;

fn production() {
    let _ = crate::engine::EngineConfig::default();
}
";
    let violations = forbidden_modules_in_source(source, &["models"]);
    assert!(
        !violations.is_empty(),
        "a test attribute inside a block comment must not hide the import behind it"
    );
    let violations = forbidden_modules_in_source(source, &["engine"]);
    assert!(
        !violations.is_empty(),
        "a test attribute inside a block comment must not hide the production code behind it"
    );
}

#[test]
fn a_test_attribute_inside_a_string_does_not_hide_the_code_behind_it() {
    let source = "\
const NOTE: &str = \"#[cfg(test)]\";

fn production() {
    let _ = crate::engine::EngineConfig::default();
}
";
    let violations = forbidden_modules_in_source(source, &["engine"]);
    assert!(
        !violations.is_empty(),
        "a test attribute inside a string must not hide the production code behind it"
    );
}

#[test]
fn a_grouped_import_with_an_alias_is_seen() {
    for source in [
        "use crate::{models as adapter};\n",
        "use crate::{models::Adapter as adapter};\n",
        "use crate::{ /* a comment */ models as adapter, engine::Engine };\n",
    ] {
        let violations = forbidden_modules_in_source(source, &["models"]);
        assert!(
            !violations.is_empty(),
            "an aliased grouped import must be seen in `{source}`"
        );
    }
}

#[test]
fn a_grouped_import_with_a_comment_is_seen() {
    for source in [
        "use crate::{/* comment */ models::Adapter};\n",
        "use crate::{models::Adapter /* comment */};\n",
        "use crate::{models::Adapter, /* a, comma */ engine::Engine};\n",
    ] {
        let violations = forbidden_modules_in_source(source, &["models"]);
        assert!(
            !violations.is_empty(),
            "a commented grouped import must be seen in `{source}`"
        );
    }
    let violations = forbidden_modules_in_source(
        "use crate::{models::Adapter, /* a, comma */ engine::Engine};\n",
        &["engine"],
    );
    assert!(
        !violations.is_empty(),
        "a comma inside a comment must not split the grouped import"
    );
}
