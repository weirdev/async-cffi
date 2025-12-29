use std::{
    collections::HashMap,
    fs,
    io::{Error, ErrorKind, Result, Write as _},
    path::Path,
    process::{Command, Stdio},
};

use rs_schema::TraitSchema;

use crate::{CodeGen as _, trait_to_async_cffi_py_schema, trait_to_async_cffi_rs_schema};

pub fn generate_rs(
    ios_trait_schema: &TraitSchema,
    target_path: &Path,
    supertrait_schemas: &HashMap<String, TraitSchema>,
) -> Result<()> {
    let file_schema = trait_to_async_cffi_rs_schema(ios_trait_schema, supertrait_schemas)
        .map_err(|m| Error::new(ErrorKind::Other, m))?;
    let new_contents = file_schema.codegen(0);
    let formatted_new_contents = format_rust_code(&new_contents).unwrap_or_else(|err| {
        println!("cargo:warning=Failed to format generated code: {err}");
        new_contents
    });
    write_if_changed(&target_path, &formatted_new_contents)
}

fn format_rust_code(code: &str) -> Result<String> {
    let edition = std::env::var("CARGO_PKG_EDITION").unwrap_or_else(|err| {
        println!(
            "cargo:warning=Using default edition 2024 for rustfmt (CARGO_PKG_EDITION missing: {err})"
        );
        "2024".to_string()
    });

    let mut process = Command::new("rustfmt")
        .args(["--edition", &edition, "--emit", "stdout"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .map_err(|e| Error::new(ErrorKind::Other, format!("failed to spawn rustfmt: {e}")))?;

    if let Some(mut stdin) = process.stdin.take() {
        stdin.write_all(code.as_bytes()).map_err(|e| {
            Error::new(
                ErrorKind::Other,
                format!("failed to write to rustfmt stdin: {e}"),
            )
        })?;
    }

    let output = process.wait_with_output().map_err(|e| {
        Error::new(
            ErrorKind::Other,
            format!("failed to read rustfmt output: {e}"),
        )
    })?;

    if !output.status.success() {
        return Err(Error::new(
            ErrorKind::Other,
            format!(
                "rustfmt exited with status {}: {}",
                output.status,
                String::from_utf8_lossy(&output.stderr)
            ),
        ));
    }

    String::from_utf8(output.stdout).map_err(|e| {
        Error::new(
            ErrorKind::Other,
            format!("rustfmt produced invalid UTF-8: {e}"),
        )
    })
}

fn format_python_code(code: &str) -> Result<String> {
    let mut process = Command::new("python")
        .args(["-m", "black", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .map_err(|e| Error::new(ErrorKind::Other, format!("failed to spawn black: {e}")))?;

    if let Some(mut stdin) = process.stdin.take() {
        stdin.write_all(code.as_bytes()).map_err(|e| {
            Error::new(
                ErrorKind::Other,
                format!("failed to write to black stdin: {e}"),
            )
        })?;
    }

    let output = process.wait_with_output().map_err(|e| {
        Error::new(
            ErrorKind::Other,
            format!("failed to read black output: {e}"),
        )
    })?;

    if !output.status.success() {
        return Err(Error::new(
            ErrorKind::Other,
            format!(
                "black exited with status {}: {}",
                output.status,
                String::from_utf8_lossy(&output.stderr)
            ),
        ));
    }

    String::from_utf8(output.stdout).map_err(|e| {
        Error::new(
            ErrorKind::Other,
            format!("black produced invalid UTF-8: {e}"),
        )
    })
}

pub fn generate_py(
    ios_trait_schema: &TraitSchema,
    target_path: &Path,
    supertrait_schemas: &HashMap<String, TraitSchema>,
) -> Result<()> {
    let file_schema = trait_to_async_cffi_py_schema(ios_trait_schema, supertrait_schemas)
        .map_err(|m| Error::new(ErrorKind::Other, m))?;

    let new_contents = file_schema.codegen(0);
    let formatted_new_contents = format_python_code(&new_contents).unwrap_or_else(|err| {
        println!("cargo:warning=Failed to format generated Python code: {err}");
        new_contents
    });

    write_if_changed(&target_path, &formatted_new_contents)
}

fn write_if_changed(target_path: &Path, contents: &str) -> Result<()> {
    let normalized_new_contents = normalize_contents(contents);
    let should_write = fs::read_to_string(target_path)
        .map(|existing| normalize_contents(&existing) != normalized_new_contents)
        .unwrap_or(true);

    if should_write {
        if let Some(parent) = target_path.parent() {
            fs::create_dir_all(parent).map_err(|e| {
                Error::new(
                    ErrorKind::Other,
                    format!(
                        "failed to create parent directories for {}: {e}",
                        parent.display()
                    ),
                )
            })?;
        }

        let mut file = fs::File::create(target_path).map_err(|e| {
            Error::new(
                ErrorKind::Other,
                format!("failed to create file {}: {e}", target_path.display()),
            )
        })?;
        file.write_all(normalized_new_contents.as_bytes())
            .map_err(|e| {
                Error::new(
                    ErrorKind::Other,
                    format!("failed to write file {}: {e}", target_path.display()),
                )
            })?;
        file.sync_all().map_err(|e| {
            Error::new(
                ErrorKind::Other,
                format!("failed to sync file {}: {e}", target_path.display()),
            )
        })?;
    }

    Ok(())
}

fn normalize_contents(contents: &str) -> String {
    let mut normalized = contents.replace("\r\n", "\n");
    if !normalized.ends_with('\n') {
        normalized.push('\n');
    }

    normalized
}
