use anyhow::{Result, bail};

/// Schema for a parsed CFFI type, including any generic arguments.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CffiTypeSchema {
    pub ty: String,
    pub generic_ty_args: Vec<CffiTypeSchema>,
}

#[derive(PartialEq, Eq, Debug)]
/// Represents a single element in the expected CFFI type stack for an argument.
pub struct CffiTypeElementSpec {
    pub is_pointer: bool,
    pub is_optional: bool,
    pub explicit_type: Option<String>,
}

/// Parse a CFFI type annotation string into a structured schema.
pub fn cffi_type_str_to_schema(cffi_type: &str) -> Result<CffiTypeSchema> {
    if let Some((schema, remaining)) = parse_cffi_ty(cffi_type) {
        if remaining.trim().is_empty() {
            Ok(schema)
        } else {
            bail!("CFFI type had trailing characters")
        }
    } else {
        bail!("CFFI type could not be parsed to schema")
    }
}

/// Parse a CFFI type annotation string into a stack of type element specifications.
pub fn cffi_type_str_to_type_stack(cffi_type: &str) -> Result<Vec<CffiTypeElementSpec>> {
    let schema = cffi_type_str_to_schema(cffi_type)?;
    cffi_ty_schema_to_type_stack(&schema)
}

/// Convert a parsed CFFI type schema into a flattened vector of its nested type names.
pub fn cffi_ty_schema_to_string_stack(mut ty_schema: &CffiTypeSchema) -> Result<Vec<String>> {
    let mut stack = Vec::new();
    loop {
        stack.push(ty_schema.ty.clone());
        if ty_schema.generic_ty_args.is_empty() {
            return Ok(stack);
        }
        if ty_schema.generic_ty_args.len() > 1 {
            bail!("Non-simple CFFI type stack");
        }
        ty_schema = &ty_schema.generic_ty_args[0];
    }
}

pub fn cffi_ty_schema_to_type_stack(schema: &CffiTypeSchema) -> Result<Vec<CffiTypeElementSpec>> {
    cffi_ty_schema_to_string_stack(schema)?
        .into_iter()
        .map(|expected_elem| match expected_elem.as_str() {
            "opt_ptr" => Ok(CffiTypeElementSpec {
                is_optional: true,
                is_pointer: true,
                explicit_type: None,
            }),
            "ptr" => Ok(CffiTypeElementSpec {
                is_pointer: true,
                is_optional: false,
                explicit_type: None,
            }),
            "CffiPointerBuffer" => Ok(CffiTypeElementSpec {
                is_pointer: true,
                is_optional: false,
                explicit_type: Some("CffiPointerBuffer".to_string()),
            }),
            _ => Ok(CffiTypeElementSpec {
                is_optional: false,
                is_pointer: false,
                explicit_type: if expected_elem.starts_with("Cffi") {
                    Some(expected_elem.to_string())
                } else {
                    None
                },
            }),
        })
        .collect::<Result<Vec<_>>>()
}

pub fn cffi_ty_stack_to_rust_cffi_ty(stack: &Vec<CffiTypeElementSpec>) -> Result<String> {
    if stack.is_empty() {
        bail!("Empty cffi type stack cannot be converted to rust type");
    }

    if let Some(ety) = &stack[0].explicit_type {
        return Ok(ety.clone());
    }

    if stack[0].is_pointer || stack[0].is_optional {
        return Ok("SafePtr".to_string());
    }

    bail!("No translation from cffi type stack to rust type")
}

fn parse_cffi_ty(ty: &str) -> Option<(CffiTypeSchema, &str)> {
    let ty = ty.trim();
    if ty.is_empty() {
        None
    } else {
        let end = ty.find(|c| ",<>".contains(c)).unwrap_or(ty.len());
        let end_c = ty.chars().nth(end);
        let outer_ty = ty[..end].trim().to_string();
        if end_c == Some('<') {
            let right = &ty[end + 1..];
            if let Some((args, right)) = parse_cffi_type_args(right) {
                Some((
                    CffiTypeSchema {
                        ty: outer_ty,
                        generic_ty_args: args,
                    },
                    right,
                ))
            } else {
                None
            }
        } else {
            Some((
                CffiTypeSchema {
                    ty: outer_ty,
                    generic_ty_args: vec![],
                },
                &ty[end..],
            ))
        }
    }
}

fn parse_cffi_type_args(tys: &str) -> Option<(Vec<CffiTypeSchema>, &str)> {
    let mut tys = tys.trim();
    let mut args = Vec::new();
    while let Some((arg, remaining)) = parse_cffi_ty(tys) {
        args.push(arg);
        tys = remaining.trim();
        let next = tys.chars().nth(0);
        if next == Some('>') {
            tys = tys[1..].trim();
            break;
        }

        if next != Some(',') {
            return None;
        }
        tys = tys[1..].trim();

        let next = tys.chars().nth(0);
        if next == Some('>') {
            tys = tys[1..].trim();
            break;
        }
    }

    Some((args, tys))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_nested_cffi_type_schema() {
        let schema = cffi_type_str_to_schema("opt_ptr<CffiBuffer<ptr>>").unwrap();

        assert_eq!(schema.ty, "opt_ptr");
        assert_eq!(schema.generic_ty_args.len(), 1);
        assert_eq!(schema.generic_ty_args[0].ty, "CffiBuffer");
        assert_eq!(schema.generic_ty_args[0].generic_ty_args.len(), 1);
        assert_eq!(schema.generic_ty_args[0].generic_ty_args[0].ty, "ptr");
    }

    #[test]
    fn rejects_trailing_characters_in_cffi_type() {
        let err = cffi_type_str_to_schema("ptr<opt_ptr> extra").unwrap_err();
        assert!(err.to_string().contains("trailing characters"));
    }

    #[test]
    fn flattens_schema_into_string_stack() {
        let schema = cffi_type_str_to_schema("opt_ptr<CffiBuffer<ptr>>").unwrap();
        let stack = cffi_ty_schema_to_string_stack(&schema).unwrap();

        assert_eq!(stack, vec!["opt_ptr", "CffiBuffer", "ptr"]);
    }

    #[test]
    fn converts_schema_to_type_stack() {
        let schema = cffi_type_str_to_schema("opt_ptr<CffiBuffer<ptr>>").unwrap();
        let stack = cffi_ty_schema_to_type_stack(&schema).unwrap();

        assert_eq!(stack.len(), 3);
        assert!(stack[0].is_pointer);
        assert!(stack[0].is_optional);
        assert_eq!(stack[1].explicit_type.as_deref(), Some("CffiBuffer"));
        assert!(!stack[2].is_optional);
        assert!(stack[2].is_pointer);
    }

    #[test]
    fn errors_on_non_simple_type_stack() {
        let err = cffi_type_str_to_type_stack("ptr<ptr,ptr>").unwrap_err();
        assert!(err.to_string().contains("Non-simple CFFI type stack"));
    }
}
