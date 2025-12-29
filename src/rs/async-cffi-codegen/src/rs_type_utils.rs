use anyhow::{Result, bail};

use rs_schema::{FunctionArgSchema, GenericParamSchema, TypeSchema};

use crate::{
    CodeGen,
    cffi_type_utils::{
        CffiTypeElementSpec, cffi_ty_stack_to_rust_cffi_ty, cffi_type_str_to_type_stack,
    },
};

impl CodeGen for TypeSchema {
    fn codegen(&self, indent: usize) -> String {
        let generic_args_str = if self.generic_ty_args.len() == 0 {
            String::new()
        } else {
            let args = self
                .generic_ty_args
                .iter()
                .map(|ta| ta.codegen(indent))
                .collect::<Vec<_>>()
                .join(", ");
            format!("<{}>", args)
        };

        format!("{}{}", self.ty, generic_args_str)
    }
}

pub fn substitute_generic_args(
    ty: &TypeSchema,
    generics: &Vec<GenericParamSchema>,
) -> Result<TypeSchema> {
    let outer_ty = if let Some(cffi_ty) = generics
        .iter()
        .find(|g| ty.ty == g.name)
        .and_then(|g| g.annotations.as_ref())
        .and_then(|a| a.cffi_type.as_ref())
    {
        cffi_ty_stack_to_rust_cffi_ty(&cffi_type_str_to_type_stack(cffi_ty)?)?
    } else {
        ty.ty.clone()
    };

    let generic_args = ty
        .generic_ty_args
        .iter()
        .map(|a| substitute_generic_args(a, generics))
        .collect::<Result<Vec<_>, _>>()?;

    Ok(TypeSchema {
        ty: outer_ty,
        generic_ty_args: generic_args,
    })
}

/// Generate transforms based on explicit cffi_type annotations for an argument.
pub fn annotation_transforms_for_arg(
    arg: &FunctionArgSchema,
    target_type: Vec<CffiTypeElementSpec>,
) -> Result<Option<String>> {
    if let Some(annotations) = &arg.annotations {
        let annotation_transform = if annotations.collection_as_item {
            Some(format!(
                "let {} = {}.into_iter().next().unwrap();\n",
                arg.name, arg.name
            ))
        } else {
            None
        };

        if let Some(cffi_type) = &annotations.cffi_type {
            let expected_cffi_type_stack = cffi_type_str_to_type_stack(cffi_type)?;

            if target_type != expected_cffi_type_stack {
                bail!(
                    "cffi_type annotation {:?} does not match type derived from argument spec {:?}. Arg type: {}",
                    expected_cffi_type_stack,
                    target_type,
                    arg.ty
                        .as_ref()
                        .map(|ty| ty.ty.clone())
                        .unwrap_or("".to_string())
                );
            }
        }

        Ok(annotation_transform)
    } else {
        Ok(None)
    }
}

/// Convert a parsed type schema into a flattened vector of its nested type names.
pub fn ty_schema_to_string_stack(mut ty_schema: &TypeSchema) -> Result<Vec<String>> {
    let mut stack = Vec::new();
    loop {
        stack.push(ty_schema.ty.clone());
        if ty_schema.generic_ty_args.is_empty() {
            return Ok(stack);
        }
        if ty_schema.generic_ty_args.len() > 1 {
            bail!("Non-simple type stack");
        }
        ty_schema = &ty_schema.generic_ty_args[0];
    }
}

/// Convert a trait function argument into its expected CFFI type stack specification.
pub fn fn_arg_to_async_cffi_type_spec(arg: &FunctionArgSchema) -> Result<Vec<CffiTypeElementSpec>> {
    let fn_arg_type_stack = match arg.ty.as_ref() {
        None => {
            return Ok(vec![
                CffiTypeElementSpec {
                    is_pointer: true,
                    is_optional: false,
                    explicit_type: None,
                },
                CffiTypeElementSpec {
                    is_pointer: false,
                    is_optional: false,
                    explicit_type: None,
                },
            ]);
        }
        Some(ty) => ty_schema_to_string_stack(ty)?,
    };

    let mut cffi_type_stack = vec![];
    let mut current_element = CffiTypeElementSpec {
        is_pointer: false,
        is_optional: false,
        explicit_type: None,
    };
    for fn_arg_type_elem in fn_arg_type_stack {
        match fn_arg_type_elem.as_str() {
            "Option" => {
                current_element.is_optional = true;
                // Option always expressed with a pointer
                current_element.is_pointer = true;
            }
            "Vec" => {
                // TODO: Currently just assuming Vec always expressed as CffiPointerBuffer
                current_element.is_pointer = true;
                current_element.explicit_type = Some("CffiPointerBuffer".to_string());
                cffi_type_stack.push(current_element);
                current_element = CffiTypeElementSpec {
                    is_pointer: false,
                    is_optional: false,
                    explicit_type: None,
                };
            }
            "AnyArc" => {
                // AnyArc always expressed with a pointer
                current_element.is_pointer = true;
                cffi_type_stack.push(current_element);
                current_element = CffiTypeElementSpec {
                    is_pointer: false,
                    is_optional: false,
                    explicit_type: None,
                };
            }
            "Box" => {
                current_element.is_pointer = true;
            }
            "usize" => current_element.explicit_type = Some("c_ulong".to_string()),
            arg_ty_elem => {
                if current_element.is_pointer && arg_ty_elem.starts_with("dyn") {
                    current_element.is_pointer = false;
                    current_element.explicit_type = Some(format!("Cffi{}", &arg_ty_elem[4..]))
                }

                // No-op
            }
        }
    }
    cffi_type_stack.push(current_element);

    Ok(cffi_type_stack)
}
