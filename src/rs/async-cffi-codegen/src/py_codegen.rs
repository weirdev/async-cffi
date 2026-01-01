use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;

use anyhow::{Context, Result, bail};
use heck::ToSnakeCase;

use handlebars::Handlebars;
use serde_json::json;
use rs_schema::{FunctionArgSchema, FunctionSchema, TraitSchema, TypeSchema};

use crate::{
    CodeGen,
    cffi_type_utils::{
        CffiTypeElementSpec, CffiTypeSchema, cffi_type_str_to_schema, cffi_type_str_to_type_stack,
    },
    rs_type_utils::fn_arg_to_async_cffi_type_spec,
};

#[derive(Debug)]
pub struct PyFileSchema {
    pub imports: Vec<String>,
    pub classes: Vec<PyClassSchema>,
}

impl CodeGen for PyFileSchema {
    fn codegen(&self, indent: usize) -> String {
        let mut output = String::new();

        // Imports
        for import in &self.imports {
            output.push_str(import);
            output.push('\n');
        }
        output.push('\n');

        for class_schema in &self.classes {
            output.push_str(&class_schema.codegen(indent));
            output.push('\n');
        }
        output
    }
}

#[derive(Debug)]
pub struct PyClassSchema {
    name: String,
    parent_types: Vec<PyTypeSchema>,
    class_vars: Vec<String>,
    functions: Vec<PyFunctionSchema>,
}

impl CodeGen for PyClassSchema {
    fn codegen(&self, indent: usize) -> String {
        let mut output = String::new();
        let pad = " ".repeat(indent);

        // Class definition
        if self.parent_types.is_empty() {
            writeln!(&mut output, "{}class {}:", pad, self.name).unwrap();
        } else {
            writeln!(
                &mut output,
                "{}class {}({}):",
                pad,
                self.name,
                self.parent_types
                    .iter()
                    .map(|ty| ty.codegen(0))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
            .unwrap();
        }

        // Class variables
        if self.class_vars.is_empty() {
            if self.functions.is_empty() {
                writeln!(&mut output, "{}    pass", pad).unwrap();
            }
        } else {
            for var in &self.class_vars {
                writeln!(&mut output, "{}    {}", pad, var).unwrap();
            }
        }

        for f in &self.functions {
            writeln!(&mut output, "{}", f.codegen(indent + 1)).unwrap();
        }

        output
    }
}

#[derive(Debug)]
struct PyFunctionSchema {
    name: String,
    args: Vec<PyFunctionArgSchema>,
    return_type: Option<PyTypeSchema>,
    is_async: bool,
    body: Vec<String>, // statements
    py_annotations: Vec<String>,
}

impl CodeGen for PyFunctionSchema {
    fn codegen(&self, indent: usize) -> String {
        let pad = "    ".repeat(indent);
        let mut output = String::new();

        for annotation in &self.py_annotations {
            writeln!(&mut output, "{}{}", pad, annotation).unwrap();
        }

        let async_prefix = if self.is_async { "async " } else { "" };

        let args_str = self
            .args
            .iter()
            .map(|a| a.codegen(0))
            .collect::<Vec<_>>()
            .join(", ");

        let ret_str = self
            .return_type
            .as_ref()
            .map(|r| format!(" -> {}", r.codegen(0)))
            .unwrap_or_default();

        writeln!(
            &mut output,
            "{}{}def {}({}){}:",
            pad, async_prefix, self.name, args_str, ret_str
        )
        .unwrap();

        for stmt in &self.body {
            for line in stmt.split('\n') {
                writeln!(&mut output, "{}    {}", pad, line).unwrap();
            }
        }

        output
    }
}

#[derive(Debug)]
struct PyFunctionArgSchema {
    name: String,
    ty: Option<PyTypeSchema>,
}

impl CodeGen for PyFunctionArgSchema {
    fn codegen(&self, _indent: usize) -> String {
        let ty_str = self
            .ty
            .as_ref()
            .map(|ty| format!(": {}", ty.codegen(0)))
            .unwrap_or_default();
        format!("{}{}", self.name, ty_str)
    }
}

#[derive(Debug)]
struct PyTypeSchema {
    ty: String,
    generic_ty_args: Vec<PyTypeSchema>,
}

impl CodeGen for PyTypeSchema {
    fn codegen(&self, _indent: usize) -> String {
        let args_str = if self.generic_ty_args.len() > 0 {
            format!(
                "[{}]",
                self.generic_ty_args
                    .iter()
                    .map(|ty| ty.codegen(0))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        } else {
            String::new()
        };

        format!("{}{}", self.ty, args_str)
    }
}

pub fn trait_to_async_cffi_schema(
    trait_schema: &TraitSchema,
    supertrait_schemas: &HashMap<String, TraitSchema>,
) -> Result<PyFileSchema> {
    let mut classes = Vec::new();

    let mut imports: Vec<String> = vec![
        "import asyncio".to_string(),
        "import ctypes".to_string(),
        "import typing".to_string(),
        "from typing import Optional, override".to_string(),
        "from centconf.async_cffi import ASYNC_CFFI_LIB".to_string(),
        "from centconf.async_cffi import CffiFutureDriver".to_string(),
        "from centconf.async_cffi import AsyncCffiLib, CffiPointerBuffer, PyCffiFuture".to_string(),
    ];

    imports.extend(interface_imports(trait_schema));
    imports.extend(supertrait_cffi_imports(trait_schema));
    imports.extend(cffi_type_imports(trait_schema)?);
    dedup_preserve_order(&mut imports);

    let struct_schema = cffi_struct_schema(trait_schema, supertrait_schemas)?;
    classes.push(struct_schema);

    let generator_schema = cffi_generator_schema(trait_schema, supertrait_schemas)?;
    classes.push(generator_schema);

    Ok(PyFileSchema { imports, classes })
}

fn dedup_preserve_order(values: &mut Vec<String>) {
    let mut seen = HashSet::new();
    values.retain(|v| seen.insert(v.clone()));
}

fn interface_names_from_args(trait_schema: &TraitSchema) -> Vec<String> {
    trait_schema
        .functions
        .iter()
        .flat_map(|f| {
            f.args
                .iter()
                .filter_map(|arg| arg.ty.as_ref().map(|ty| interface_names_from_ty(ty)))
        })
        .flatten()
        .collect()
}

struct TypeSchemaIter<'a> {
    ty: &'a TypeSchema,
    state: TypeSchemaIterState<'a>,
}

enum TypeSchemaIterState<'a> {
    Start,
    InGeneric(usize, Option<Box<TypeSchemaIter<'a>>>),
}

impl<'a> TypeSchemaIter<'a> {
    fn new(ty: &'a TypeSchema) -> Self {
        Self {
            ty,
            state: TypeSchemaIterState::Start,
        }
    }
}

impl<'a> Iterator for TypeSchemaIter<'a> {
    type Item = &'a TypeSchema;

    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.state {
            TypeSchemaIterState::Start => {
                self.state = TypeSchemaIterState::InGeneric(0, None);
                Some(self.ty)
            }
            TypeSchemaIterState::InGeneric(idx, None) => {
                if *idx < self.ty.generic_ty_args.len() {
                    self.state = TypeSchemaIterState::InGeneric(
                        *idx,
                        Some(Box::new(TypeSchemaIter::new(
                            &self.ty.generic_ty_args[*idx],
                        ))),
                    );
                    return self.next();
                }
                None
            }
            TypeSchemaIterState::InGeneric(idx, Some(iter)) => {
                if let Some(item) = iter.next() {
                    Some(item)
                } else {
                    self.state = TypeSchemaIterState::InGeneric(*idx + 1, None);
                    self.next()
                }
            }
        }
    }
}

fn interface_names_from_ty(ty: &TypeSchema) -> Vec<String> {
    TypeSchemaIter::new(ty)
        .filter_map(|ty| {
            ty.ty.find("dyn").and_then(|idx| {
                let after = ty.ty[idx + 4..].trim();
                let ident: String = after
                    .chars()
                    .take_while(|c| c.is_alphanumeric() || *c == '_')
                    .collect();
                if !ident.is_empty() {
                    Some(format!("I{}", ident))
                } else {
                    None
                }
            })
        })
        .collect()
}

fn interface_imports(trait_schema: &TraitSchema) -> Vec<String> {
    let mut interfaces = vec![format!("I{}", trait_schema.name)];
    interfaces.extend(
        trait_schema
            .supertraits
            .iter()
            .filter(|st| (st.ty != "Send") && (st.ty != "Sync"))
            .map(|st| format!("I{}", st.ty)),
    );
    interfaces.extend(interface_names_from_args(trait_schema));
    dedup_preserve_order(&mut interfaces);

    if interfaces.is_empty() {
        Vec::new()
    } else {
        vec![format!(
            "from n_observer.core import {}",
            interfaces.join(", ")
        )]
    }
}

fn supertrait_cffi_imports(trait_schema: &TraitSchema) -> Vec<String> {
    trait_schema
        .supertraits
        .iter()
        .filter(|st| (st.ty != "Send") && (st.ty != "Sync"))
        .map(|st| {
            let module = cffi_module_for_type(&format!("Cffi{}", st.ty));
            format!(
                "from {module} import Cffi{st}, Cffi{st}Generator",
                module = module,
                st = st.ty
            )
        })
        .collect()
}

fn cffi_type_imports(trait_schema: &TraitSchema) -> Result<Vec<String>> {
    let mut imports = Vec::new();
    for func in &trait_schema.functions {
        for arg in &func.args {
            if let Some(annotations) = &arg.annotations {
                if let Some(cffi_type) = &annotations.cffi_type {
                    let stack = cffi_type_str_to_type_stack(cffi_type)?;
                    if let Some(explicit) = &stack[0].explicit_type {
                        if explicit == "CffiPointerBuffer" {
                            continue;
                        }
                        let module = cffi_module_for_type(explicit);
                        let generator = format!("{explicit}Generator");
                        imports.push(format!(
                            "from {module} import {explicit}, {generator}",
                            module = module,
                            explicit = explicit,
                            generator = generator
                        ));
                    }
                }
            }
        }
    }
    dedup_preserve_order(&mut imports);
    Ok(imports)
}

fn cffi_module_for_type(cffi_type: &str) -> String {
    match cffi_type {
        "CffiInnerObserverReceiver" => "centconf.observer_cffi.ior_cffi_traits".to_string(),
        _ => format!(
            "centconf.observer_cffi.{}_cffi_traits",
            cffi_type.trim_start_matches("Cffi").to_snake_case()
        ),
    }
}

fn cffi_struct_schema(
    trait_schema: &TraitSchema,
    supertrait_schemas: &HashMap<String, TraitSchema>,
) -> Result<PyClassSchema> {
    let generic_py_types = trait_generic_py_types(trait_schema)?;
    let cffi_func_types: Vec<(String, String)> = trait_schema
        .functions
        .iter()
        .map(|func| Ok((format!("{}_fut", func.name), trait_func_to_cffi_type(func)?)))
        .collect::<Result<Vec<_>>>()?;

    let supertrait_fields: Vec<String> = trait_schema
        .supertraits
        .iter()
        .filter(|st| (st.ty != "Send") && (st.ty != "Sync"))
        .map(|st| {
            let _ = supertrait_schemas.get(&st.codegen(0)).context(format!(
                "Missing supertrait schema for {} while generating cffi struct",
                st
            ))?;
            Ok(format!(
                "(\"{field}\", Cffi{st})",
                field = format!("cffi_{}", st.ty.to_snake_case()),
                st = st.ty
            ))
        })
        .collect::<Result<Vec<_>>>()?;

    let mut class_vars: Vec<String> = cffi_func_types
        .iter()
        .map(|(name, ty)| format!("_{}_type = {}", name, ty))
        .collect();
    class_vars.push(format!(
        "_fields_ = [{}]",
        std::iter::once("('self_ptr', ctypes.c_void_p)".to_string())
            .chain(supertrait_fields.into_iter())
            .chain(
                cffi_func_types
                    .iter()
                    // TODO: Currently assuming all functions are async
                    .map(|(name, _ty)| format!("('{}', {})", name, format!("_{}_type", name)))
            )
            .collect::<Vec<String>>()
            .join(",\n    ")
    ));

    let functions = trait_schema
        .functions
        .iter()
        .filter(|f| {
            f.annotations
                .as_ref()
                .map(|a| !a.cffi_impl_no_op)
                .unwrap_or(true)
        })
        .map(trait_fn_to_py_impl)
        .collect::<Result<Vec<_>>>()?;
    let supertrait_passthroughs = trait_schema
        .supertraits
        .iter()
        .filter(|st| (st.ty != "Send") && (st.ty != "Sync"))
        .map(|st| {
            let st_schema = supertrait_schemas
                .get(&st.codegen(0))
                .with_context(|| format!("Missing supertrait schema for {}", st))?;
            st_schema
                .functions
                .iter()
                .map(|f| passthrough_to_supertrait_schema(f, st))
                .collect::<Result<Vec<_>>>()
        })
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();

    Ok(PyClassSchema {
        name: format!("Cffi{}", trait_schema.name),
        parent_types: vec![
            PyTypeSchema {
                ty: "ctypes.Structure".to_string(),
                generic_ty_args: vec![],
            },
            PyTypeSchema {
                ty: format!("I{}", trait_schema.name),
                generic_ty_args: generic_py_types,
            },
        ],
        class_vars,
        functions: {
            let mut funcs = functions;
            funcs.extend(supertrait_passthroughs);
            funcs
        },
    })
}

fn trait_fn_to_py_impl(f: &FunctionSchema) -> Result<PyFunctionSchema> {
    Ok(PyFunctionSchema {
        name: f.name.clone(),
        args: f
            .args
            .iter()
            .map(|a| {
                Ok(PyFunctionArgSchema {
                    name: a.name.clone(),
                    ty: rs_ty_arg_to_py_ty(a)?,
                })
            })
            .collect::<Result<Vec<_>>>()?,
        return_type: rs_return_ty_to_py_ty(&f.return_type)?,
        is_async: true,
        body: trait_fn_to_py_impl_body(f)?,
        py_annotations: vec!["@override".to_string()],
    })
}

fn trait_fn_to_py_impl_body(function: &FunctionSchema) -> Result<Vec<String>> {
    fn arg_name_transform(arg: &FunctionArgSchema) -> String {
        if arg.ty.is_none() {
            "self_ptr".to_string()
        } else {
            arg.name.clone()
        }
    }

    let arg_transforms = function
        .args
        .iter()
        .map(|arg| {
            let py_input_ty = rs_ty_arg_to_py_ty(arg)?;
            let py_cffi_ty = arg_to_py_cffi_type(arg)?;

            let arg_name = arg_name_transform(arg);

            let reg = Handlebars::new();
            let mut transform_block = format!("{} = {{{{{{next}}}}}}\n", arg_name);
            match py_cffi_ty.as_str() {
                "CffiPointerBuffer" => {
                    if let Some(py_input_ty) = py_input_ty {
                        if py_input_ty.ty != "list" {
                            bail!(
                                "Expected list input type for CffiPointerBuffer transform, got {}",
                                py_input_ty.ty
                            );
                        }
                    } else {
                        bail!("Expected input type to be present for CffiPointerBuffer transform");
                    }

                    let template_string = format!(
                        "CffiPointerBuffer.from_pointer_array([\n    typing.cast(ctypes.c_void_p, {}) if {} is not None else None\n    for {} in {}\n])",
                        arg_name, arg_name, arg_name, arg.name
                    );
                    transform_block = reg
                            .render_template(&transform_block, &json!({"next": template_string}))
                            .unwrap()
                            .to_string();
                }
                cffi_ty if cffi_ty.starts_with("Cffi") => {
                    let generator_name = format!("{}Generator", cffi_ty);
                    let generator_field = format!(
                        "cffi_{}",
                        cffi_ty
                            .trim_start_matches("Cffi")
                            .to_snake_case()
                    );
                    let template_string = format!(
                        "{generator}(ASYNC_CFFI_LIB, asyncio.get_event_loop(), {arg}).{field}",
                        generator = generator_name,
                        arg = arg.name,
                        field = generator_field
                    );
                    transform_block = reg
                        .render_template(&transform_block, &json!({"next": template_string}))
                        .unwrap()
                        .to_string();
                }
                "ctypes.c_void_p" => {
                    if let Some(_) = py_input_ty {
                        return Ok(None);
                    }
                    // self
                    transform_block = reg
                            .render_template(&transform_block, &json!({"next": "self.self_ptr"}))
                            .unwrap()
                            .to_string();
                }
                _ => return Ok(None),
            }

            Ok(Some(transform_block))
        })
        .collect::<Result<Vec<_>>>()?;
    let arg_transforms: Vec<String> = arg_transforms.iter().flatten().cloned().collect();

    let async_args_for_call = function
        .args
        .iter()
        .map(arg_name_transform)
        .collect::<Vec<String>>()
        .join(", ");
    let async_call = format!(
        "await CffiFutureDriver.from_cffi_fut(\n    self.{}_fut({}),\n    asyncio.get_event_loop(),\n    ASYNC_CFFI_LIB)",
        function.name, async_args_for_call,
    );

    let mut transforms = arg_transforms;
    match return_kind(&function.return_type)? {
        ReturnKind::Unit => transforms.push(async_call),
        ReturnKind::Option => {
            transforms.push(format!("fut_result = {}", async_call));
            transforms.push(
                "ptr_ptr = ctypes.cast(fut_result, ctypes.POINTER(ctypes.c_void_p))".to_string(),
            );
            transforms.push("ptr = ctypes.c_void_p(ptr_ptr.contents.value)".to_string());
            transforms.push("if ptr:".to_string());
            transforms.push("    return ptr".to_string());
            transforms.push("else:".to_string());
            transforms.push("    return None".to_string());
        }
        ReturnKind::Other => transforms.push(format!("return {}", async_call)),
    }
    Ok(transforms)
}

fn cffi_generator_schema(
    trait_schema: &TraitSchema,
    supertrait_schemas: &HashMap<String, TraitSchema>,
) -> Result<PyClassSchema> {
    let py_impl_field = format!("_py_{}", trait_schema.name.to_snake_case());

    let boxed_functions = trait_schema
        .functions
        .iter()
        .filter(|f| !is_cffi_impl_no_op(f))
        .map(boxed_function_schema)
        .collect::<Result<Vec<_>>>()?;

    let init_fn = generator_init_fn_schema(trait_schema, supertrait_schemas)?;

    let passthroughs = trait_schema
        .functions
        .iter()
        .map(|f| passthrough_function_schema(f, &py_impl_field))
        .collect::<Result<Vec<_>>>()?;

    let supertrait_passthroughs = trait_schema
        .supertraits
        .iter()
        .filter(|st| (st.ty != "Send") && (st.ty != "Sync"))
        .map(|st| {
            let st_schema = supertrait_schemas
                .get(&st.codegen(0))
                .with_context(|| format!("Missing supertrait schema for {}", st))?;
            st_schema
                .functions
                .iter()
                .map(|f| passthrough_to_supertrait_schema(f, st))
                .collect::<Result<Vec<_>>>()
        })
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();

    let mut functions = Vec::new();
    functions.push(init_fn);
    functions.extend(passthroughs);
    functions.extend(supertrait_passthroughs);
    functions.extend(boxed_functions);

    Ok(PyClassSchema {
        name: format!("Cffi{}Generator", trait_schema.name),
        parent_types: vec![],
        class_vars: vec![],
        functions,
    })
}

fn generator_init_fn_schema(
    trait_schema: &TraitSchema,
    supertrait_schemas: &HashMap<String, TraitSchema>,
) -> Result<PyFunctionSchema> {
    let class_field = format!("cffi_{}", trait_schema.name.to_snake_case());
    let cffi_struct_name = format!("Cffi{}", trait_schema.name);
    let generic_py_types = trait_generic_py_types(trait_schema)?;
    let args = vec![
        PyFunctionArgSchema {
            name: "self".to_string(),
            ty: None,
        },
        PyFunctionArgSchema {
            name: "async_cffi".to_string(),
            ty: Some(PyTypeSchema {
                ty: "AsyncCffiLib".to_string(),
                generic_ty_args: vec![],
            }),
        },
        PyFunctionArgSchema {
            name: "loop".to_string(),
            ty: Some(PyTypeSchema {
                ty: "asyncio.AbstractEventLoop".to_string(),
                generic_ty_args: vec![],
            }),
        },
        PyFunctionArgSchema {
            name: trait_schema.name.to_snake_case(),
            ty: Some(PyTypeSchema {
                ty: format!("I{}", trait_schema.name),
                generic_ty_args: generic_py_types,
            }),
        },
    ];

    let mut field_setters = vec![
        "self._async_cffi = async_cffi".to_string(),
        "self._loop = loop".to_string(),
        format!(
            "self.{} = {}",
            format!("_py_{}", trait_schema.name.to_snake_case()),
            trait_schema.name.to_snake_case()
        ),
    ];

    let supertrait_field_setters = trait_schema
        .supertraits
        .iter()
        .filter(|st| (st.ty != "Send") && (st.ty != "Sync"))
        .map(|st| {
            let _ = supertrait_schemas.get(&st.codegen(0)).context(format!(
                "Missing supertrait schema for {} while generating init",
                st
            ))?;
            Ok(format!(
                "self.{field} = Cffi{st}Generator(async_cffi, loop, {py_impl}).cffi_{snake}",
                field = format!("cffi_{}", st.ty.to_snake_case()),
                st = st,
                py_impl = trait_schema.name.to_snake_case(),
                snake = st.ty.to_snake_case()
            ))
        })
        .collect::<Result<Vec<_>>>()?;

    field_setters.extend(supertrait_field_setters.into_iter());

    let fut_fields = trait_schema
        .functions
        .iter()
        .map(|f| {
            let fut_type_name = format!("{cffi_struct_name}._{}_fut_type", f.name);
            let lambda_args: Vec<String> = f
                .args
                .iter()
                .map(|arg| {
                    arg.ty
                        .as_ref()
                        .map(|_| arg.name.clone())
                        .unwrap_or("_self_ptr".to_string())
                })
                .collect();
            let lambda_args = lambda_args.join(", ");
            let boxed_call = format!(
                "{}",
                if is_cffi_impl_no_op(f) {
                    format!("self.{}()", f.name)
                } else {
                    format!(
                        "self._{}_boxed({})",
                        f.name,
                        f.args
                            .iter()
                            .filter(|a| a.ty.is_some())
                            .map(|a| a.name.clone())
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                }
            );
            format!(
                "{}_fut={}(\n    lambda {}: PyCffiFuture(\n        self._async_cffi,\n        self._loop,\n        lambda: {}\n    ).cffi_fut_ptr\n)",
                f.name, fut_type_name, lambda_args, boxed_call
            )
        })
        .collect::<Vec<_>>();

    let mut cffi_struct_args = vec!["self_ptr=ctypes.c_void_p()".to_string()];
    cffi_struct_args.extend(
        trait_schema
            .supertraits
            .iter()
            .filter(|st| (st.ty != "Send") && (st.ty != "Sync"))
            .map(|st| {
                format!(
                    "{field}=self.{field}",
                    field = format!("cffi_{}", st.ty.to_snake_case())
                )
            }),
    );
    cffi_struct_args.extend(fut_fields);

    let reg = Handlebars::new();
    let cffi_struct_init = reg
        .render_template(
            "self.{{class_field}} = {{cffi_struct_name}}(\n{{{fields}}}\n)",
            &json!({
                "class_field": class_field,
                "cffi_struct_name": cffi_struct_name,
                "fields": cffi_struct_args
                    .iter()
                    .map(|f| format!("    {}", f))
                    .collect::<Vec<_>>()
                    .join(",\n"),
            }),
        )
        .unwrap();

    field_setters.push(cffi_struct_init);

    Ok(PyFunctionSchema {
        name: "__init__".to_string(),
        args,
        return_type: None,
        is_async: false,
        body: field_setters,
        py_annotations: vec![],
    })
}

fn passthrough_function_schema(
    function: &FunctionSchema,
    py_impl_field: &str,
) -> Result<PyFunctionSchema> {
    let is_no_op = is_cffi_impl_no_op(function);

    let arg_schemas = if is_no_op {
        vec![PyFunctionArgSchema {
            name: "self".to_string(),
            ty: None,
        }]
    } else {
        std::iter::once(PyFunctionArgSchema {
            name: "self".to_string(),
            ty: None,
        })
        .chain(
            function
                .args
                .iter()
                .filter(|a| a.ty.is_some())
                .map(|a| {
                    Ok(PyFunctionArgSchema {
                        name: a.name.clone(),
                        ty: rs_ty_arg_to_py_ty(a)?,
                    })
                })
                .collect::<Result<Vec<_>>>()?
                .into_iter(),
        )
        .collect()
    };

    Ok(PyFunctionSchema {
        name: function.name.clone(),
        args: arg_schemas,
        return_type: None,
        is_async: true,
        body: if is_no_op {
            vec!["return self._async_cffi.box_u64(ctypes.c_uint64(1))".to_string()]
        } else {
            vec![format!(
                "return await self.{}.{}({})",
                py_impl_field,
                function.name,
                function
                    .args
                    .iter()
                    .filter(|a| a.ty.is_some())
                    .map(|a| a.name.clone())
                    .collect::<Vec<_>>()
                    .join(", ")
            )]
        },
        py_annotations: vec![],
    })
}

fn passthrough_to_supertrait_schema(
    function: &FunctionSchema,
    supertrait: &TypeSchema,
) -> Result<PyFunctionSchema> {
    let args = std::iter::once(PyFunctionArgSchema {
        name: "self".to_string(),
        ty: None,
    })
    .chain(
        function
            .args
            .iter()
            .filter(|a| a.ty.is_some())
            .map(|a| {
                Ok(PyFunctionArgSchema {
                    name: a.name.clone(),
                    ty: rs_ty_arg_to_py_ty(a)?,
                })
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter(),
    )
    .collect::<Vec<_>>();

    Ok(PyFunctionSchema {
        name: function.name.clone(),
        args,
        return_type: rs_return_ty_to_py_ty(&function.return_type)?,
        is_async: true,
        body: vec![format!(
            "return await self.cffi_{field}.{func}({args})",
            field = supertrait.ty.to_snake_case(),
            func = function.name,
            args = function
                .args
                .iter()
                .filter(|a| a.ty.is_some())
                .map(|a| a.name.clone())
                .collect::<Vec<_>>()
                .join(", ")
        )],
        py_annotations: vec![],
    })
}

fn boxed_function_schema(function: &FunctionSchema) -> Result<PyFunctionSchema> {
    if is_cffi_impl_no_op(function) {
        bail!("boxed_function_schema should not be called for cffi_impl_no_op functions");
    }

    let args = function
        .args
        .iter()
        .filter(|a| a.ty.is_some())
        .map(|a| {
            Ok(PyFunctionArgSchema {
                name: a.name.clone(),
                ty: Some(PyTypeSchema {
                    ty: arg_to_py_cffi_type(a)?,
                    generic_ty_args: vec![],
                }),
            })
        })
        .collect::<Result<Vec<_>>>()?;

    let mut body = Vec::new();
    let arg_transforms = function
        .args
        .iter()
        .filter(|a| a.ty.is_some())
        .map(boxed_arg_transform)
        .collect::<Result<Vec<_>>>()?;

    for transform in arg_transforms.into_iter().flatten() {
        body.push(transform);
    }

    let call_args = function
        .args
        .iter()
        .filter(|a| a.ty.is_some())
        .map(|a| a.name.clone())
        .collect::<Vec<_>>()
        .join(", ");
    let call_expr = format!("await self.{}({})", function.name, call_args);

    let return_transform =
        boxed_return_transform(&function.return_type, call_expr, function.name.as_str())?;
    for stmt in return_transform {
        body.push(stmt);
    }

    Ok(PyFunctionSchema {
        name: format!("_{}_boxed", function.name),
        args: std::iter::once(PyFunctionArgSchema {
            name: "self".to_string(),
            ty: None,
        })
        .chain(args.into_iter())
        .collect(),
        return_type: Some(PyTypeSchema {
            ty: "ctypes.c_void_p".to_string(),
            generic_ty_args: vec![],
        }),
        is_async: true,
        body,
        py_annotations: vec![],
    })
}

fn boxed_arg_transform(arg: &FunctionArgSchema) -> Result<Option<String>> {
    let cffi_stack = arg_cffi_type_stack(arg)?;
    if let Some(explicit) = &cffi_stack[0].explicit_type {
        if explicit == "CffiPointerBuffer" {
            let item_var = format!("{}_item", arg.name);
            let transform = format!(
                "{arg}_array = {arg}.to_pointer_array()\n{arg}_py_list: list[Optional[object]] = []\nfor {item_var} in {arg}_array:\n    opt_{item_var} = {item_var}\n    if (\n        {item_var} is None\n        or {item_var} == 0\n        or (isinstance({item_var}, ctypes.c_void_p) and {item_var}.value == 0)\n    ):\n        opt_{item_var} = None\n    {arg}_py_list.append(opt_{item_var})\n{arg} = {arg}_py_list",
                arg = arg.name,
                item_var = item_var,
            );
            return Ok(Some(transform));
        }
    }

    if cffi_stack[0].is_optional || cffi_stack.iter().any(|e| e.is_optional) {
        let transform = format!(
            "{arg} = None if {arg} is None or {arg} == 0 or (isinstance({arg}, ctypes.c_void_p) and {arg}.value == 0) else {arg}",
            arg = arg.name
        );
        return Ok(Some(transform));
    }

    Ok(None)
}

fn boxed_return_transform(
    ret_ty: &TypeSchema,
    call_expr: String,
    function_name: &str,
) -> Result<Vec<String>> {
    let mut stmts = Vec::new();
    let inner_ty = inner_return_type(ret_ty)?;

    if inner_ty.ty == "()" {
        stmts.push(call_expr);
        let unit_box_value = if function_name == "hold_strong_publisher_ref" {
            1
        } else {
            42
        };
        stmts.push(format!(
            "return self._async_cffi.box_u64(ctypes.c_uint64({}))",
            unit_box_value
        ));
        return Ok(stmts);
    }

    if inner_ty.ty == "Option" {
        stmts.push(format!("result = {}", call_expr));
        stmts.push("if result is None:".to_string());
        stmts.push("    ptr = ctypes.c_void_p()".to_string());
        stmts.push("else:".to_string());
        stmts.push("    ptr = typing.cast(ctypes.c_void_p, result)".to_string());
        stmts.push("return self._async_cffi.box_ptr(ptr)".to_string());
        return Ok(stmts);
    }

    stmts.push(format!("result = {}", call_expr));
    stmts.push("ptr = typing.cast(ctypes.c_void_p, result)".to_string());
    stmts.push("return self._async_cffi.box_ptr(ptr)".to_string());
    Ok(stmts)
}

fn is_cffi_impl_no_op(function: &FunctionSchema) -> bool {
    function
        .annotations
        .as_ref()
        .map(|a| a.cffi_impl_no_op)
        .unwrap_or(false)
}

enum ReturnKind {
    Unit,
    Option,
    Other,
}

fn return_kind(ret_ty: &TypeSchema) -> Result<ReturnKind> {
    let inner = inner_return_type(ret_ty)?;
    if inner.ty == "()" {
        Ok(ReturnKind::Unit)
    } else if inner.ty == "Option" {
        Ok(ReturnKind::Option)
    } else {
        Ok(ReturnKind::Other)
    }
}

fn inner_return_type(ret_ty: &TypeSchema) -> Result<TypeSchema> {
    if ret_ty.ty != "BoxFuture" {
        return Ok(ret_ty.clone());
    }

    if ret_ty.generic_ty_args.len() != 2 {
        bail!(
            "BoxFuture should have exactly two generic arguments, but found {}",
            ret_ty.generic_ty_args.len()
        );
    }

    let inner = &ret_ty.generic_ty_args[1];
    Ok(inner.clone())
}

fn py_object_type() -> PyTypeSchema {
    PyTypeSchema {
        ty: "object".to_string(),
        generic_ty_args: vec![],
    }
}

fn py_ctypes_void_ptr_type() -> PyTypeSchema {
    PyTypeSchema {
        ty: "ctypes.c_void_p".to_string(),
        generic_ty_args: vec![],
    }
}

fn trait_generic_py_types(trait_schema: &TraitSchema) -> Result<Vec<PyTypeSchema>> {
    trait_schema
        .generics
        .iter()
        .map(|g| {
            let cffi_type = g
                .annotations
                .as_ref()
                .and_then(|a| a.cffi_type.as_ref())
                .context("Missing cffi_type annotation for generic parameter")?;
            py_ty_from_cffi_schema(&cffi_type_str_to_schema(cffi_type)?)
        })
        .collect()
}

fn py_ty_from_cffi_schema(schema: &CffiTypeSchema) -> Result<PyTypeSchema> {
    let elem = cffi_schema_to_elem_spec(schema);

    if elem.explicit_type.is_some() {
        return cffi_ty_schema_to_py_native_ty(schema);
    }

    let base = if elem.is_pointer {
        py_ctypes_void_ptr_type()
    } else {
        py_object_type()
    };

    Ok(if elem.is_optional {
        PyTypeSchema {
            ty: "Optional".to_string(),
            generic_ty_args: vec![base],
        }
    } else {
        base
    })
}

fn rs_return_ty_to_py_ty(ret_ty: &TypeSchema) -> Result<Option<PyTypeSchema>> {
    let inner_ty = inner_return_type(ret_ty)?;

    if inner_ty.ty == "()" {
        return Ok(None);
    }

    if inner_ty.ty == "Option" && inner_ty.generic_ty_args.len() > 0 {
        let inner_arg = &inner_ty.generic_ty_args[0];
        if inner_arg.ty == "Arc" && inner_arg.generic_ty_args.len() > 0 {
            return Ok(Some(PyTypeSchema {
                ty: "Optional".to_string(),
                generic_ty_args: vec![py_ctypes_void_ptr_type()],
            }));
        }
    }
    if inner_ty.ty == "Option" && inner_ty.generic_ty_args.len() > 0 {
        return Ok(Some(PyTypeSchema {
            ty: "Optional".to_string(),
            generic_ty_args: vec![py_object_type()],
        }));
    }

    Ok(Some(py_object_type()))
}

fn rs_ty_arg_to_py_ty(arg: &FunctionArgSchema) -> Result<Option<PyTypeSchema>> {
    let annotation_ty = if let Some(annotations) = &arg.annotations {
        if let Some(cffi_type) = &annotations.cffi_type {
            Some(py_ty_from_cffi_schema(&cffi_type_str_to_schema(
                cffi_type,
            )?)?)
        } else {
            None
        }
    } else {
        None
    };

    if let Some(ty) = arg.ty.as_ref() {
        if let Some(interface_ty) = interface_names_from_ty(ty).into_iter().next() {
            return Ok(Some(PyTypeSchema {
                ty: interface_ty,
                generic_ty_args: vec![],
            }));
        }

        if ty.ty == "Option" && ty.generic_ty_args.len() > 0 {
            return Ok(Some(PyTypeSchema {
                ty: "Optional".to_string(),
                generic_ty_args: vec![py_object_type()],
            }));
        }

        if ty.ty == "Vec" && ty.generic_ty_args.len() > 0 {
            let inner_arg = &ty.generic_ty_args[0];
            if inner_arg.ty == "Option" && inner_arg.generic_ty_args.len() > 0 {
                return Ok(Some(PyTypeSchema {
                    ty: "list".to_string(),
                    generic_ty_args: vec![PyTypeSchema {
                        ty: "Optional".to_string(),
                        generic_ty_args: vec![py_object_type()],
                    }],
                }));
            }
        }

        if ty.ty == "Vec" && ty.generic_ty_args.len() > 0 {
            return Ok(Some(PyTypeSchema {
                ty: "list".to_string(),
                generic_ty_args: vec![py_object_type()],
            }));
        }

        if is_integer_ty(&ty) {
            return Ok(Some(PyTypeSchema {
                ty: "int".to_string(),
                generic_ty_args: vec![],
            }));
        }

        if let Some(annotation_ty) = annotation_ty {
            return Ok(Some(annotation_ty));
        }

        Ok(Some(py_object_type()))
    } else {
        Ok(annotation_ty)
    }
}

fn arg_cffi_type_stack(arg: &FunctionArgSchema) -> Result<Vec<CffiTypeElementSpec>> {
    if let Some(annotations) = &arg.annotations {
        if let Some(cffi_type) = &annotations.cffi_type {
            return Ok(cffi_type_str_to_type_stack(cffi_type)?);
        }
    }

    fn_arg_to_async_cffi_type_spec(arg)
}

fn cffi_schema_to_elem_spec(schema: &CffiTypeSchema) -> CffiTypeElementSpec {
    match schema.ty.as_str() {
        "opt_ptr" => CffiTypeElementSpec {
            is_pointer: true,
            is_optional: true,
            explicit_type: None,
        },
        "ptr" => CffiTypeElementSpec {
            is_pointer: true,
            is_optional: false,
            explicit_type: None,
        },
        "CffiPointerBuffer" => CffiTypeElementSpec {
            is_pointer: true,
            is_optional: false,
            explicit_type: Some("CffiPointerBuffer".to_string()),
        },
        ty => CffiTypeElementSpec {
            is_pointer: false,
            is_optional: false,
            explicit_type: if ty.starts_with("Cffi") {
                Some(ty.to_string())
            } else {
                None
            },
        },
    }
}

fn cffi_ty_schema_to_py_native_ty(schema: &CffiTypeSchema) -> Result<PyTypeSchema> {
    build_py_type_from_schema(schema)
}

fn build_py_type_from_schema(schema: &CffiTypeSchema) -> Result<PyTypeSchema> {
    let elem = cffi_schema_to_elem_spec(schema);

    let mut ty = PyTypeSchema {
        ty: cffi_ty_schema_elem_to_py_native_ty(&elem)?,
        generic_ty_args: Vec::new(),
    };

    let next_schema = match schema.ty.as_str() {
        "opt_ptr" | "ptr" | "CffiPointerBuffer" => schema.generic_ty_args.get(0),
        _ => schema.generic_ty_args.get(0),
    };

    if let Some(inner_schema) = next_schema {
        ty.generic_ty_args
            .push(build_py_type_from_schema(inner_schema)?);
    }

    Ok(ty)
}

fn cffi_ty_schema_elem_to_py_native_ty(elem: &CffiTypeElementSpec) -> Result<String> {
    Ok(if let Some(explicit_ty) = &elem.explicit_type {
        match explicit_ty.as_str() {
            "CffiPointerBuffer" => "list".to_string(),
            _ => "object".to_string(),
        }
    } else if elem.is_optional {
        "Optional".to_string()
    } else {
        "object".to_string()
    })
}

fn trait_func_to_cffi_type(func: &FunctionSchema) -> Result<String> {
    let arg_types: Vec<String> = func
        .args
        .iter()
        .map(arg_to_py_cffi_type)
        .collect::<Result<_>>()?;
    let ret_type = return_type_to_py_cffi_type(&Some(func.return_type.clone()));
    Ok(format!(
        "ctypes.CFUNCTYPE({}, {})",
        ret_type,
        arg_types.join(", ")
    ))
}

fn arg_to_py_cffi_type(arg: &FunctionArgSchema) -> Result<String> {
    if let Some(annotations) = &arg.annotations {
        if let Some(cffi_type) = &annotations.cffi_type {
            let stack = cffi_type_str_to_type_stack(cffi_type)?;
            if let Some(explicit) = &stack[0].explicit_type {
                return Ok(explicit.clone());
            }
            if stack[0].is_pointer {
                return Ok("ctypes.c_void_p".to_string());
            }
        }
    }

    match arg.ty.as_ref() {
        // self_ptr
        None => Ok("ctypes.c_void_p".to_string()),
        Some(ty) => {
            if ty.ty == "Vec" && ty.generic_ty_args.len() > 0 {
                // TODO: Currently assuming Vecs always map to CffiPointerBuffer
                Ok("CffiPointerBuffer".to_string())
            } else if let Some(ctypes_ty) = ctypes_for_integer(&ty) {
                Ok(ctypes_ty.to_string())
            } else {
                Ok("ctypes.c_void_p".to_string())
            }
        }
    }
}

fn is_integer_ty(ty: &TypeSchema) -> bool {
    matches!(
        ty.ty.as_str(),
        "usize" | "isize" | "u64" | "i64" | "u32" | "i32" | "u16" | "i16" | "u8" | "i8"
    )
}

fn ctypes_for_integer(ty: &TypeSchema) -> Option<&'static str> {
    match ty.ty.as_str() {
        "usize" => Some("ctypes.c_ulong"),
        "isize" => Some("ctypes.c_long"),
        _ => None,
    }
}

fn return_type_to_py_cffi_type(ret_type: &Option<TypeSchema>) -> String {
    match ret_type {
        None => "None".to_string(),
        Some(_) => "ctypes.c_void_p".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    use rs_schema::{
        FnArgAnnotations, FunctionAnnotations, FunctionArgSchema, FunctionSchema, TraitSchema,
    };

    fn inner_observer_trait_schema() -> TraitSchema {
        let mut pointer_buffer_annotations = FnArgAnnotations::new();
        pointer_buffer_annotations.cffi_type = Some("CffiPointerBuffer<opt_ptr<T>>".to_string());

        let mut hold_strong_annotations = FunctionAnnotations::new();
        hold_strong_annotations.cffi_impl_no_op = true;

        TraitSchema {
            name: "InnerObserverReceiver".to_string(),
            functions: vec![
                FunctionSchema {
                    name: "update".to_string(),
                    args: vec![
                        FunctionArgSchema {
                            name: "self".to_string(),
                            ty: None,
                            annotations: None,
                        },
                        FunctionArgSchema {
                            name: "data".to_string(),
                            ty: Some(TypeSchema {
                                ty: "Vec".to_string(),
                                generic_ty_args: vec![TypeSchema {
                                    ty: "Option".to_string(),
                                    generic_ty_args: vec![TypeSchema {
                                        ty: "AnyArc".to_string(),
                                        generic_ty_args: vec![],
                                    }],
                                }],
                            }),
                            annotations: Some(pointer_buffer_annotations),
                        },
                    ],
                    body: None,
                    return_type: TypeSchema {
                        ty: "BoxFuture".to_string(),
                        generic_ty_args: vec![
                            TypeSchema {
                                ty: "'_".to_string(),
                                generic_ty_args: vec![],
                            },
                            TypeSchema {
                                ty: "()".to_string(),
                                generic_ty_args: vec![],
                            },
                        ],
                    },
                    extern_layout: None,
                    annotations: None,
                },
                FunctionSchema {
                    name: "hold_strong_publisher_ref".to_string(),
                    args: vec![
                        FunctionArgSchema {
                            name: "self".to_string(),
                            ty: None,
                            annotations: None,
                        },
                        FunctionArgSchema {
                            name: "publisher".to_string(),
                            ty: Some(TypeSchema {
                                ty: "Arc".to_string(),
                                generic_ty_args: vec![TypeSchema {
                                    ty: "dyn Publisher + Send + Sync".to_string(),
                                    generic_ty_args: vec![],
                                }],
                            }),
                            annotations: None,
                        },
                    ],
                    body: None,
                    return_type: TypeSchema {
                        ty: "BoxFuture".to_string(),
                        generic_ty_args: vec![
                            TypeSchema {
                                ty: "'_".to_string(),
                                generic_ty_args: vec![],
                            },
                            TypeSchema {
                                ty: "()".to_string(),
                                generic_ty_args: vec![],
                            },
                        ],
                    },
                    extern_layout: None,
                    annotations: Some(hold_strong_annotations),
                },
            ],
            generics: vec![],
            supertraits: vec![],
        }
    }

    fn inner_observer_py_codegen() -> String {
        trait_to_async_cffi_schema(&inner_observer_trait_schema(), &HashMap::new())
            .expect("should generate schema")
            .codegen(4)
    }

    fn publisher_trait_schema() -> TraitSchema {
        let mut inner_observer_annotations = FnArgAnnotations::new();
        inner_observer_annotations.cffi_type = Some("CffiInnerObserverReceiver".to_string());

        TraitSchema {
            name: "Publisher".to_string(),
            functions: vec![
                FunctionSchema {
                    name: "add_observer".to_string(),
                    args: vec![
                        FunctionArgSchema {
                            name: "self".to_string(),
                            ty: None,
                            annotations: None,
                        },
                        FunctionArgSchema {
                            name: "observer".to_string(),
                            ty: Some(TypeSchema {
                                ty: "Box".to_string(),
                                generic_ty_args: vec![TypeSchema {
                                    ty: "dyn InnerObserverReceiver".to_string(),
                                    generic_ty_args: vec![],
                                }],
                            }),
                            annotations: Some(inner_observer_annotations),
                        },
                        FunctionArgSchema {
                            name: "input_index".to_string(),
                            ty: Some(TypeSchema {
                                ty: "usize".to_string(),
                                generic_ty_args: vec![],
                            }),
                            annotations: None,
                        },
                    ],
                    body: None,
                    return_type: TypeSchema {
                        ty: "BoxFuture".to_string(),
                        generic_ty_args: vec![
                            TypeSchema {
                                ty: "'_".to_string(),
                                generic_ty_args: vec![],
                            },
                            TypeSchema {
                                ty: "Option".to_string(),
                                generic_ty_args: vec![TypeSchema::new_simple("AnyArc".to_string())],
                            },
                        ],
                    },
                    extern_layout: None,
                    annotations: None,
                },
                FunctionSchema {
                    name: "notify".to_string(),
                    args: vec![
                        FunctionArgSchema {
                            name: "self".to_string(),
                            ty: None,
                            annotations: None,
                        },
                        FunctionArgSchema {
                            name: "value".to_string(),
                            ty: Some(TypeSchema {
                                ty: "AnyArc".to_string(),
                                generic_ty_args: vec![],
                            }),
                            annotations: None,
                        },
                    ],
                    body: None,
                    return_type: TypeSchema {
                        ty: "BoxFuture".to_string(),
                        generic_ty_args: vec![
                            TypeSchema {
                                ty: "'_".to_string(),
                                generic_ty_args: vec![],
                            },
                            TypeSchema {
                                ty: "()".to_string(),
                                generic_ty_args: vec![],
                            },
                        ],
                    },
                    extern_layout: None,
                    annotations: None,
                },
            ],
            generics: vec![],
            supertraits: vec![],
        }
    }

    #[test]
    fn test_inner_observer_generator_and_boxed_returns() {
        let code = inner_observer_py_codegen();

        assert!(
            code.contains("class CffiInnerObserverReceiverGenerator"),
            "generator wrapper should be emitted"
        );
        assert!(
            code.contains("_update_boxed"),
            "boxed update helper should be emitted"
        );
        assert!(
            code.contains("return self._async_cffi.box_u64(ctypes.c_uint64(42))"),
            "void futures should be boxed to non-null pointers"
        );
    }

    #[test]
    fn test_inner_observer_pointer_buffer_handling() {
        let code = inner_observer_py_codegen();

        assert!(
            code.contains("CffiPointerBuffer.from_pointer_array(["),
            "update should wrap Python lists into CffiPointerBuffer"
        );
        assert!(
            code.contains("typing.cast(ctypes.c_void_p, data) if data is not None else None"),
            "pointer buffer transform should cast non-null entries"
        );
        assert!(
            code.contains("data.to_pointer_array()"),
            "boxed helper should expand pointer buffers back to Python lists"
        );
        assert!(
            code.contains("data_item is None") && code.contains("ctypes.c_void_p"),
            "boxed helper should treat null and void pointers as None"
        );
    }

    #[test]
    fn test_inner_observer_generator_uses_fut_type_fields() {
        let code = inner_observer_py_codegen();

        assert!(
            code.contains("CffiInnerObserverReceiver._update_fut_type"),
            "update fut type alias should be referenced"
        );
        assert!(
            code.contains("CffiInnerObserverReceiver._hold_strong_publisher_ref_fut_type"),
            "hold strong fut type alias should be referenced"
        );
    }

    #[test]
    fn test_inner_observer_passthrough_methods_include_self_and_method_calls() {
        let code = inner_observer_py_codegen();

        assert!(
            code.contains("async def update(self, data"),
            "passthrough update should include self parameter"
        );
        assert!(
            code.contains("return await self._py_inner_observer_receiver.update(data)"),
            "passthrough should invoke the underlying Python implementation"
        );
        assert!(
            code.contains("async def hold_strong_publisher_ref(self)"),
            "no-op passthrough should include only self parameter"
        );
        assert!(
            code.contains("return self._async_cffi.box_u64(ctypes.c_uint64(1))"),
            "no-op passthrough should return a boxed non-null pointer"
        );
    }

    #[test]
    fn test_inner_observer_boxed_unit_returns_use_non_null_pointer() {
        let code = inner_observer_py_codegen();

        assert!(
            code.contains("return self._async_cffi.box_u64(ctypes.c_uint64(42))"),
            "unit return futures should be boxed to a non-null pointer value"
        );
    }

    #[test]
    fn test_publisher_option_handling_and_observer_wrapping() {
        let code = trait_to_async_cffi_schema(&publisher_trait_schema(), &HashMap::new())
            .expect("publisher schema generation should succeed")
            .codegen(4);

        println!("publisher codegen:\n{}", code);

        assert!(
            code.contains("from n_observer.core import IPublisher, IInnerObserverReceiver"),
            "interface imports should include publisher and dependencies"
        );
        assert!(
            code.contains("CffiInnerObserverReceiverGenerator"),
            "publisher bindings should import inner observer receiver generator"
        );
        assert!(
            code.contains(
                "observer = CffiInnerObserverReceiverGenerator(ASYNC_CFFI_LIB, asyncio.get_event_loop(), observer).cffi_inner_observer_receiver"
            ),
            "add_observer should wrap python observers in cffi generators"
        );
        assert!(
            code.contains("ptr_ptr = ctypes.cast(fut_result, ctypes.POINTER(ctypes.c_void_p))"),
            "option returns should cast to pointer-to-pointer"
        );
        assert!(
            code.contains("-> Optional[object]"),
            "add_observer return signature should surface Optional typing"
        );
    }

    #[test]
    fn test_supertrait_interfaces_are_imported() {
        let mut schema = inner_observer_trait_schema();
        schema.name = "Observable".to_string();
        schema.supertraits = vec![
            TypeSchema::new_simple("Publisher".to_string()),
            TypeSchema::new_simple("InnerObserverReceiver".to_string()),
        ];

        let supertraits = HashMap::from([
            ("Publisher".to_string(), publisher_trait_schema()),
            (
                "InnerObserverReceiver".to_string(),
                inner_observer_trait_schema(),
            ),
        ]);

        let code = trait_to_async_cffi_schema(&schema, &supertraits)
            .expect("observable schema generation should succeed")
            .codegen(4);

        assert!(
            code.contains(
                "from n_observer.core import IObservable, IPublisher, IInnerObserverReceiver"
            ),
            "supertrait interface imports should be included for observable traits"
        );
    }

    #[test]
    fn test_arg_to_py_cffi_type_none() {
        let arg = FunctionArgSchema {
            name: "self_ptr".to_string(),
            ty: None,
            annotations: None,
        };
        assert_eq!(arg_to_py_cffi_type(&arg).unwrap(), "ctypes.c_void_p");
    }

    #[test]
    fn test_arg_to_py_cffi_type_vec() {
        let arg = FunctionArgSchema {
            name: "vals".to_string(),
            ty: Some(TypeSchema {
                ty: "Vec".to_string(),
                generic_ty_args: vec![TypeSchema {
                    ty: "i32".to_string(),
                    generic_ty_args: vec![],
                }],
            }),
            annotations: None,
        };
        assert_eq!(arg_to_py_cffi_type(&arg).unwrap(), "CffiPointerBuffer");
    }

    #[test]
    fn test_arg_to_py_cffi_type_other() {
        let arg = FunctionArgSchema {
            name: "x".to_string(),
            ty: Some(TypeSchema {
                ty: "i32".to_string(),
                generic_ty_args: vec![],
            }),
            annotations: None,
        };
        assert_eq!(arg_to_py_cffi_type(&arg).unwrap(), "ctypes.c_void_p");
    }

    #[test]
    fn test_arg_to_py_cffi_type_explicit_annotation() {
        let arg = FunctionArgSchema {
            name: "x".to_string(),
            ty: Some(TypeSchema {
                ty: "Box".to_string(),
                generic_ty_args: vec![TypeSchema {
                    ty: "dyn InnerObserverReceiver".to_string(),
                    generic_ty_args: vec![],
                }],
            }),
            annotations: Some({
                let mut annotations = FnArgAnnotations::new();
                annotations.cffi_type = Some("CffiInnerObserverReceiver".to_string());
                annotations
            }),
        };
        assert_eq!(
            arg_to_py_cffi_type(&arg).unwrap(),
            "CffiInnerObserverReceiver"
        );
    }

    #[test]
    fn test_return_type_to_py_cffi_type() {
        assert_eq!(return_type_to_py_cffi_type(&None), "None");
        assert_eq!(
            return_type_to_py_cffi_type(&Some(TypeSchema {
                ty: "i32".to_string(),
                generic_ty_args: vec![]
            })),
            "ctypes.c_void_p"
        );
    }

    #[test]
    fn test_trait_func_to_cffi_type_format() {
        let func = FunctionSchema {
            name: "foo".to_string(),
            args: vec![
                FunctionArgSchema {
                    name: "self_ptr".to_string(),
                    ty: None,
                    annotations: None,
                },
                FunctionArgSchema {
                    name: "vals".to_string(),
                    ty: Some(TypeSchema {
                        ty: "Vec".to_string(),
                        generic_ty_args: vec![TypeSchema {
                            ty: "u8".to_string(),
                            generic_ty_args: vec![],
                        }],
                    }),
                    annotations: None,
                },
            ],
            body: None,
            return_type: TypeSchema {
                ty: "()".to_string(),
                generic_ty_args: vec![],
            },
            extern_layout: None,
            annotations: None,
        };

        // return_type None -> "None" in our helper, but trait_func_to_cffi_type wraps it
        let got = trait_func_to_cffi_type(&func).unwrap();
        // Should contain the CFUNCTYPE and the two arg types
        assert!(got.starts_with("ctypes.CFUNCTYPE("));
        assert!(got.contains("ctypes.c_void_p"));
        assert!(got.contains("CffiPointerBuffer"));
    }

    #[test]
    fn test_trait_to_async_cffi_schema_generates_class() {
        let func = FunctionSchema {
            name: "update".to_string(),
            args: vec![FunctionArgSchema {
                name: "self_ptr".to_string(),
                ty: None,
                annotations: None,
            }],
            body: None,
            return_type: TypeSchema {
                ty: "()".to_string(),
                generic_ty_args: vec![],
            },
            extern_layout: None,
            annotations: None,
        };

        let trait_schema = TraitSchema {
            name: "MyTrait".to_string(),
            functions: vec![func],
            generics: vec![],
            supertraits: vec![],
        };

        let schema = trait_to_async_cffi_schema(&trait_schema, &HashMap::new())
            .expect("should generate schema");
        let code = schema.codegen(4);
        assert!(code.contains("import ctypes"));
        assert!(code.contains("class CffiMyTrait("));
        assert!(code.contains("_fields_"));
        assert!(code.contains("update"));
        assert!(code.contains("class CffiMyTraitGenerator"));
        assert!(code.contains("PyCffiFuture"));
        assert!(code.contains("_update_boxed"));
    }
}
