// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Datafusion user defined functions

use arrow_array::ArrayRef;
use arrow_schema::DataType;
use datafusion_common::Result;
use datafusion_expr::registry::FunctionRegistry;
use datafusion_expr::{Volatility, create_udf};
use datafusion_functions::utils::make_scalar_function;
use std::sync::Arc;

pub use lance_datafusion_exec::udf::{CONTAINS_TOKENS_UDF, json};

mod element;

pub(crate) use element::rewrite_element_lambdas;
pub use element::{ANY_ELEMENT, any_element_predicate, plan_element_predicate};

/// Register UDF functions to datafusion context.
pub fn register_functions(ctx: &mut dyn FunctionRegistry) -> Result<()> {
    ctx.register_udf(Arc::new(CONTAINS_TOKENS_UDF.clone()))?;
    // JSON functions
    ctx.register_udf(Arc::new(json::json_extract_udf()))?;
    ctx.register_udf(Arc::new(json::json_extract_with_type_udf()))?;
    ctx.register_udf(Arc::new(json::json_exists_udf()))?;
    ctx.register_udf(Arc::new(json::json_get_udf()))?;
    ctx.register_udf(Arc::new(json::json_get_string_udf()))?;
    ctx.register_udf(Arc::new(json::json_get_int_udf()))?;
    ctx.register_udf(Arc::new(json::json_get_float_udf()))?;
    ctx.register_udf(Arc::new(json::json_get_bool_udf()))?;
    ctx.register_udf(Arc::new(json::json_array_contains_udf()))?;
    ctx.register_udf(Arc::new(json::json_array_length_udf()))?;
    // Predicates on the elements of a list<struct> column
    ctx.register_udf(Arc::new(element::any_element_udf()))?;
    register_geo_stub_functions(ctx)
}

/// When the `geo` feature is disabled, register stub UDFs for spatial SQL functions
/// so that users get a clear error mentioning the feature flag instead of
/// DataFusion's generic "Unknown function" error.
fn register_geo_stub_functions(ctx: &mut dyn FunctionRegistry) -> Result<()> {
    let geo_funcs = [
        "st_intersects",
        "st_contains",
        "st_within",
        "st_touches",
        "st_crosses",
        "st_overlaps",
        "st_covers",
        "st_coveredby",
        "st_distance",
        "st_area",
        "st_length",
    ];

    for name in geo_funcs {
        let func_name = name.to_string();
        let stub = Arc::new(make_scalar_function(
            move |_args: &[ArrayRef]| {
                Err(datafusion_common::DataFusionError::Plan(format!(
                    "Function '{}' requires geospatial support, which is disabled in this build.",
                    func_name
                )))
            },
            vec![],
        ));

        ctx.register_udf(Arc::new(create_udf(
            name,
            vec![DataType::Binary, DataType::Binary],
            DataType::Boolean,
            Volatility::Immutable,
            stub,
        )))?;
    }
    Ok(())
}
