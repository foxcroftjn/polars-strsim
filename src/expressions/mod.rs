#![allow(clippy::unused_unit)]
mod strsim;

use polars::prelude::*;
use pyo3_polars::derive::{polars_expr, CallerContext};
use strsim::SimilarityFunctionType;

#[polars_expr(output_type=Float64)]
fn levenshtein(inputs: &[Series], context: CallerContext) -> PolarsResult<Series> {
    strsim::parallel_apply(inputs, context, SimilarityFunctionType::Levenshtein)
}

#[polars_expr(output_type=Float64)]
fn jaro(inputs: &[Series], context: CallerContext) -> PolarsResult<Series> {
    strsim::parallel_apply(inputs, context, SimilarityFunctionType::Jaro)
}

#[polars_expr(output_type=Float64)]
fn jaro_winkler(inputs: &[Series], context: CallerContext) -> PolarsResult<Series> {
    strsim::parallel_apply(inputs, context, SimilarityFunctionType::JaroWinkler)
}

#[polars_expr(output_type=Float64)]
fn jaccard(inputs: &[Series], context: CallerContext) -> PolarsResult<Series> {
    strsim::parallel_apply(inputs, context, SimilarityFunctionType::Jaccard)
}

#[polars_expr(output_type=Float64)]
fn sorensen_dice(inputs: &[Series], context: CallerContext) -> PolarsResult<Series> {
    strsim::parallel_apply(inputs, context, SimilarityFunctionType::SorensenDice)
}
