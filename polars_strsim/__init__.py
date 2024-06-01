from pathlib import Path
import polars as pl
from polars.plugins import register_plugin_function
from polars._typing import IntoExpr
from polars_strsim.utils import parse_into_expr


def levenshtein(expr: IntoExpr, other: IntoExpr) -> pl.Expr:
    expr = parse_into_expr(expr, dtype=pl.Utf8)
    other = parse_into_expr(other, dtype=pl.Utf8)
    return register_plugin_function(
        plugin_path=Path(__file__).parent,
        function_name="levenshtein",
        args=[expr, other],
        is_elementwise=True,
    )


def jaro(expr: IntoExpr, other: IntoExpr) -> pl.Expr:
    expr = parse_into_expr(expr, dtype=pl.Utf8)
    other = parse_into_expr(other, dtype=pl.Utf8)
    return register_plugin_function(
        plugin_path=Path(__file__).parent,
        function_name="jaro",
        args=[expr, other],
        is_elementwise=True,
    )


def jaro_winkler(expr: IntoExpr, other: IntoExpr) -> pl.Expr:
    expr = parse_into_expr(expr, dtype=pl.Utf8)
    other = parse_into_expr(other, dtype=pl.Utf8)
    return register_plugin_function(
        plugin_path=Path(__file__).parent,
        function_name="jaro_winkler",
        args=[expr, other],
        is_elementwise=True,
    )


def jaccard(expr: IntoExpr, other: IntoExpr) -> pl.Expr:
    expr = parse_into_expr(expr, dtype=pl.Utf8)
    other = parse_into_expr(other, dtype=pl.Utf8)
    return register_plugin_function(
        plugin_path=Path(__file__).parent,
        function_name="jaccard",
        args=[expr, other],
        is_elementwise=True,
    )


def sorensen_dice(expr: IntoExpr, other: IntoExpr) -> pl.Expr:
    expr = parse_into_expr(expr, dtype=pl.Utf8)
    other = parse_into_expr(other, dtype=pl.Utf8)
    return register_plugin_function(
        plugin_path=Path(__file__).parent,
        function_name="sorensen_dice",
        args=[expr, other],
        is_elementwise=True,
    )


__all__ = [
    "levenshtein",
    "jaro",
    "jaro_winkler",
    "jaccard",
    "sorensen_dice",
]
