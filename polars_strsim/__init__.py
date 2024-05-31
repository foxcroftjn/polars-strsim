from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from polars_strsim.utils import parse_into_expr, register_plugin, parse_version

if TYPE_CHECKING:
    from polars.type_aliases import IntoExpr

if parse_version(pl.__version__) < parse_version("0.20.16"):
    from polars.utils.udfs import _get_shared_lib_location

    lib: str | Path = _get_shared_lib_location(__file__)
else:
    lib = Path(__file__).parent

def levenshtein(expr: IntoExpr, other: IntoExpr) -> pl.Expr:
    expr = parse_into_expr(expr, dtype=pl.Utf8)
    other = parse_into_expr(other, dtype=pl.Utf8)
    return expr.register_plugin(
        lib=lib,
        args=[other],
        symbol="levenshtein",
        is_elementwise=True,
    )

def jaro(expr: IntoExpr, other: IntoExpr) -> pl.Expr:
    expr = parse_into_expr(expr, dtype=pl.Utf8)
    other = parse_into_expr(other, dtype=pl.Utf8)
    return expr.register_plugin(
        lib=lib,
        args=[other],
        symbol="jaro",
        is_elementwise=True,
    )

def jaro_winkler(expr: IntoExpr, other: IntoExpr) -> pl.Expr:
    expr = parse_into_expr(expr, dtype=pl.Utf8)
    other = parse_into_expr(other, dtype=pl.Utf8)
    return expr.register_plugin(
        lib=lib,
        args=[other],
        symbol="jaro_winkler",
        is_elementwise=True,
    )

def jaccard(expr: IntoExpr, other: IntoExpr) -> pl.Expr:
    expr = parse_into_expr(expr, dtype=pl.Utf8)
    other = parse_into_expr(other, dtype=pl.Utf8)
    return expr.register_plugin(
        lib=lib,
        args=[other],
        symbol="jaccard",
        is_elementwise=True,
    )

def sorensen_dice(expr: IntoExpr, other: IntoExpr) -> pl.Expr:
    expr = parse_into_expr(expr, dtype=pl.Utf8)
    other = parse_into_expr(other, dtype=pl.Utf8)
    return expr.register_plugin(
        lib=lib,
        args=[other],
        symbol="sorensen_dice",
        is_elementwise=True,
    )

__all__ = [
    "levenshtein",
    "jaro",
    "jaro_winkler",
    "jaccard",
    "sorensen_dice",
]
