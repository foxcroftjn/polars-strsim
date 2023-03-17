mod strsim;

use polars::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::error::PyPolarsErr;
use pyo3_polars::{PyDataFrame, PySeries};

#[pyfunction]
fn levenshtein(pydf: PyDataFrame, col_a: &str, col_b: &str) -> PyResult<PySeries> {
    let df: DataFrame = pydf.into();
    let s = strsim::parallel_levenshtein(df, col_a, col_b).map_err(PyPolarsErr::from)?;
    Ok(PySeries(s))
}

/// A Python module implemented in Rust.
#[pymodule]
fn polars_strsim(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(levenshtein, m)?)?;
    Ok(())
}
