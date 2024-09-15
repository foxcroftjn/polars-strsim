# String Similarity Measures for Polars

This package provides python bindings to compute various string similarity measures directly on a polars dataframe. All string similarity measures are implemented in rust and computed in parallel.

The similarity measures that have been implemented are:

- Levenshtein
- Jaro
- Jaro-Winkler
- Jaccard
- Sørensen-Dice

Each similarity measure returns a value normalized between 0.0 and 1.0 (inclusive), where 0.0 indicates the inputs are maximally different and 1.0 means the strings are maximally similar.

## Installing the Library

### With pip

```bash
pip install polars-strsim
```

### From Source

To build and install this library from source, first ensure you have [cargo](https://doc.rust-lang.org/cargo/getting-started/installation.html) installed. You will also need maturin, which you can install via `pip install 'maturin[patchelf]'`

polars-strsim can then be installed in your current python environment by running `maturin develop --release`

## Using the Library

**Input:**

```python
import polars as pl
from polars_strsim import levenshtein, jaro, jaro_winkler, jaccard, sorensen_dice

df = pl.DataFrame(
    {
        "name_a": ["phillips", "phillips", ""        , "", None      , None],
        "name_b": ["phillips", "philips" , "phillips", "", "phillips", None],
    }
).with_columns(
    levenshtein=levenshtein("name_a", "name_b"),
    jaro=jaro("name_a", "name_b"),
    jaro_winkler=jaro_winkler("name_a", "name_b"),
    jaccard=jaccard("name_a", "name_b"),
    sorensen_dice=sorensen_dice("name_a", "name_b"),
)

print(df)
```
**Output:**
```
shape: (6, 7)
┌──────────┬──────────┬─────────────┬──────────┬──────────────┬─────────┬───────────────┐
│ name_a   ┆ name_b   ┆ levenshtein ┆ jaro     ┆ jaro_winkler ┆ jaccard ┆ sorensen_dice │
│ ---      ┆ ---      ┆ ---         ┆ ---      ┆ ---          ┆ ---     ┆ ---           │
│ str      ┆ str      ┆ f64         ┆ f64      ┆ f64          ┆ f64     ┆ f64           │
╞══════════╪══════════╪═════════════╪══════════╪══════════════╪═════════╪═══════════════╡
│ phillips ┆ phillips ┆ 1.0         ┆ 1.0      ┆ 1.0          ┆ 1.0     ┆ 1.0           │
│ phillips ┆ philips  ┆ 0.875       ┆ 0.958333 ┆ 0.975        ┆ 0.875   ┆ 0.933333      │
│          ┆ phillips ┆ 0.0         ┆ 0.0      ┆ 0.0          ┆ 0.0     ┆ 0.0           │
│          ┆          ┆ 1.0         ┆ 1.0      ┆ 1.0          ┆ 1.0     ┆ 1.0           │
│ null     ┆ phillips ┆ null        ┆ null     ┆ null         ┆ null    ┆ null          │
│ null     ┆ null     ┆ null        ┆ null     ┆ null         ┆ null    ┆ null          │
└──────────┴──────────┴─────────────┴──────────┴──────────────┴─────────┴───────────────┘
```
