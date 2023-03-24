import polars_strsim.polars_strsim as rust

def levenshtein(df, col_a, col_b, name='levenshtein'):
    return df.with_columns(rust.levenshtein(df, col_a, col_b, name))

def jaro(df, col_a, col_b, name='jaro'):
    return df.with_columns(rust.jaro(df, col_a, col_b, name))

def jaro_winkler(df, col_a, col_b, name='jaro_winkler'):
    return df.with_columns(rust.jaro_winkler(df, col_a, col_b, name))

def jaccard(df, col_a, col_b, name='jaccard'):
    return df.with_columns(rust.jaccard(df, col_a, col_b, name))

__all__ = [
    "levenshtein",
    "jaro",
    "jaro_winkler",
    "jaccard",
]
