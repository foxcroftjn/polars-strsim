import polars as pl
from polars_strsim import levenshtein, jaro, jaro_winkler, jaccard, sorensen_dice

df = pl.DataFrame(
    {
        "name_a": ["phillips", "phillips", "", "", None, None],
        "name_b": ["phillips", "philips", "phillips", "", "phillips", None],
    }
).with_columns(
    levenshtein=levenshtein("name_a", "name_b"),
    jaro=jaro("name_a", "name_b"),
    jaro_winkler=jaro_winkler("name_a", "name_b"),
    jaccard=jaccard("name_a", "name_b"),
    sorensen_dice=sorensen_dice("name_a", "name_b"),
)

with pl.Config(ascii_tables=True):
    print(df)
