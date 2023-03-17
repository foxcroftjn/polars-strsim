use polars::prelude::*;
use rayon::prelude::*;

fn split_offsets(len: usize, n: usize) -> Vec<(usize, usize)> {
    if n == 1 {
        vec![(0, len)]
    } else {
        let chunk_size = len / n;

        (0..n)
            .map(|partition| {
                let offset = partition * chunk_size;
                let len = if partition == (n - 1) {
                    len - offset
                } else {
                    chunk_size
                };
                (partition * chunk_size, len)
            })
            .collect()
    }
}

fn parallel_apply(
    df: DataFrame,
    col_a: &str,
    col_b: &str,
    name: &str,
    function: fn(&str, &str) -> f32,
) -> PolarsResult<Series> {
    let offsets = split_offsets(df.height(), rayon::current_num_threads());

    let out = Float32Chunked::new(
        name,
        offsets
            .par_iter()
            .map(|(offset, len)| {
                let sub_df = df.slice(*offset as i64, *len);
                let string_a = sub_df.column(col_a)?;
                let string_b = sub_df.column(col_b)?;

                Ok(string_a
                    .utf8()?
                    .into_iter()
                    .zip(string_b.utf8()?.into_iter())
                    .map(|(a, b)| match (a, b) {
                        (Some(a), Some(b)) => Some(function(a, b)),
                        _ => None,
                    })
                    .collect::<Vec<_>>())
            })
            .collect::<PolarsResult<Vec<_>>>()?
            .into_iter()
            .flatten()
            .collect::<Vec<Option<f32>>>(),
    );

    Ok(out.into_series())
}

// adapted from https://en.wikipedia.org/wiki/Levenshtein_distance#Iterative_with_two_matrix_rows
fn compute_levenshtein(a: &str, b: &str) -> f32 {
    if (a == "" && b == "") || (a == b) {
        return 1.0;
    }
    let a = a.chars().collect::<Vec<_>>();
    let b = b.chars().collect::<Vec<_>>();
    let mut matrix = [(0..=b.len()).collect(), vec![0; b.len() + 1]];
    for i in 0..a.len() {
        let v0 = i % 2;
        let v1 = (i + 1) % 2;
        matrix[v1][0] = i + 1;
        for j in 0..b.len() {
            matrix[v1][j + 1] = if a[i] == b[j] {
                matrix[v0][j]
            } else {
                matrix[v0][j] + 1
            }
            .min(matrix[v0][j + 1] + 1)
            .min(matrix[v1][j] + 1);
        }
    }
    return 1.0 - (matrix[a.len() % 2][b.len()] as f32 / a.len().max(b.len()) as f32);
}

pub(super) fn parallel_levenshtein(
    df: DataFrame,
    col_a: &str,
    col_b: &str,
) -> PolarsResult<Series> {
    Ok(parallel_apply(
        df,
        col_a,
        col_b,
        "levenshtein",
        compute_levenshtein,
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_eq_float(a: f32, b: f32) {
        assert_eq!(format!("{:.6}", a), format!("{:.6}", b));
    }

    #[test]
    #[rustfmt::skip]
    fn test_levenshtein() {
        assert_eq_float(compute_levenshtein("string", "string"), 1.0);
        assert_eq_float(compute_levenshtein("", ""), 1.0);
        assert_eq_float(compute_levenshtein("string", ""), 0.0);
        assert_eq_float(compute_levenshtein("", "string"), 0.0);
        assert_eq_float(compute_levenshtein("phillips", "philips"), 7.0 / 8.0);
        assert_eq_float(compute_levenshtein("kelly", "kelley"), 5.0 / 6.0);
        assert_eq_float(compute_levenshtein("wood", "woods"), 4.0 / 5.0);
        assert_eq_float(compute_levenshtein("russell", "russel"), 6.0 / 7.0);
        assert_eq_float(compute_levenshtein("macdonald", "mcdonald"), 8.0 / 9.0);
        assert_eq_float(compute_levenshtein("gray", "grey"), 3.0 / 4.0);
        assert_eq_float(compute_levenshtein("myers", "myres"), 3.0 / 5.0);
        assert_eq_float(compute_levenshtein("chamberlain", "chamberlin"), 10.0 / 11.0);
        assert_eq_float(compute_levenshtein("pennington", "penington"), 9.0 / 10.0);
        assert_eq_float(compute_levenshtein("ziegler", "zeigler"), 5.0 / 7.0);
        assert_eq_float(compute_levenshtein("hendrix", "hendricks"), 2.0 / 3.0);
        assert_eq_float(compute_levenshtein("abel", "able"), 1.0 / 2.0);
        assert_eq_float(compute_levenshtein("plantagenet", "lancaster"), 5.0 / 11.0);
        assert_eq_float(compute_levenshtein("featherstone", "featherston"), 11.0 / 12.0);
        assert_eq_float(compute_levenshtein("shackelford", "shackleford"), 9.0 / 11.0);
        assert_eq_float(compute_levenshtein("hazelwood", "hazlewood"), 7.0 / 9.0);
        assert_eq_float(compute_levenshtein("plantagenet", "gaunt"), 3.0 / 11.0);
        assert_eq_float(compute_levenshtein("powhatan", "rolfe"), 1.0 / 8.0);
        assert_eq_float(compute_levenshtein("landen", "austrasia"), 0.0);
        assert_eq_float(compute_levenshtein("fenstermacher", "fenstermaker"), 11.0 / 13.0);
        assert_eq_float(compute_levenshtein("hetherington", "heatherington"), 12.0 / 13.0);
        assert_eq_float(compute_levenshtein("defalaise", "arletta"), 1.0 / 9.0);
        assert_eq_float(compute_levenshtein("demeschines", "meschin"), 7.0 / 11.0);
        assert_eq_float(compute_levenshtein("archambault", "archambeau"), 8.0 / 11.0);
        assert_eq_float(compute_levenshtein("mormaer", "thane"), 1.0 / 7.0);
        assert_eq_float(compute_levenshtein("fitzpiers", "piers"), 5.0 / 9.0);
        assert_eq_float(compute_levenshtein("normandy", "brittany"), 1.0 / 4.0);
        assert_eq_float(compute_levenshtein("aetheling", "exile"), 2.0 / 9.0);
        assert_eq_float(compute_levenshtein("barlowe", "almy"), 2.0 / 7.0);
        assert_eq_float(compute_levenshtein("macmurrough", "macmurchada"), 6.0 / 11.0);
        assert_eq_float(compute_levenshtein("tourault", "archambault"), 4.0 / 11.0);
        assert_eq_float(compute_levenshtein("detoeni", "toni"), 4.0 / 7.0);
        assert_eq_float(compute_levenshtein("dechatellerault", "chatellerault"), 13.0 / 15.0);
        assert_eq_float(compute_levenshtein("hatherly", "hanford"), 3.0 / 8.0);
        assert_eq_float(compute_levenshtein("christoffersen", "christofferson"), 13.0 / 14.0);
        assert_eq_float(compute_levenshtein("blackshear", "blackshire"), 7.0 / 10.0);
        assert_eq_float(compute_levenshtein("fitzjohn", "fitzgeoffrey"), 5.0 / 12.0);
        assert_eq_float(compute_levenshtein("decrepon", "hardaknutsson"), 3.0 / 13.0);
        assert_eq_float(compute_levenshtein("dentzer", "henckel"), 3.0 / 7.0);
        assert_eq_float(compute_levenshtein("hignite", "hignight"), 5.0 / 8.0);
        assert_eq_float(compute_levenshtein("selbee", "blott"), 1.0 / 6.0);
        assert_eq_float(compute_levenshtein("cavendishbentinck", "bentinck"), 8.0 / 17.0);
        assert_eq_float(compute_levenshtein("reinschmidt", "cuntze"), 1.0 / 11.0);
        assert_eq_float(compute_levenshtein("vancouwenhoven", "couwenhoven"), 11.0 / 14.0);
        assert_eq_float(compute_levenshtein("aldin", "nalle"), 1.0 / 5.0);
        assert_eq_float(compute_levenshtein("offley", "thoroughgood"), 1.0 / 12.0);
        assert_eq_float(compute_levenshtein("sumarlidasson", "somerledsson"), 9.0 / 13.0);
        assert_eq_float(compute_levenshtein("wye", "why"), 1.0 / 3.0);
        assert_eq_float(compute_levenshtein("landvatter", "merckle"), 1.0 / 10.0);
        assert_eq_float(compute_levenshtein("moytoy", "oconostota"), 3.0 / 10.0);
        assert_eq_float(compute_levenshtein("mountbatten", "battenberg"), 2.0 / 11.0);
        assert_eq_float(compute_levenshtein("wentworthfitzwilliam", "fitzwilliam"), 11.0 / 20.0);
        assert_eq_float(compute_levenshtein("ingaldesthorpe", "ingoldsthrop"), 9.0 / 14.0);
        assert_eq_float(compute_levenshtein("munning", "munningmunny"), 7.0 / 12.0);
        assert_eq_float(compute_levenshtein("sinor", "snier"), 2.0 / 5.0);
        assert_eq_float(compute_levenshtein("featherstonhaugh", "featherstonehaugh"), 16.0 / 17.0);
        assert_eq_float(compute_levenshtein("hepburnstuartforbestrefusis", "trefusis"), 8.0 / 27.0);
        assert_eq_float(compute_levenshtein("destroismaisons", "destrosmaisons"), 14.0 / 15.0);
        assert_eq_float(compute_levenshtein("demoleyns", "molines"), 4.0 / 9.0);
        assert_eq_float(compute_levenshtein("chetwyndstapylton", "stapylton"), 9.0 / 17.0);
        assert_eq_float(compute_levenshtein("vanderburchgraeff", "burchgraeff"), 11.0 / 17.0);
        assert_eq_float(compute_levenshtein("manitouabeouich", "manithabehich"), 11.0 / 15.0);
        assert_eq_float(compute_levenshtein("decrocketagne", "crocketagni"), 10.0 / 13.0);
        assert_eq_float(compute_levenshtein("vannoorstrant", "juriaens"), 2.0 / 13.0);
        assert_eq_float(compute_levenshtein("twisletonwykehamfiennes", "fiennes"), 7.0 / 23.0);
        assert_eq_float(compute_levenshtein("hennikermajor", "henniker"), 8.0 / 13.0);
        assert_eq_float(compute_levenshtein("haakonsdatter", "haakonson"), 7.0 / 13.0);
        assert_eq_float(compute_levenshtein("aupry", "auprybertrand"), 5.0 / 13.0);
        assert_eq_float(compute_levenshtein("thorsteinsdottir", "thurstenson"), 9.0 / 16.0);
        assert_eq_float(compute_levenshtein("grossnicklaus", "greenehouse"), 4.0 / 13.0);
    }
}
