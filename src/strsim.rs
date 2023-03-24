use polars::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;

const INITIAL_BUFFER_LENGTH: usize = 50;

enum SimilarityFunctionType {
    Levenshtein,
    Jaro,
    JaroWinkler,
    Jaccard,
}

trait SimilarityFunction {
    fn compute(&mut self, a: &str, b: &str) -> f64;
}

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
    function: SimilarityFunctionType,
) -> PolarsResult<Series> {
    let offsets = split_offsets(df.height(), rayon::current_num_threads());

    let out = Float64Chunked::new(
        name,
        offsets
            .par_iter()
            .map(|(offset, len)| {
                let sub_df = df.slice(*offset as i64, *len);
                let string_a = sub_df.column(col_a)?;
                let string_b = sub_df.column(col_b)?;
                let mut function: Box<dyn SimilarityFunction> = match function {
                    SimilarityFunctionType::Levenshtein => Box::new(Levenshtein::new()),
                    SimilarityFunctionType::Jaro => Box::new(Jaro::new()),
                    SimilarityFunctionType::JaroWinkler => Box::new(JaroWinkler::new()),
                    SimilarityFunctionType::Jaccard => Box::new(Jaccard::new()),
                };

                Ok(string_a
                    .utf8()?
                    .into_iter()
                    .zip(string_b.utf8()?.into_iter())
                    .map(|(a, b)| match (a, b) {
                        (Some(a), Some(b)) => Some(function.compute(a, b)),
                        _ => None,
                    })
                    .collect::<Vec<_>>())
            })
            .collect::<PolarsResult<Vec<_>>>()?
            .into_iter()
            .flatten()
            .collect::<Vec<Option<f64>>>(),
    );

    Ok(out.into_series())
}

struct Levenshtein {
    a_buffer: Vec<char>,
    b_buffer: Vec<char>,
    matrix: Vec<[usize; 2]>,
}

impl Levenshtein {
    fn new() -> Levenshtein {
        Levenshtein {
            a_buffer: Vec::with_capacity(INITIAL_BUFFER_LENGTH),
            b_buffer: Vec::with_capacity(INITIAL_BUFFER_LENGTH),
            matrix: Vec::with_capacity(INITIAL_BUFFER_LENGTH),
        }
    }
}

impl SimilarityFunction for Levenshtein {
    // adapted from https://en.wikipedia.org/wiki/Levenshtein_distance#Iterative_with_two_matrix_rows
    fn compute(&mut self, a: &str, b: &str) -> f64 {
        if (a == "" && b == "") || (a == b) {
            return 1.0;
        }
        let a = {
            self.a_buffer.clear();
            self.a_buffer.extend(a.chars());
            &self.a_buffer
        };
        let b = {
            self.b_buffer.clear();
            self.b_buffer.extend(b.chars());
            &self.b_buffer
        };
        let matrix = {
            self.matrix.clear();
            self.matrix.extend((0..=b.len()).map(|i| [i, 0]));
            &mut self.matrix
        };
        for (i, &a_i) in a.iter().enumerate() {
            let v0 = i % 2;
            let v1 = (i + 1) % 2;
            matrix[0][v1] = i + 1;
            for (j, &b_j) in b.iter().enumerate() {
                matrix[j + 1][v1] = if a_i == b_j {
                    matrix[j][v0]
                } else {
                    matrix[j][v0] + 1
                }
                .min(matrix[j + 1][v0] + 1)
                .min(matrix[j][v1] + 1);
            }
        }
        return 1.0 - (matrix[b.len()][a.len() % 2] as f64 / a.len().max(b.len()) as f64);
    }
}

pub(super) fn parallel_levenshtein(
    df: DataFrame,
    col_a: &str,
    col_b: &str,
    name: &str,
) -> PolarsResult<Series> {
    Ok(parallel_apply(
        df,
        col_a,
        col_b,
        name,
        SimilarityFunctionType::Levenshtein,
    )?)
}

struct Jaro {
    a_buffer: Vec<char>,
    b_buffer: Vec<char>,
    flagged: Vec<[bool; 2]>,
}

impl Jaro {
    fn new() -> Jaro {
        Jaro {
            a_buffer: Vec::with_capacity(INITIAL_BUFFER_LENGTH),
            b_buffer: Vec::with_capacity(INITIAL_BUFFER_LENGTH),
            flagged: Vec::with_capacity(INITIAL_BUFFER_LENGTH),
        }
    }
}

impl SimilarityFunction for Jaro {
    fn compute(&mut self, a: &str, b: &str) -> f64 {
        if (a == "" && b == "") || (a == b) {
            return 1.0;
        } else if a == "" || b == "" {
            return 0.0;
        }
        let a = {
            self.a_buffer.clear();
            self.a_buffer.extend(a.chars());
            &self.a_buffer
        };
        let b = {
            self.b_buffer.clear();
            self.b_buffer.extend(b.chars());
            &self.b_buffer
        };
        if a.len() == 1 && b.len() == 1 {
            return if a[0] == b[0] { 1.0 } else { 0.0 };
        }
        let bound = a.len().max(b.len()) / 2 - 1;
        let mut m = 0;
        let flagged = {
            self.flagged.clear();
            self.flagged
                .extend(std::iter::repeat([false; 2]).take(a.len().max(b.len())));
            &mut self.flagged
        };
        for (i, &a_i) in a.iter().enumerate() {
            let lowerbound = if bound > i { 0 } else { i - bound };
            let upperbound = (i + bound).min(b.len() - 1);
            for j in lowerbound..=upperbound {
                if a_i == b[j] && !flagged[j][1] {
                    m += 1;
                    flagged[i][0] = true;
                    flagged[j][1] = true;
                    break;
                }
            }
        }
        let t = flagged
            .iter()
            .enumerate()
            .filter_map(|(i, [flag, _])| match flag {
                true => Some(i),
                false => None,
            })
            .zip(
                flagged
                    .iter()
                    .enumerate()
                    .filter_map(|(j, [_, flag])| match flag {
                        true => Some(j),
                        false => None,
                    }),
            )
            .filter(|&(i, j)| a[i] != b[j])
            .count();
        if m == 0 {
            return 0.0;
        } else {
            return (m as f64 / a.len() as f64
                + m as f64 / b.len() as f64
                + (m - t / 2) as f64 / m as f64)
                / 3.0;
        }
    }
}

pub(super) fn parallel_jaro(
    df: DataFrame,
    col_a: &str,
    col_b: &str,
    name: &str,
) -> PolarsResult<Series> {
    Ok(parallel_apply(
        df,
        col_a,
        col_b,
        name,
        SimilarityFunctionType::Jaro,
    )?)
}

struct JaroWinkler {
    jaro: Jaro,
}

impl JaroWinkler {
    fn new() -> JaroWinkler {
        JaroWinkler { jaro: Jaro::new() }
    }
}

impl SimilarityFunction for JaroWinkler {
    fn compute(&mut self, a: &str, b: &str) -> f64 {
        let jaro_similarity = self.jaro.compute(a, b);
        return if jaro_similarity > 0.7 {
            let shared_prefix_length = a
                .chars()
                .zip(b.chars())
                .take(4)
                .take_while(|(c, d)| c == d)
                .count() as f64;
            jaro_similarity + (shared_prefix_length * 0.1 * (1.0 - jaro_similarity))
        } else {
            jaro_similarity
        };
    }
}

pub(super) fn parallel_jaro_winkler(
    df: DataFrame,
    col_a: &str,
    col_b: &str,
    name: &str,
) -> PolarsResult<Series> {
    Ok(parallel_apply(
        df,
        col_a,
        col_b,
        name,
        SimilarityFunctionType::JaroWinkler,
    )?)
}

struct Jaccard {
    buffer: HashMap<char, [usize; 2]>,
}

impl Jaccard {
    fn new() -> Jaccard {
        Jaccard {
            buffer: HashMap::with_capacity(INITIAL_BUFFER_LENGTH),
        }
    }
}

impl SimilarityFunction for Jaccard {
    fn compute(&mut self, a: &str, b: &str) -> f64 {
        if (a == "" && b == "") || (a == b) {
            return 1.0;
        } else if a == "" || b == "" {
            return 0.0;
        }
        let buffer = {
            self.buffer.clear();
            &mut self.buffer
        };
        a.chars()
            .for_each(|c| buffer.entry(c).or_insert([0; 2])[0] += 1);
        b.chars()
            .for_each(|c| buffer.entry(c).or_insert([0; 2])[1] += 1);
        let frac = buffer.values().fold([0; 2], |mut f, v| {
            f[0] += v[0].min(v[1]);
            f[1] += v[0].max(v[1]);
            f
        });
        return frac[0] as f64 / frac[1] as f64;
    }
}

pub(super) fn parallel_jaccard(
    df: DataFrame,
    col_a: &str,
    col_b: &str,
    name: &str,
) -> PolarsResult<Series> {
    Ok(parallel_apply(
        df,
        col_a,
        col_b,
        name,
        SimilarityFunctionType::Jaccard,
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;
    const THRESHOLD: f64 = 0.00000001;

    trait Test: SimilarityFunction {
        fn test(&mut self, a: &str, b: &str, expected_result: f64) {
            assert!(
                (self.compute(a, b) - expected_result).abs() < THRESHOLD,
                "\"{}\", \"{}\" was computed as {}, expected {}",
                a,
                b,
                self.compute(a, b),
                expected_result
            );
        }
    }

    impl Test for Levenshtein {}
    impl Test for Jaro {}
    impl Test for JaroWinkler {}
    impl Test for Jaccard {}

    #[test]
    fn levenshtein_edge_cases() {
        let mut lev = Levenshtein::new();
        lev.test("s", "s", 1.0);
        lev.test("s", "", 0.0);
        lev.test("", "s", 0.0);
        lev.test("", "", 1.0);
        lev.test("string", "", 0.0);
        lev.test("", "string", 0.0);
    }

    #[test]
    fn levenshtein_test_cases() {
        let mut lev = Levenshtein::new();
        lev.test("phillips", "philips", 0.875);
        lev.test("kelly", "kelley", 0.83333333);
        lev.test("wood", "woods", 0.8);
        lev.test("russell", "russel", 0.85714286);
        lev.test("macdonald", "mcdonald", 0.88888889);
        lev.test("gray", "grey", 0.75);
        lev.test("myers", "myres", 0.6);
        lev.test("chamberlain", "chamberlin", 0.90909091);
        lev.test("pennington", "penington", 0.9);
        lev.test("ziegler", "zeigler", 0.71428571);
        lev.test("hendrix", "hendricks", 0.66666667);
        lev.test("abel", "able", 0.5);
        lev.test("plantagenet", "lancaster", 0.45454545);
        lev.test("featherstone", "featherston", 0.91666667);
        lev.test("shackelford", "shackleford", 0.81818182);
        lev.test("hazelwood", "hazlewood", 0.77777778);
        lev.test("plantagenet", "gaunt", 0.27272727);
        lev.test("powhatan", "rolfe", 0.125);
        lev.test("landen", "austrasia", 0.0);
        lev.test("fenstermacher", "fenstermaker", 0.84615385);
        lev.test("hetherington", "heatherington", 0.92307692);
        lev.test("defalaise", "arletta", 0.11111111);
        lev.test("demeschines", "meschin", 0.63636364);
        lev.test("archambault", "archambeau", 0.72727273);
        lev.test("mormaer", "thane", 0.14285714);
        lev.test("fitzpiers", "piers", 0.55555556);
        lev.test("normandy", "brittany", 0.25);
        lev.test("aetheling", "exile", 0.22222222);
        lev.test("barlowe", "almy", 0.28571429);
        lev.test("macmurrough", "macmurchada", 0.54545455);
        lev.test("tourault", "archambault", 0.36363636);
        lev.test("detoeni", "toni", 0.57142857);
        lev.test("dechatellerault", "chatellerault", 0.86666667);
        lev.test("hatherly", "hanford", 0.375);
        lev.test("christoffersen", "christofferson", 0.92857143);
        lev.test("blackshear", "blackshire", 0.7);
        lev.test("fitzjohn", "fitzgeoffrey", 0.41666667);
        lev.test("decrepon", "hardaknutsson", 0.23076923);
        lev.test("dentzer", "henckel", 0.42857143);
        lev.test("hignite", "hignight", 0.625);
        lev.test("selbee", "blott", 0.16666667);
        lev.test("cavendishbentinck", "bentinck", 0.47058824);
        lev.test("reinschmidt", "cuntze", 0.090909091);
        lev.test("vancouwenhoven", "couwenhoven", 0.78571429);
        lev.test("aldin", "nalle", 0.2);
        lev.test("offley", "thoroughgood", 0.083333333);
        lev.test("sumarlidasson", "somerledsson", 0.69230769);
        lev.test("wye", "why", 0.33333333);
        lev.test("landvatter", "merckle", 0.1);
        lev.test("moytoy", "oconostota", 0.3);
        lev.test("mountbatten", "battenberg", 0.18181818);
        lev.test("wentworthfitzwilliam", "fitzwilliam", 0.55);
        lev.test("ingaldesthorpe", "ingoldsthrop", 0.64285714);
        lev.test("munning", "munningmunny", 0.58333333);
        lev.test("sinor", "snier", 0.4);
        lev.test("featherstonhaugh", "featherstonehaugh", 0.94117647);
        lev.test("hepburnstuartforbestrefusis", "trefusis", 0.2962963);
        lev.test("destroismaisons", "destrosmaisons", 0.93333333);
        lev.test("demoleyns", "molines", 0.44444444);
        lev.test("chetwyndstapylton", "stapylton", 0.52941176);
        lev.test("vanderburchgraeff", "burchgraeff", 0.64705882);
        lev.test("manitouabeouich", "manithabehich", 0.73333333);
        lev.test("decrocketagne", "crocketagni", 0.76923077);
        lev.test("vannoorstrant", "juriaens", 0.15384615);
        lev.test("twisletonwykehamfiennes", "fiennes", 0.30434783);
        lev.test("hennikermajor", "henniker", 0.61538462);
        lev.test("haakonsdatter", "haakonson", 0.53846154);
        lev.test("aupry", "auprybertrand", 0.38461538);
        lev.test("thorsteinsdottir", "thurstenson", 0.5625);
        lev.test("grossnicklaus", "greenehouse", 0.30769231);
    }

    #[test]
    fn jaro_edge_cases() {
        let mut j = Jaro::new();
        j.test("s", "a", 0.0);
        j.test("s", "s", 1.0);
        j.test("", "", 1.0);
        j.test("string", "", 0.0);
        j.test("", "string", 0.0);
    }

    #[test]
    fn jaro_test_cases() {
        let mut j = Jaro::new();
        j.test("phillips", "philips", 0.95833333);
        j.test("kelly", "kelley", 0.94444444);
        j.test("wood", "woods", 0.93333333);
        j.test("russell", "russel", 0.95238095);
        j.test("macdonald", "mcdonald", 0.96296296);
        j.test("hansen", "hanson", 0.88888889);
        j.test("gray", "grey", 0.83333333);
        j.test("petersen", "peterson", 0.91666667);
        j.test("chamberlain", "chamberlin", 0.96969697);
        j.test("blankenship", "blankinship", 0.87272727);
        j.test("dickinson", "dickenson", 0.92592593);
        j.test("mullins", "mullens", 0.9047619);
        j.test("olsen", "olson", 0.86666667);
        j.test("pennington", "penington", 0.85555556);
        j.test("cunningham", "cuningham", 0.92962963);
        j.test("gillespie", "gillispie", 0.88425926);
        j.test("callaway", "calloway", 0.86904762);
        j.test("mackay", "mckay", 0.87777778);
        j.test("christensen", "christenson", 0.93939394);
        j.test("mcallister", "mcalister", 0.96666667);
        j.test("cleveland", "cleaveland", 0.89259259);
        j.test("denison", "dennison", 0.86309524);
        j.test("livingston", "levingston", 0.8962963);
        j.test("hendrix", "hendricks", 0.84126984);
        j.test("stillwell", "stilwell", 0.9212963);
        j.test("schaefer", "schafer", 0.91071429);
        j.test("moseley", "mosley", 0.8968254);
        j.test("plantagenet", "lancaster", 0.68181818);
        j.test("horner", "homer", 0.73888889);
        j.test("whittington", "whitington", 0.9030303);
        j.test("featherstone", "featherston", 0.97222222);
        j.test("reeder", "reader", 0.82222222);
        j.test("scarborough", "scarbrough", 0.93636364);
        j.test("higginbotham", "higgenbotham", 0.94444444);
        j.test("flanagan", "flannagan", 0.87962963);
        j.test("debeauchamp", "beauchamp", 0.9023569);
        j.test("donovan", "donavan", 0.84920635);
        j.test("plantagenet", "gaunt", 0.62424242);
        j.test("reedy", "rudy", 0.78333333);
        j.test("egan", "eagan", 0.85);
        j.test("powhatan", "rolfe", 0.44166667);
        j.test("landen", "austrasia", 0.42592593);
        j.test("pelletier", "peltier", 0.87830688);
        j.test("arrowood", "arwood", 0.86111111);
        j.test("canterbury", "canterberry", 0.90606061);
        j.test("scotland", "canmore", 0.51190476);
        j.test("depercy", "percy", 0.83809524);
        j.test("rau", "raw", 0.77777778);
        j.test("powhatan", "daughter", 0.47222222);
        j.test("anjou", "jerusalem", 0.54074074);
        j.test("deferrers", "ferrers", 0.83068783);
        j.test("devermandois", "vermandois", 0.91111111);
        j.test("bainbridge", "bambridge", 0.85462963);
        j.test("zachary", "zackery", 0.80952381);
        j.test("witherspoon", "weatherspoon", 0.88080808);
        j.test("breckenridge", "brackenridge", 0.91414141);
        j.test("fenstermacher", "fenstermaker", 0.92094017);
        j.test("declermont", "clermont", 0.89166667);
        j.test("hetherington", "heatherington", 0.97435897);
        j.test("defalaise", "arletta", 0.58862434);
        j.test("demeschines", "meschin", 0.83116883);
        j.test("benningfield", "beningfield", 0.94191919);
        j.test("bretagne", "brittany", 0.75);
        j.test("beresford", "berrisford", 0.81296296);
        j.test("wydeville", "woodville", 0.85185185);
        j.test("mormaer", "thane", 0.56190476);
        j.test("fitzpiers", "piers", 0.43703704);
        j.test("decourtenay", "courtenay", 0.82828283);
        j.test("debadlesmere", "badlesmere", 0.77777778);
        j.test("dewarenne", "warenne", 0.78306878);
        j.test("deroet", "roet", 0.80555556);
        j.test("demeschines", "meschines", 0.79124579);
        j.test("normandy", "brittany", 0.66666667);
        j.test("garnsey", "guernsey", 0.75793651);
        j.test("aetheling", "exile", 0.53333333);
        j.test("barlowe", "almy", 0.5952381);
        j.test("haraldsdatter", "haraldsdotter", 0.94871795);
        j.test("macmurrough", "macmurchada", 0.75757576);
        j.test("falaise", "arletta", 0.52380952);
        j.test("deberkeley", "berkeley", 0.80833333);
        j.test("tourault", "archambault", 0.62651515);
        j.test("valliance", "pray", 0.4537037);
        j.test("fischbach", "fishback", 0.78902116);
        j.test("dechatellerault", "chatellerault", 0.87863248);
        j.test("trico", "rapalje", 0.44761905);
        j.test("hatherly", "hanford", 0.60119048);
        j.test("aquitaine", "eleanor", 0.5026455);
        j.test("devere", "vere", 0.72222222);
        j.test("coppedge", "coppage", 0.81349206);
        j.test("rockefeller", "rockafellow", 0.77651515);
        j.test("rubenstein", "rubinstein", 0.85925926);
        j.test("roet", "swynford", 0.0);
        j.test("bodenheimer", "bodenhamer", 0.86902357);
        j.test("dehauteville", "hauteville", 0.81111111);
        j.test("maugis", "miville", 0.53968254);
        j.test("fitzjohn", "fitzgeoffrey", 0.68055556);
        j.test("decrepon", "hardaknutsson", 0.43589744);
        j.test("deswynnerton", "swinnerton", 0.80925926);
        j.test("detaillefer", "angouleme", 0.6026936);
        j.test("fleitel", "flatel", 0.78253968);
        j.test("temperley", "timperley", 0.80092593);
        j.test("dentzer", "henckel", 0.61904762);
        j.test("deroet", "swynford", 0.43055556);
        j.test("daingerfield", "dangerfield", 0.88131313);
        j.test("selbee", "blott", 0.45555556);
        j.test("berkeley", "martiau", 0.42261905);
        j.test("cavendishbentinck", "bentinck", 0.50637255);
        j.test("mcmurrough", "leinster", 0.40833333);
        j.test("debraose", "briose", 0.81944444);
        j.test("reinschmidt", "cuntze", 0.33838384);
        j.test("vancouwenhoven", "couwenhoven", 0.77705628);
        j.test("fenstermaker", "fenstemaker", 0.91161616);
        j.test("kerrich", "keridge", 0.71428571);
        j.test("oberbroeckling", "oberbrockling", 0.97619048);
        j.test("fitzmaurice", "fitzmorris", 0.77878788);
        j.test("flowerdew", "yeardly", 0.58730159);
        j.test("shufflebotham", "shufflebottom", 0.8974359);
        j.test("demontdidier", "montdidier", 0.84444444);
        j.test("facteau", "facto", 0.79047619);
        j.test("aldin", "nalle", 0.6);
        j.test("helphenstine", "helphinstine", 0.88383838);
        j.test("fitzalan", "goushill", 0.41666667);
        j.test("riseborough", "roseborough", 0.83939394);
        j.test("gruffydd", "rhys", 0.58333333);
        j.test("hornberger", "homberger", 0.7712963);
        j.test("tattershall", "tatarsole", 0.70947571);
        j.test("taillefer", "angouleme", 0.62962963);
        j.test("goldhatch", "tritton", 0.41798942);
        j.test("sumarlidasson", "somerledsson", 0.81410256);
        j.test("cuvellier", "cuvilje", 0.78571429);
        j.test("amberg", "glatfelder", 0.51111111);
        j.test("ruel", "ruelle", 0.88888889);
        j.test("billung", "sachsen", 0.42857143);
        j.test("delazouche", "zouche", 0.7);
        j.test("springham", "springhorn", 0.82592593);
        j.test("deserres", "dessert", 0.7797619);
        j.test("gendre", "bourgery", 0.52777778);
        j.test("braconie", "brackhonge", 0.85833333);
        j.test("pleydellbouverie", "bouverie", 0.45833333);
        j.test("plamondon", "plomondon", 0.84259259);
        j.test("aubigny", "albini", 0.74603175);
        j.test("freemanmitford", "mitford", 0.43650794);
        j.test("fightmaster", "fight", 0.81818182);
        j.test("wye", "why", 0.55555556);
        j.test("clabough", "clabo", 0.875);
        j.test("lautzenheiser", "lautzenhiser", 0.9465812);
        j.test("tellico", "clan", 0.46428571);
        j.test("doublehead", "cornblossom", 0.52424242);
        j.test("landvatter", "merckle", 0.41428571);
        j.test("grimoult", "sedilot", 0.60714286);
        j.test("kluczykowski", "kluck", 0.80555556);
        j.test("moytoy", "oconostota", 0.48888889);
        j.test("steenbergen", "stenbergen", 0.86969697);
        j.test("wolfensberger", "wolfersberger", 0.86538462);
        j.test("lydecker", "leydecker", 0.83796296);
        j.test("mountbatten", "battenberg", 0.58484848);
        j.test("hawbaker", "hawbecker", 0.83664021);
        j.test("degrandison", "grandson", 0.82575758);
        j.test("ardouin", "badeau", 0.64285714);
        j.test("verdun", "ardenne", 0.66269841);
        j.test("riemenschneider", "reimenschneider", 0.97777778);
        j.test("wentworthfitzwilliam", "fitzwilliam", 0.78939394);
        j.test("vanschouwen", "cornelissen", 0.56969697);
        j.test("frederickse", "lubbertsen", 0.42121212);
        j.test("haraldsson", "forkbeard", 0.54444444);
        j.test("ingaldesthorpe", "ingoldsthrop", 0.87049062);
        j.test("blennerhassett", "blaverhasset", 0.81587302);
        j.test("dechow", "dago", 0.61111111);
        j.test("levere", "lavere", 0.75555556);
        j.test("denivelles", "itta", 0.45);
        j.test("decressingham", "cressingham", 0.91841492);
        j.test("esten", "eustance", 0.76666667);
        j.test("maves", "mebs", 0.63333333);
        j.test("deliercourt", "juillet", 0.56168831);
        j.test("auxerre", "argengau", 0.49007937);
        j.test("delisoures", "lisours", 0.9);
        j.test("muscoe", "hucklescott", 0.62929293);
        j.test("feese", "fuse", 0.67222222);
        j.test("laughinghouse", "lathinghouse", 0.86033411);
        j.test("decrocketagne", "crocketagne", 0.7972028);
        j.test("petitpas", "bugaret", 0.3452381);
        j.test("leatherbarrow", "letherbarrow", 0.89102564);
        j.test("goughcalthorpe", "calthorpe", 0.76984127);
        j.test("stooksbury", "stookesberry", 0.88333333);
        j.test("devalletort", "valletort", 0.86531987);
        j.test("duranceau", "duranso", 0.75661376);
        j.test("ordepowlett", "powlett", 0.78354978);
        j.test("featherstonhaugh", "featherstonehaugh", 0.98039216);
        j.test("hepburnstuartforbestrefusis", "trefusis", 0.56856261);
        j.test("minkrevicius", "minkavitch", 0.76111111);
        j.test("frande", "andersdotter", 0.61666667);
        j.test("alwyn", "joan", 0.48333333);
        j.test("landvatter", "varonica", 0.55);
        j.test("dewindsor", "fitzotho", 0.49074074);
        j.test("renkenberger", "rinkenberger", 0.82323232);
        j.test("volkertsen", "noorman", 0.57619048);
        j.test("bottenfield", "bottomfield", 0.84175084);
        j.test("decherleton", "cherlton", 0.86742424);
        j.test("sinquefield", "sinkfield", 0.83038721);
        j.test("courchesne", "courchaine", 0.825);
        j.test("humphrie", "umfery", 0.625);
        j.test("loignon", "longnon", 0.79365079);
        j.test("oesterle", "osterle", 0.81547619);
        j.test("evemy", "evering", 0.67619048);
        j.test("niquette", "nequette", 0.82142857);
        j.test("lemeunier", "daubigeon", 0.46296296);
        j.test("hartsvelder", "hartzfelder", 0.87878788);
        j.test("destroismaisons", "destrosmaisons", 0.93015873);
        j.test("warminger", "wanninger", 0.8042328);
        j.test("chetwyndstapylton", "stapylton", 0.47657952);
        j.test("fivekiller", "ghigau", 0.42222222);
        j.test("rochet", "garrigues", 0.51851852);
        j.test("leyendecker", "lyendecker", 0.83636364);
        j.test("roeloffse", "kierstede", 0.5462963);
        j.test("jarry", "rapin", 0.46666667);
        j.test("lawter", "lantersee", 0.7962963);
        j.test("requa", "regna", 0.73333333);
        j.test("devaloines", "volognes", 0.72777778);
        j.test("featherstonhaugh", "fetherstonbaugh", 0.88849206);
        j.test("sacherell", "searth", 0.66296296);
        j.test("coeffes", "forestier", 0.62328042);
        j.test("dewease", "duese", 0.70714286);
        j.test("neumeister", "newmaster", 0.77830688);
        j.test("delusignan", "lusigan", 0.85238095);
        j.test("hearnsberger", "harnsberger", 0.8510101);
        j.test("vanderburchgraeff", "burchgraeff", 0.76114082);
        j.test("tiptoft", "tybotot", 0.63095238);
        j.test("crepon", "forkbeard", 0.5);
        j.test("rugglesbrise", "brise", 0.50555556);
        j.test("brassier", "decheilus", 0.32407407);
        j.test("coningsby", "connyngesby", 0.78872054);
        j.test("ingaldesthorpe", "ingoldsthorp", 0.90079365);
        j.test("streitenberger", "strattenbarger", 0.76623377);
        j.test("manitouabeouich", "manithabehich", 0.82952603);
        j.test("jaeckler", "margaretha", 0.44722222);
        j.test("paulo", "campot", 0.57777778);
        j.test("essenmacher", "eunmaker", 0.6540404);
        j.test("decrocketagne", "crocketagni", 0.79277389);
        j.test("unruhe", "kornmann", 0.36111111);
        j.test("reidelberger", "rudelberger", 0.78080808);
        j.test("bradtmueller", "bradmiller", 0.8462963);
        j.test("schreckengast", "shreckengast", 0.91880342);
        j.test("fivekiller", "kingfisher", 0.67777778);
        j.test("reichenberger", "richenberger", 0.83547009);
        j.test("muttlebury", "mattleberry", 0.84242424);
        j.test("jobidon", "bidon", 0.77142857);
        j.test("badlesmere", "northampton", 0.46060606);
        j.test("oxier", "ockshire", 0.65833333);
        j.test("siebenthaler", "sevendollar", 0.69227994);
        j.test("vannoorstrant", "juriaens", 0.51923077);
        j.test("stautzenberger", "stantzenberger", 0.9010989);
        j.test("molandersmolandes", "molandes", 0.82352941);
        j.test("altstaetter", "allstetter", 0.83198653);
        j.test("moredock", "nedock", 0.75277778);
        j.test("bouslaugh", "baughlough", 0.68306878);
        j.test("schoenbachler", "schoenbaechler", 0.92490842);
        j.test("doors", "streypress", 0.36666667);
        j.test("andrieszen", "larens", 0.71111111);
        j.test("hughesdaeth", "daeth", 0.36060606);
        j.test("cullumbine", "colleunbine", 0.75909091);
        j.test("twisletonwykehamfiennes", "fiennes", 0.51055901);
        j.test("scherber", "sharver", 0.71309524);
        j.test("coerten", "harmens", 0.50793651);
        j.test("pitres", "fitzroger", 0.7037037);
        j.test("degloucester", "fitzroger", 0.50925926);
        j.test("sevestre", "delessart", 0.66018519);
        j.test("smelker", "schmelcher", 0.81904762);
        j.test("during", "shaumloffel", 0.41919192);
        j.test("otterlifter", "wawli", 0.52727273);
        j.test("wackerle", "weckerla", 0.77380952);
        j.test("manselpleydell", "pleydell", 0.73214286);
        j.test("schwabenlender", "schwabenlander", 0.92673993);
        j.test("thurner", "thur", 0.85714286);
        j.test("rauhuff", "rowhuff", 0.74285714);
        j.test("hennikermajor", "henniker", 0.87179487);
        j.test("depitres", "fitzroger", 0.64814815);
        j.test("hotzenbella", "hotzenpeller", 0.85606061);
        j.test("haakonsdatter", "haakonson", 0.77207977);
        j.test("martinvegue", "vegue", 0.43030303);
        j.test("alcombrack", "alkenbrack", 0.8);
        j.test("kirberger", "moelich", 0.33597884);
        j.test("fregia", "fruger", 0.69444444);
        j.test("braunberger", "bramberg", 0.83712121);
        j.test("katterheinrich", "katterhenrich", 0.95054945);
        j.test("bechtelheimer", "becktelheimer", 0.89316239);
        j.test("encke", "ink", 0.68888889);
        j.test("kettleborough", "kettleboro", 0.92307692);
        j.test("ardion", "rabouin", 0.71587302);
        j.test("wittelsbach", "palatine", 0.53787879);
        j.test("dechaux", "chapelain", 0.50529101);
        j.test("vancortenbosch", "cortenbosch", 0.83766234);
        j.test("swyersexey", "sexey", 0.65);
        j.test("deherville", "sohier", 0.52222222);
        j.test("coeffes", "fourestier", 0.6047619);
        j.test("kemeystynte", "tynte", 0.51313131);
        j.test("knutti", "margaritha", 0.34444444);
        j.test("boeckhout", "elswaerts", 0.48148148);
        j.test("vansintern", "sintern", 0.75714286);
        j.test("knatchbullhugessen", "hugessen", 0.40740741);
        j.test("aupry", "auprybertrand", 0.79487179);
        j.test("bigot", "guillebour", 0.52222222);
        j.test("thorsteinsdottir", "thurstenson", 0.68244949);
        j.test("schwinghammer", "swinghammer", 0.88811189);
        j.test("mickelborough", "mickleburrough", 0.87118437);
        j.test("mignon", "guiet", 0.41111111);
        j.test("tantaquidgeon", "quidgeon", 0.70512821);
        j.test("duyts", "satyrs", 0.58888889);
        j.test("cornelise", "esselsteyn", 0.61481481);
        j.test("gillington", "gullotine", 0.73068783);
        j.test("rogerstillstone", "tillstone", 0.71851852);
        j.test("voidy", "vedie", 0.62222222);
        j.test("smithdorrien", "dorrien", 0.76587302);
        j.test("groethausen", "grothouse", 0.87205387);
        j.test("grossnicklaus", "greenehouse", 0.55788656);
        j.test("wilmotsitwell", "sitwell", 0.56630037);
        j.test("boertgens", "harmense", 0.59351852);
        j.test("koetterhagen", "katterhagen", 0.81414141);
        j.test("berthelette", "barthelette", 0.80606061);
        j.test("schoettler", "brechting", 0.53148148);
        j.test("etringer", "thilges", 0.69047619);
        j.test("sigurdsson", "lodbrok", 0.46507937);
        j.test("deligny", "bidon", 0.56507937);
        j.test("winsofer", "hubbarde", 0.33333333);
        j.test("straatmaker", "stratenmaker", 0.84747475);
        j.test("ouderkerk", "heemstraat", 0.43333333);
        j.test("comalander", "cumberlander", 0.69722222);
    }

    #[test]
    fn jaro_winkler_edge_cases() {
        let mut jw = JaroWinkler::new();
        jw.test("s", "s", 1.0);
        jw.test("", "", 1.0);
        jw.test("string", "", 0.0);
        jw.test("", "string", 0.0);
    }

    #[test]
    fn jaro_winkler_test_cases() {
        let mut jw = JaroWinkler::new();
        jw.test("phillips", "philips", 0.975);
        jw.test("kelly", "kelley", 0.96666667);
        jw.test("matthews", "mathews", 0.97083333);
        jw.test("wood", "woods", 0.96);
        jw.test("hayes", "hays", 0.95333333);
        jw.test("russell", "russel", 0.97142857);
        jw.test("rogers", "rodgers", 0.96190476);
        jw.test("hansen", "hanson", 0.93333333);
        jw.test("gray", "grey", 0.86666667);
        jw.test("petersen", "peterson", 0.95);
        jw.test("myers", "myres", 0.94666667);
        jw.test("snyder", "snider", 0.91111111);
        jw.test("chamberlain", "chamberlin", 0.98181818);
        jw.test("blankenship", "blankinship", 0.92363636);
        jw.test("lloyd", "loyd", 0.94);
        jw.test("byrd", "bird", 0.85);
        jw.test("dickinson", "dickenson", 0.95555556);
        jw.test("whitaker", "whittaker", 0.97777778);
        jw.test("mullins", "mullens", 0.94285714);
        jw.test("frye", "fry", 0.94166667);
        jw.test("olsen", "olson", 0.90666667);
        jw.test("pennington", "penington", 0.89888889);
        jw.test("cunningham", "cuningham", 0.95074074);
        jw.test("kirby", "kerby", 0.88);
        jw.test("gillespie", "gillispie", 0.93055556);
        jw.test("hewitt", "hewett", 0.92222222);
        jw.test("lowery", "lowry", 0.96111111);
        jw.test("callaway", "calloway", 0.92142857);
        jw.test("melton", "milton", 0.9);
        jw.test("mckenzie", "mckinzie", 0.90833333);
        jw.test("macleod", "mcleod", 0.95714286);
        jw.test("davenport", "devenport", 0.89583333);
        jw.test("hutchison", "hutchinson", 0.95777778);
        jw.test("mackay", "mckay", 0.89);
        jw.test("christensen", "christenson", 0.96363636);
        jw.test("bannister", "banister", 0.97407407);
        jw.test("mcallister", "mcalister", 0.98);
        jw.test("cleveland", "cleaveland", 0.92481481);
        jw.test("denison", "dennison", 0.90416667);
        jw.test("hendrix", "hendricks", 0.9047619);
        jw.test("caudill", "candill", 0.92380952);
        jw.test("stillwell", "stilwell", 0.95277778);
        jw.test("klein", "kline", 0.89333333);
        jw.test("mayo", "mays", 0.88333333);
        jw.test("albright", "allbright", 0.97037037);
        jw.test("macpherson", "mcpherson", 0.97);
        jw.test("schaefer", "schafer", 0.94642857);
        jw.test("hoskins", "haskins", 0.91428571);
        jw.test("moseley", "mosley", 0.92777778);
        jw.test("plantagenet", "lancaster", 0.68181818);
        jw.test("kendrick", "kindrick", 0.925);
        jw.test("horner", "homer", 0.79111111);
        jw.test("waddell", "waddle", 0.93809524);
        jw.test("whittington", "whitington", 0.94181818);
        jw.test("featherstone", "featherston", 0.98333333);
        jw.test("broughton", "braughton", 0.94074074);
        jw.test("reeder", "reader", 0.85777778);
        jw.test("stedman", "steadman", 0.9375);
        jw.test("satterfield", "saterfield", 0.97878788);
        jw.test("scarborough", "scarbrough", 0.96181818);
        jw.test("devine", "divine", 0.84);
        jw.test("macfarlane", "mcfarlane", 0.87);
        jw.test("debeauchamp", "beauchamp", 0.9023569);
        jw.test("albritton", "allbritton", 0.97333333);
        jw.test("hutto", "hutts", 0.92);
        jw.test("swafford", "swofford", 0.8952381);
        jw.test("donovan", "donavan", 0.89444444);
        jw.test("plantagenet", "gaunt", 0.62424242);
        jw.test("rinehart", "rhinehart", 0.89166667);
        jw.test("macarthur", "mcarthur", 0.92916667);
        jw.test("hemingway", "hemmingway", 0.97666667);
        jw.test("schumacher", "schumaker", 0.93777778);
        jw.test("reedy", "rudy", 0.805);
        jw.test("dietrich", "deitrich", 0.9625);
        jw.test("egan", "eagan", 0.865);
        jw.test("steiner", "stiner", 0.91746032);
        jw.test("powhatan", "rolfe", 0.44166667);
        jw.test("beeler", "buler", 0.765);
        jw.test("landen", "austrasia", 0.42592593);
        jw.test("pelletier", "peltier", 0.91481481);
        jw.test("farnham", "farnum", 0.90952381);
        jw.test("arrowood", "arwood", 0.88888889);
        jw.test("canterbury", "canterberry", 0.94363636);
        jw.test("livengood", "livingood", 0.94814815);
        jw.test("scotland", "canmore", 0.51190476);
        jw.test("anjou", "danjou", 0.94444444);
        jw.test("stanfield", "standfield", 0.93555556);
        jw.test("depercy", "percy", 0.83809524);
        jw.test("weatherford", "wetherford", 0.97575758);
        jw.test("baumgardner", "bumgardner", 0.91272727);
        jw.test("rau", "raw", 0.82222222);
        jw.test("powhatan", "daughter", 0.47222222);
        jw.test("anjou", "jerusalem", 0.54074074);
        jw.test("deferrers", "ferrers", 0.83068783);
        jw.test("seagraves", "segraves", 0.93703704);
        jw.test("jaeger", "jager", 0.90222222);
        jw.test("ammerman", "amerman", 0.92857143);
        jw.test("muncy", "munsey", 0.87555556);
        jw.test("bainbridge", "bambridge", 0.8837037);
        jw.test("morehouse", "moorehouse", 0.91407407);
        jw.test("witherspoon", "weatherspoon", 0.89272727);
        jw.test("breckenridge", "brackenridge", 0.93131313);
        jw.test("giddings", "geddings", 0.88214286);
        jw.test("hochstetler", "hostetler", 0.95151515);
        jw.test("chivers", "chevers", 0.87936508);
        jw.test("macaulay", "mcaulay", 0.87678571);
        jw.test("fenstermacher", "fenstermaker", 0.9525641);
        jw.test("hetherington", "heatherington", 0.97948718);
        jw.test("defalaise", "arletta", 0.58862434);
        jw.test("breckenridge", "breckinridge", 0.94848485);
        jw.test("demeschines", "meschin", 0.83116883);
        jw.test("killingsworth", "killingswort", 0.98461538);
        jw.test("benningfield", "beningfield", 0.95934343);
        jw.test("bretagne", "brittany", 0.8);
        jw.test("stonebraker", "stonebreaker", 0.96515152);
        jw.test("beresford", "berrisford", 0.86907407);
        jw.test("yeo", "geo", 0.77777778);
        jw.test("henninger", "heninger", 0.94490741);
        jw.test("budgen", "bridgen", 0.86428571);
        jw.test("mormaer", "thane", 0.56190476);
        jw.test("braithwaite", "brathwaite", 0.93212121);
        jw.test("belfield", "bellfield", 0.91574074);
        jw.test("fitzpiers", "piers", 0.43703704);
        jw.test("decourtenay", "courtenay", 0.82828283);
        jw.test("teegarden", "teagarden", 0.90740741);
        jw.test("deholand", "holand", 0.91666667);
        jw.test("demowbray", "mowbray", 0.92592593);
        jw.test("macnaughton", "mcnaughton", 0.94272727);
        jw.test("dewarenne", "warenne", 0.78306878);
        jw.test("deroet", "roet", 0.80555556);
        jw.test("demeschines", "meschines", 0.79124579);
        jw.test("normandy", "brittany", 0.66666667);
        jw.test("brewington", "bruington", 0.91703704);
        jw.test("garnsey", "guernsey", 0.78214286);
        jw.test("aetheling", "exile", 0.53333333);
        jw.test("barlowe", "almy", 0.5952381);
        jw.test("mulholland", "mullholland", 0.95545455);
        jw.test("beddingfield", "bedingfield", 0.98055556);
        jw.test("couwenhoven", "covenhoven", 0.92484848);
        jw.test("macquarrie", "mcquarrie", 0.90333333);
        jw.test("haraldsdatter", "haraldsdotter", 0.96923077);
        jw.test("seed", "leed", 0.83333333);
        jw.test("pitsenbarger", "pittsenbarger", 0.98205128);
        jw.test("macmurrough", "macmurchada", 0.85454545);
        jw.test("falaise", "arletta", 0.52380952);
        jw.test("deberkeley", "berkeley", 0.80833333);
        jw.test("guernsey", "gurnsey", 0.89047619);
        jw.test("tourault", "archambault", 0.62651515);
        jw.test("valliance", "pray", 0.4537037);
        jw.test("enfinger", "infinger", 0.86904762);
        jw.test("fischbach", "fishback", 0.85231481);
        jw.test("pelham", "pellum", 0.84444444);
        jw.test("dechatellerault", "chatellerault", 0.87863248);
        jw.test("trico", "rapalje", 0.44761905);
        jw.test("hatherly", "hanford", 0.60119048);
        jw.test("aquitaine", "eleanor", 0.5026455);
        jw.test("devere", "vere", 0.72222222);
        jw.test("coppedge", "coppage", 0.88809524);
        jw.test("rockefeller", "rockafellow", 0.86590909);
        jw.test("rubenstein", "rubinstein", 0.90148148);
        jw.test("mcmurrough", "macmurrough", 0.97272727);
        jw.test("roet", "swynford", 0.0);
        jw.test("bodenheimer", "bodenhamer", 0.92141414);
        jw.test("dehauteville", "hauteville", 0.81111111);
        jw.test("jubinville", "jubenville", 0.92740741);
        jw.test("decantilupe", "cantilupe", 0.93939394);
        jw.test("kitteringham", "ketteringham", 0.92272727);
        jw.test("maugis", "miville", 0.53968254);
        jw.test("cornford", "comford", 0.85079365);
        jw.test("alsobrook", "alsabrook", 0.91898148);
        jw.test("villines", "valines", 0.83214286);
        jw.test("fitzjohn", "fitzgeoffrey", 0.68055556);
        jw.test("decrepon", "hardaknutsson", 0.43589744);
        jw.test("deswynnerton", "swinnerton", 0.80925926);
        jw.test("terriot", "terriau", 0.88571429);
        jw.test("detaillefer", "angouleme", 0.6026936);
        jw.test("fleitel", "flatel", 0.82603175);
        jw.test("temperley", "timperley", 0.82083333);
        jw.test("dentzer", "henckel", 0.61904762);
        jw.test("provencher", "provancher", 0.91555556);
        jw.test("deroet", "swynford", 0.43055556);
        jw.test("arganbright", "argenbright", 0.95757576);
        jw.test("vencill", "vincell", 0.82857143);
        jw.test("daingerfield", "dangerfield", 0.90505051);
        jw.test("selbee", "blott", 0.45555556);
        jw.test("berkeley", "martiau", 0.42261905);
        jw.test("cavendishbentinck", "bentinck", 0.50637255);
        jw.test("mcmurrough", "leinster", 0.40833333);
        jw.test("debraose", "briose", 0.81944444);
        jw.test("turberville", "tuberville", 0.94909091);
        jw.test("reinschmidt", "cuntze", 0.33838384);
        jw.test("kember", "thember", 0.84920635);
        jw.test("vancouwenhoven", "couwenhoven", 0.77705628);
        jw.test("fenstermaker", "fenstemaker", 0.9469697);
        jw.test("oberbroeckling", "oberbrockling", 0.98571429);
        jw.test("hems", "herns", 0.82666667);
        jw.test("fitzmaurice", "fitzmorris", 0.86727273);
        jw.test("mannon", "manon", 0.91444444);
        jw.test("peddicord", "petticord", 0.88148148);
        jw.test("flowerdew", "yeardly", 0.58730159);
        jw.test("shufflebotham", "shufflebottom", 0.93846154);
        jw.test("facteau", "facto", 0.87428571);
        jw.test("aldin", "nalle", 0.6);
        jw.test("helphenstine", "helphinstine", 0.93030303);
        jw.test("debesford", "besford", 0.87830688);
        jw.test("fitzalan", "goushill", 0.41666667);
        jw.test("riseborough", "roseborough", 0.85545455);
        jw.test("gruffydd", "rhys", 0.58333333);
        jw.test("hornberger", "homberger", 0.81703704);
        jw.test("tattershall", "tatarsole", 0.796633);
        jw.test("taillefer", "angouleme", 0.62962963);
        jw.test("reierson", "rierson", 0.91964286);
        jw.test("wrinkle", "rinkle", 0.95238095);
        jw.test("goldhatch", "tritton", 0.41798942);
        jw.test("sumarlidasson", "somerledsson", 0.83269231);
        jw.test("amberg", "glatfelder", 0.51111111);
        jw.test("raistrick", "rastrick", 0.9037037);
        jw.test("bajolet", "bayol", 0.83238095);
        jw.test("billung", "sachsen", 0.42857143);
        jw.test("delazouche", "zouche", 0.7);
        jw.test("springham", "springhorn", 0.89555556);
        jw.test("deserres", "dessert", 0.84583333);
        jw.test("gendre", "bourgery", 0.52777778);
        jw.test("braconie", "brackhonge", 0.915);
        jw.test("pleydellbouverie", "bouverie", 0.45833333);
        jw.test("fricks", "frix", 0.825);
        jw.test("plamondon", "plomondon", 0.87407407);
        jw.test("aubigny", "albini", 0.77142857);
        jw.test("freemanmitford", "mitford", 0.43650794);
        jw.test("fightmaster", "fight", 0.89090909);
        jw.test("wye", "why", 0.55555556);
        jw.test("birtwistle", "bertwistle", 0.87333333);
        jw.test("lautzenheiser", "lautzenhiser", 0.96794872);
        jw.test("puntenney", "puntney", 0.92698413);
        jw.test("demaranville", "demoranville", 0.93989899);
        jw.test("tellico", "clan", 0.46428571);
        jw.test("doublehead", "cornblossom", 0.52424242);
        jw.test("landvatter", "merckle", 0.41428571);
        jw.test("smy", "sury", 0.75);
        jw.test("macvane", "mcvane", 0.90714286);
        jw.test("grimoult", "sedilot", 0.60714286);
        jw.test("walgrave", "waldegrave", 0.895);
        jw.test("moytoy", "oconostota", 0.48888889);
        jw.test("steenbergen", "stenbergen", 0.90878788);
        jw.test("wolfensberger", "wolfersberger", 0.91923077);
        jw.test("lydecker", "leydecker", 0.85416667);
        jw.test("scheele", "schule", 0.84777778);
        jw.test("mountbatten", "battenberg", 0.58484848);
        jw.test("detalvas", "talvace", 0.7797619);
        jw.test("zwiefelhofer", "zwifelhofer", 0.91691919);
        jw.test("hawbaker", "hawbecker", 0.90198413);
        jw.test("degrandison", "grandson", 0.82575758);
        jw.test("ardouin", "badeau", 0.64285714);
        jw.test("loughmiller", "laughmiller", 0.94545455);
        jw.test("verdun", "ardenne", 0.66269841);
        jw.test("jorisse", "joire", 0.87047619);
        jw.test("wentworthfitzwilliam", "fitzwilliam", 0.78939394);
        jw.test("cornforth", "comforth", 0.86931217);
        jw.test("vanschouwen", "cornelissen", 0.56969697);
        jw.test("fuselier", "fusillier", 0.88564815);
        jw.test("frederickse", "lubbertsen", 0.42121212);
        jw.test("haraldsson", "forkbeard", 0.54444444);
        jw.test("fausnaugh", "fosnaugh", 0.81011905);
        jw.test("aymard", "emard", 0.73888889);
        jw.test("ingaldesthorpe", "ingoldsthrop", 0.90934343);
        jw.test("blennerhassett", "blaverhasset", 0.85269841);
        jw.test("ednywain", "edwyn", 0.84666667);
        jw.test("aubigny", "daubigny", 0.95833333);
        jw.test("hinderliter", "henderliter", 0.91545455);
        jw.test("deroucy", "rouci", 0.79047619);
        jw.test("dechow", "dago", 0.61111111);
        jw.test("kenchington", "kinchington", 0.88545455);
        jw.test("levere", "lavere", 0.78);
        jw.test("denivelles", "itta", 0.45);
        jw.test("delusignan", "lusignam", 0.85833333);
        jw.test("decressingham", "cressingham", 0.91841492);
        jw.test("austerfield", "osterfield", 0.90606061);
        jw.test("esten", "eustance", 0.79);
        jw.test("maves", "mebs", 0.63333333);
        jw.test("wieneke", "wineke", 0.87301587);
        jw.test("deliercourt", "juillet", 0.56168831);
        jw.test("auxerre", "argengau", 0.49007937);
        jw.test("beedell", "budell", 0.80428571);
        jw.test("muscoe", "hucklescott", 0.62929293);
        jw.test("feese", "fuse", 0.67222222);
        jw.test("laughinghouse", "lathinghouse", 0.88826729);
        jw.test("decrocketagne", "crocketagne", 0.7972028);
        jw.test("petitpas", "bugaret", 0.3452381);
        jw.test("leatherbarrow", "letherbarrow", 0.91282051);
        jw.test("goughcalthorpe", "calthorpe", 0.76984127);
        jw.test("stooksbury", "stookesberry", 0.93);
        jw.test("leichleiter", "lechleiter", 0.92242424);
        jw.test("devalletort", "valletort", 0.86531987);
        jw.test("duranceau", "duranso", 0.85396825);
        jw.test("ordepowlett", "powlett", 0.78354978);
        jw.test("freudenberg", "frendenberg", 0.93424242);
        jw.test("featherstonhaugh", "featherstonehaugh", 0.98823529);
        jw.test("hepburnstuartforbestrefusis", "trefusis", 0.56856261);
        jw.test("minkrevicius", "minkavitch", 0.85666667);
        jw.test("stuedemann", "studeman", 0.92416667);
        jw.test("frande", "andersdotter", 0.61666667);
        jw.test("alwyn", "joan", 0.48333333);
        jw.test("abendschon", "obenchain", 0.75555556);
        jw.test("landvatter", "varonica", 0.55);
        jw.test("dewindsor", "fitzotho", 0.49074074);
        jw.test("renkenberger", "rinkenberger", 0.84090909);
        jw.test("volkertsen", "noorman", 0.57619048);
        jw.test("casaubon", "casobon", 0.86944444);
        jw.test("decherleton", "cherlton", 0.86742424);
        jw.test("karraker", "kanaker", 0.80634921);
        jw.test("sinquefield", "sinkfield", 0.88127104);
        jw.test("lycon", "laican", 0.73);
        jw.test("cyphert", "seyphert", 0.75793651);
        jw.test("humphrie", "umfery", 0.625);
        jw.test("loignon", "longnon", 0.83492063);
        jw.test("cletheroe", "clitheroe", 0.84074074);
        jw.test("oesterle", "osterle", 0.83392857);
        jw.test("evemy", "evering", 0.67619048);
        jw.test("niquette", "nequette", 0.83928571);
        jw.test("lemeunier", "daubigeon", 0.46296296);
        jw.test("hartsvelder", "hartzfelder", 0.92727273);
        jw.test("beiersdorf", "biersdorf", 0.93666667);
        jw.test("destroismaisons", "destrosmaisons", 0.95809524);
        jw.test("warminger", "wanninger", 0.84338624);
        jw.test("demoleyns", "molines", 0.78571429);
        jw.test("chetwyndstapylton", "stapylton", 0.47657952);
        jw.test("woodville", "wydvile", 0.85714286);
        jw.test("fivekiller", "ghigau", 0.42222222);
        jw.test("rochet", "garrigues", 0.51851852);
        jw.test("leyendecker", "lyendecker", 0.85272727);
        jw.test("auringer", "oranger", 0.81349206);
        jw.test("twelftree", "twelvetree", 0.91277778);
        jw.test("roeloffse", "kierstede", 0.5462963);
        jw.test("stalsworth", "stolsworth", 0.88740741);
        jw.test("jarry", "rapin", 0.46666667);
        jw.test("lawter", "lantersee", 0.83703704);
        jw.test("andrewartha", "andrawartha", 0.90363636);
        jw.test("requa", "regna", 0.78666667);
        jw.test("devaloines", "volognes", 0.72777778);
        jw.test("featherstonhaugh", "fetherstonbaugh", 0.91079365);
        jw.test("sacherell", "searth", 0.66296296);
        jw.test("peerenboom", "perenboom", 0.9437037);
        jw.test("coeffes", "forestier", 0.62328042);
        jw.test("dewease", "duese", 0.73642857);
        jw.test("schackmann", "shackman", 0.9025);
        jw.test("breidenbaugh", "bridenbaugh", 0.95353535);
        jw.test("schollenberger", "shollenberger", 0.97857143);
        jw.test("neumeister", "newmaster", 0.8226455);
        jw.test("bettesworth", "betsworth", 0.90572391);
        jw.test("demedici", "medici", 0.86111111);
        jw.test("volkertsen", "holgersen", 0.82592593);
        jw.test("delusignan", "lusigan", 0.85238095);
        jw.test("elchert", "elkhart", 0.84761905);
        jw.test("detaillebois", "taillebois", 0.87777778);
        jw.test("fagelson", "feygelson", 0.85297619);
        jw.test("burdeshaw", "burtashaw", 0.86296296);
        jw.test("vanderburchgraeff", "burchgraeff", 0.76114082);
        jw.test("tiptoft", "tybotot", 0.63095238);
        jw.test("crepon", "forkbeard", 0.5);
        jw.test("rugglesbrise", "brise", 0.50555556);
        jw.test("grawbarger", "grauberger", 0.8775);
        jw.test("brassier", "decheilus", 0.32407407);
        jw.test("coningsby", "connyngesby", 0.85210438);
        jw.test("barneycastle", "barnacastle", 0.92848485);
        jw.test("degreystoke", "greystock", 0.83038721);
        jw.test("streitenberger", "strattenbarger", 0.83636364);
        jw.test("manitouabeouich", "manithabehich", 0.89771562);
        jw.test("haddleton", "addleton", 0.96296296);
        jw.test("trethurffe", "tretford", 0.83666667);
        jw.test("jaeckler", "margaretha", 0.44722222);
        jw.test("braeutigam", "braitigam", 0.89824074);
        jw.test("bacorn", "bakehorn", 0.85555556);
        jw.test("burkenbine", "birkinbine", 0.8425);
        jw.test("paulo", "campot", 0.57777778);
        jw.test("essenmacher", "eunmaker", 0.6540404);
        jw.test("decrocketagne", "crocketagni", 0.79277389);
        jw.test("unruhe", "kornmann", 0.36111111);
        jw.test("baesemann", "baseman", 0.9026455);
        jw.test("worchester", "worsestor", 0.84481481);
        jw.test("reidelberger", "rudelberger", 0.80272727);
        jw.test("bradtmueller", "bradmiller", 0.90777778);
        jw.test("schreckengast", "shreckengast", 0.92692308);
        jw.test("eisentrout", "isentrout", 0.92962963);
        jw.test("fivekiller", "kingfisher", 0.67777778);
        jw.test("jinright", "ginwright", 0.88425926);
        jw.test("reichenberger", "richenberger", 0.85192308);
        jw.test("langehennig", "laughennig", 0.89521886);
        jw.test("muttlebury", "mattleberry", 0.85818182);
        jw.test("cullumbine", "columbine", 0.86916667);
        jw.test("badlesmere", "northampton", 0.46060606);
        jw.test("oxier", "ockshire", 0.65833333);
        jw.test("anway", "amvay", 0.76);
        jw.test("wagenseller", "wagonseller", 0.91090909);
        jw.test("siebenthaler", "sevendollar", 0.69227994);
        jw.test("vannoorstrant", "juriaens", 0.51923077);
        jw.test("grundvig", "gumdeig", 0.80178571);
        jw.test("freudenberger", "frendenberger", 0.94465812);
        jw.test("schinbeckler", "shinbeckler", 0.89318182);
        jw.test("stautzenberger", "stantzenberger", 0.93076923);
        jw.test("molandersmolandes", "molandes", 0.89411765);
        jw.test("altstaetter", "allstetter", 0.86558923);
        jw.test("moredock", "nedock", 0.75277778);
        jw.test("bouslaugh", "baughlough", 0.68306878);
        jw.test("schoenbachler", "schoenbaechler", 0.95494505);
        jw.test("tetterton", "letterton", 0.84259259);
        jw.test("korsing", "curring", 0.71428571);
        jw.test("breckheimer", "brickheimer", 0.87151515);
        jw.test("doors", "streypress", 0.36666667);
        jw.test("flattum", "flattenn", 0.86785714);
        jw.test("demontmorency", "montmorency", 0.94871795);
        jw.test("andrieszen", "larens", 0.71111111);
        jw.test("hughesdaeth", "daeth", 0.36060606);
        jw.test("cullumbine", "colleunbine", 0.78318182);
        jw.test("twisletonwykehamfiennes", "fiennes", 0.51055901);
        jw.test("scherber", "sharver", 0.74178571);
        jw.test("coerten", "harmens", 0.50793651);
        jw.test("pitres", "fitzroger", 0.7037037);
        jw.test("degloucester", "fitzroger", 0.50925926);
        jw.test("sevestre", "delessart", 0.66018519);
        jw.test("larzelere", "larzalere", 0.90555556);
        jw.test("bargsley", "bayeley", 0.77047619);
        jw.test("flockirth", "flogerth", 0.86388889);
        jw.test("euteneier", "eutencies", 0.88253968);
        jw.test("smelker", "schmelcher", 0.83714286);
        jw.test("auchincloss", "anchencloss", 0.85757576);
        jw.test("during", "shaumloffel", 0.41919192);
        jw.test("arizmendi", "arismendez", 0.87814815);
        jw.test("otterlifter", "wawli", 0.52727273);
        jw.test("wackerle", "weckerla", 0.79642857);
        jw.test("manselpleydell", "pleydell", 0.73214286);
        jw.test("schwabenlender", "schwabenlander", 0.95604396);
        jw.test("hemmesch", "hemish", 0.87361111);
        jw.test("austerfield", "hosterfield", 0.87878788);
        jw.test("deetherstone", "detherston", 0.92888889);
        jw.test("rauhuff", "rowhuff", 0.76857143);
        jw.test("yorek", "jarek", 0.73333333);
        jw.test("hennikermajor", "henniker", 0.92307692);
        jw.test("depitres", "fitzroger", 0.64814815);
        jw.test("riedmueller", "reidmiller", 0.84878788);
        jw.test("culm", "culin", 0.84833333);
        jw.test("jonah", "joney", 0.81333333);
        jw.test("heckingbottom", "hickingbottom", 0.92884615);
        jw.test("schnorenberg", "schnorrenberg", 0.95128205);
        jw.test("hotzenbella", "hotzenpeller", 0.91363636);
        jw.test("theiring", "tyring", 0.7775);
        jw.test("dieffenwierth", "diffenweirth", 0.93504274);
        jw.test("haakonsdatter", "haakonson", 0.86324786);
        jw.test("martinvegue", "vegue", 0.43030303);
        jw.test("threapleton", "thrippleton", 0.88922559);
        jw.test("kirberger", "moelich", 0.33597884);
        jw.test("hinneschied", "henneschid", 0.88212121);
        jw.test("palmermorewood", "morewood", 0.81547619);
        jw.test("guicciardini", "guciardini", 0.84888889);
        jw.test("fregia", "fruger", 0.69444444);
        jw.test("braunberger", "bramberg", 0.88598485);
        jw.test("katterheinrich", "katterhenrich", 0.97032967);
        jw.test("bolibaugh", "bolebough", 0.8962963);
        jw.test("bechtelheimer", "becktelheimer", 0.92521368);
        jw.test("detibetot", "tybotot", 0.75661376);
        jw.test("encke", "ink", 0.68888889);
        jw.test("kettleborough", "kettleboro", 0.95384615);
        jw.test("whittingstall", "whettingstall", 0.93675214);
        jw.test("baggenstoss", "backenstoss", 0.9030303);
        jw.test("ardion", "rabouin", 0.71587302);
        jw.test("wittelsbach", "palatine", 0.53787879);
        jw.test("dechaux", "chapelain", 0.50529101);
        jw.test("vancortenbosch", "cortenbosch", 0.83766234);
        jw.test("swyersexey", "sexey", 0.65);
        jw.test("deherville", "sohier", 0.52222222);
        jw.test("kaesemeyer", "kasemeyer", 0.88444444);
        jw.test("righthouse", "wrighthouse", 0.96969697);
        jw.test("coeffes", "fourestier", 0.6047619);
        jw.test("kemeystynte", "tynte", 0.51313131);
        jw.test("goenner", "toennes", 0.80952381);
        jw.test("schattschneider", "schatschneider", 0.98666667);
        jw.test("masonheimer", "masenheimer", 0.88757576);
        jw.test("knutti", "margaritha", 0.34444444);
        jw.test("stoughtenger", "stoutenger", 0.92666667);
        jw.test("boeckhout", "elswaerts", 0.48148148);
        jw.test("nohrenhold", "nornhold", 0.91333333);
        jw.test("transdotter", "thrandsdotter", 0.92657343);
        jw.test("vansintern", "sintern", 0.75714286);
        jw.test("knatchbullhugessen", "hugessen", 0.40740741);
        jw.test("deshayes", "dehais", 0.80222222);
        jw.test("bornbach", "bomback", 0.82380952);
        jw.test("aupry", "auprybertrand", 0.87692308);
        jw.test("loewenhagen", "lowenhagen", 0.89575758);
        jw.test("thorsteinsdottir", "thurstenson", 0.68244949);
        jw.test("schwinghammer", "swinghammer", 0.8993007);
        jw.test("mickelborough", "mickleburrough", 0.92271062);
        jw.test("friedenberg", "freedenberg", 0.89818182);
        jw.test("houmes", "hornnes", 0.7968254);
        jw.test("mignon", "guiet", 0.41111111);
        jw.test("mickelborough", "michelborough", 0.96410256);
        jw.test("tantaquidgeon", "quidgeon", 0.70512821);
        jw.test("duyts", "satyrs", 0.58888889);
        jw.test("highcock", "hycock", 0.8375);
        jw.test("cornelise", "esselsteyn", 0.61481481);
        jw.test("schoenthaler", "schonthaler", 0.92878788);
        jw.test("gillington", "gullotine", 0.75761905);
        jw.test("rogerstillstone", "tillstone", 0.71851852);
        jw.test("voidy", "vedie", 0.62222222);
        jw.test("smithdorrien", "dorrien", 0.76587302);
        jw.test("groethausen", "grothouse", 0.91043771);
        jw.test("schmeeckle", "schmuckle", 0.88777778);
        jw.test("grossnicklaus", "greenehouse", 0.55788656);
        jw.test("wilmotsitwell", "sitwell", 0.56630037);
        jw.test("boertgens", "harmense", 0.59351852);
        jw.test("koetterhagen", "katterhagen", 0.83272727);
        jw.test("berthelette", "barthelette", 0.82545455);
        jw.test("schoettler", "brechting", 0.53148148);
        jw.test("wescovich", "viscovich", 0.85185185);
        jw.test("etringer", "thilges", 0.69047619);
        jw.test("sigurdsson", "lodbrok", 0.46507937);
        jw.test("deligny", "bidon", 0.56507937);
        jw.test("winsofer", "hubbarde", 0.33333333);
        jw.test("straatmaker", "stratenmaker", 0.90848485);
        jw.test("warrenbuer", "warambour", 0.82888889);
        jw.test("ouderkerk", "heemstraat", 0.43333333);
        jw.test("comalander", "cumberlander", 0.69722222);
        jw.test("holtzendorff", "holsendorf", 0.91833333);
        jw.test("beirdneau", "birdno", 0.81666667);
    }

    #[test]
    fn jaccard_edge_cases() {
        let mut jac = Jaccard::new();
        jac.test("s", "s", 1.0);
        jac.test("", "", 1.0);
        jac.test("string", "", 0.0);
        jac.test("", "string", 0.0);
    }

    #[test]
    fn jaccard_test_cases() {
        let mut jac = Jaccard::new();
        jac.test("phillips", "philips", 0.875);
        jac.test("kelly", "kelley", 0.83333333);
        jac.test("wood", "woods", 0.8);
        jac.test("russell", "russel", 0.85714286);
        jac.test("macdonald", "mcdonald", 0.88888889);
        jac.test("hansen", "hanson", 0.71428571);
        jac.test("gray", "grey", 0.6);
        jac.test("petersen", "peterson", 0.77777778);
        jac.test("myers", "myres", 1.0);
        jac.test("chamberlain", "chamberlin", 0.90909091);
        jac.test("mullins", "mullens", 0.75);
        jac.test("olsen", "olson", 0.66666667);
        jac.test("pennington", "penington", 0.9);
        jac.test("livingston", "levingston", 0.81818182);
        jac.test("plantagenet", "lancaster", 0.42857143);
        jac.test("horner", "homer", 0.57142857);
        jac.test("featherstone", "featherston", 0.91666667);
        jac.test("higginbotham", "higgenbotham", 0.84615385);
        jac.test("plantagenet", "gaunt", 0.33333333);
        jac.test("asbury", "asberry", 0.625);
        jac.test("schumacher", "schumaker", 0.72727273);
        jac.test("reedy", "rudy", 0.5);
        jac.test("powhatan", "rolfe", 0.083333333);
        jac.test("landen", "austrasia", 0.071428571);
        jac.test("scotland", "canmore", 0.36363636);
        jac.test("woodbury", "woodberry", 0.7);
        jac.test("powhatan", "daughter", 0.23076923);
        jac.test("anjou", "jerusalem", 0.27272727);
        jac.test("zachary", "zackery", 0.55555556);
        jac.test("witherspoon", "weatherspoon", 0.76923077);
        jac.test("fenstermacher", "fenstermaker", 0.78571429);
        jac.test("hetherington", "heatherington", 0.92307692);
        jac.test("demeschines", "meschin", 0.63636364);
        jac.test("bretagne", "brittany", 0.45454545);
        jac.test("mormaer", "thane", 0.2);
        jac.test("sloman", "poythress", 0.15384615);
        jac.test("aetheling", "exile", 0.4);
        jac.test("barlowe", "almy", 0.22222222);
        jac.test("macmurrough", "macmurchada", 0.46666667);
        jac.test("tourault", "archambault", 0.35714286);
        jac.test("dechatellerault", "chatellerault", 0.86666667);
        jac.test("roosevelt", "rooswell", 0.54545455);
        jac.test("trico", "rapalje", 0.090909091);
        jac.test("hatherly", "hanford", 0.25);
        jac.test("carow", "roosevelt", 0.16666667);
        jac.test("maugis", "miville", 0.18181818);
        jac.test("decrepon", "hardaknutsson", 0.23529412);
        jac.test("deswynnerton", "swinnerton", 0.69230769);
        jac.test("drusus", "nero", 0.11111111);
        jac.test("verdon", "brouwer", 0.3);
        jac.test("cavendishbentinck", "bentinck", 0.47058824);
        jac.test("mcmurrough", "leinster", 0.058823529);
        jac.test("reinschmidt", "cuntze", 0.30769231);
        jac.test("pichon", "sevestre", 0.0);
        jac.test("oberbroeckling", "oberbrockling", 0.92857143);
        jac.test("shufflebotham", "shufflebottom", 0.73333333);
        jac.test("fitzalan", "goushill", 0.14285714);
        jac.test("taillefer", "angouleme", 0.28571429);
        jac.test("sumarlidasson", "somerledsson", 0.5625);
        jac.test("billung", "sachsen", 0.076923077);
        jac.test("springham", "springhorn", 0.58333333);
        jac.test("aubigny", "albini", 0.44444444);
        jac.test("landvatter", "merckle", 0.21428571);
        jac.test("kluczykowski", "kluck", 0.41666667);
        jac.test("wentworthfitzwilliam", "fitzwilliam", 0.55);
        jac.test("vanschouwen", "cornelissen", 0.375);
        jac.test("haraldsson", "forkbeard", 0.26666667);
        jac.test("deliercourt", "juillet", 0.38461538);
        jac.test("behol", "jennet", 0.1);
        jac.test("goughcalthorpe", "calthorpe", 0.64285714);
        jac.test("tietsoort", "willemszen", 0.1875);
        jac.test("featherstonhaugh", "featherstonehaugh", 0.94117647);
        jac.test("hepburnstuartforbestrefusis", "trefusis", 0.2962963);
        jac.test("ynyr", "gwent", 0.125);
        jac.test("dewindsor", "fitzotho", 0.13333333);
        jac.test("destroismaisons", "destrosmaisons", 0.93333333);
        jac.test("chetwyndstapylton", "stapylton", 0.52941176);
        jac.test("fivekiller", "ghigau", 0.066666667);
        jac.test("featherstonhaugh", "fetherstonbaugh", 0.82352941);
        jac.test("minkrevicius", "minkevich", 0.61538462);
        jac.test("vanderburchgraeff", "burchgraeff", 0.64705882);
        jac.test("essenmacher", "eunmaker", 0.46153846);
        jac.test("siebenthaler", "sevendollar", 0.4375);
        jac.test("twisletonwykehamfiennes", "fiennes", 0.30434783);
        jac.test("degloucester", "fitzroger", 0.3125);
        jac.test("during", "shaumloffel", 0.0625);
        jac.test("alcombrack", "alkenbrack", 0.53846154);
    }
}
