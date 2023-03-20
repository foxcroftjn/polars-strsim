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
    function: fn(&str, &str) -> f64,
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
            .collect::<Vec<Option<f64>>>(),
    );

    Ok(out.into_series())
}

// adapted from https://en.wikipedia.org/wiki/Levenshtein_distance#Iterative_with_two_matrix_rows
fn compute_levenshtein(a: &str, b: &str) -> f64 {
    if (a == "" && b == "") || (a == b) {
        return 1.0;
    }
    let a = a.chars().collect::<Vec<_>>();
    let b = b.chars().collect::<Vec<_>>();
    let mut matrix: Vec<_> = (0..=b.len()).map(|i| [i, 0]).collect();
    for i in 0..a.len() {
        let v0 = i % 2;
        let v1 = (i + 1) % 2;
        matrix[0][v1] = i + 1;
        for j in 0..b.len() {
            matrix[j + 1][v1] = if a[i] == b[j] {
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

pub(super) fn parallel_levenshtein(
    df: DataFrame,
    col_a: &str,
    col_b: &str,
    name: &str,
) -> PolarsResult<Series> {
    Ok(parallel_apply(df, col_a, col_b, name, compute_levenshtein)?)
}

fn compute_jaro(a: &str, b: &str) -> f64 {
    if (a == "" && b == "") || (a == b) {
        return 1.0;
    } else if a == "" || b == "" {
        return 0.0;
    }
    let a = a.chars().collect::<Vec<_>>();
    let b = b.chars().collect::<Vec<_>>();
    if a.len() == 1 && b.len() == 1 {
        return if a[0] == b[0] { 1.0 } else { 0.0 };
    }
    let bound = a.len().max(b.len()) / 2 - 1;
    let mut m = 0;
    let mut flagged = vec![[false; 2]; a.len().max(b.len())];
    for i in 0..a.len() {
        let lowerbound = if bound > i { 0 } else { i - bound };
        let upperbound = (i + bound).min(b.len() - 1);
        for j in lowerbound..=upperbound {
            if a[i] == b[j] && !flagged[j][1] {
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

pub(super) fn parallel_jaro(
    df: DataFrame,
    col_a: &str,
    col_b: &str,
    name: &str,
) -> PolarsResult<Series> {
    Ok(parallel_apply(df, col_a, col_b, name, compute_jaro)?)
}

fn compute_jaro_winkler(a: &str, b: &str) -> f64 {
    let jaro_similarity = compute_jaro(a, b);
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
        compute_jaro_winkler,
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;
    const THRESHOLD: f64 = 0.00000001;

    fn test_levenshtein(a: &str, b: &str, expected_result: f64) {
        assert!(
            (compute_levenshtein(a, b) - expected_result).abs() < THRESHOLD,
            "\"{}\", \"{}\" was computed as {}, expected {}",
            a,
            b,
            compute_levenshtein(a, b),
            expected_result
        );
    }

    #[test]
    fn levenshtein_edge_cases() {
        test_levenshtein("s", "s", 1.0);
        test_levenshtein("", "", 1.0);
        test_levenshtein("string", "", 0.0);
        test_levenshtein("", "string", 0.0);
    }

    #[test]
    fn levenshtein_test_cases() {
        test_levenshtein("phillips", "philips", 0.875);
        test_levenshtein("kelly", "kelley", 0.83333333);
        test_levenshtein("wood", "woods", 0.8);
        test_levenshtein("russell", "russel", 0.85714286);
        test_levenshtein("macdonald", "mcdonald", 0.88888889);
        test_levenshtein("gray", "grey", 0.75);
        test_levenshtein("myers", "myres", 0.6);
        test_levenshtein("chamberlain", "chamberlin", 0.90909091);
        test_levenshtein("pennington", "penington", 0.9);
        test_levenshtein("ziegler", "zeigler", 0.71428571);
        test_levenshtein("hendrix", "hendricks", 0.66666667);
        test_levenshtein("abel", "able", 0.5);
        test_levenshtein("plantagenet", "lancaster", 0.45454545);
        test_levenshtein("featherstone", "featherston", 0.91666667);
        test_levenshtein("shackelford", "shackleford", 0.81818182);
        test_levenshtein("hazelwood", "hazlewood", 0.77777778);
        test_levenshtein("plantagenet", "gaunt", 0.27272727);
        test_levenshtein("powhatan", "rolfe", 0.125);
        test_levenshtein("landen", "austrasia", 0.0);
        test_levenshtein("fenstermacher", "fenstermaker", 0.84615385);
        test_levenshtein("hetherington", "heatherington", 0.92307692);
        test_levenshtein("defalaise", "arletta", 0.11111111);
        test_levenshtein("demeschines", "meschin", 0.63636364);
        test_levenshtein("archambault", "archambeau", 0.72727273);
        test_levenshtein("mormaer", "thane", 0.14285714);
        test_levenshtein("fitzpiers", "piers", 0.55555556);
        test_levenshtein("normandy", "brittany", 0.25);
        test_levenshtein("aetheling", "exile", 0.22222222);
        test_levenshtein("barlowe", "almy", 0.28571429);
        test_levenshtein("macmurrough", "macmurchada", 0.54545455);
        test_levenshtein("tourault", "archambault", 0.36363636);
        test_levenshtein("detoeni", "toni", 0.57142857);
        test_levenshtein("dechatellerault", "chatellerault", 0.86666667);
        test_levenshtein("hatherly", "hanford", 0.375);
        test_levenshtein("christoffersen", "christofferson", 0.92857143);
        test_levenshtein("blackshear", "blackshire", 0.7);
        test_levenshtein("fitzjohn", "fitzgeoffrey", 0.41666667);
        test_levenshtein("decrepon", "hardaknutsson", 0.23076923);
        test_levenshtein("dentzer", "henckel", 0.42857143);
        test_levenshtein("hignite", "hignight", 0.625);
        test_levenshtein("selbee", "blott", 0.16666667);
        test_levenshtein("cavendishbentinck", "bentinck", 0.47058824);
        test_levenshtein("reinschmidt", "cuntze", 0.090909091);
        test_levenshtein("vancouwenhoven", "couwenhoven", 0.78571429);
        test_levenshtein("aldin", "nalle", 0.2);
        test_levenshtein("offley", "thoroughgood", 0.083333333);
        test_levenshtein("sumarlidasson", "somerledsson", 0.69230769);
        test_levenshtein("wye", "why", 0.33333333);
        test_levenshtein("landvatter", "merckle", 0.1);
        test_levenshtein("moytoy", "oconostota", 0.3);
        test_levenshtein("mountbatten", "battenberg", 0.18181818);
        test_levenshtein("wentworthfitzwilliam", "fitzwilliam", 0.55);
        test_levenshtein("ingaldesthorpe", "ingoldsthrop", 0.64285714);
        test_levenshtein("munning", "munningmunny", 0.58333333);
        test_levenshtein("sinor", "snier", 0.4);
        test_levenshtein("featherstonhaugh", "featherstonehaugh", 0.94117647);
        test_levenshtein("hepburnstuartforbestrefusis", "trefusis", 0.2962963);
        test_levenshtein("destroismaisons", "destrosmaisons", 0.93333333);
        test_levenshtein("demoleyns", "molines", 0.44444444);
        test_levenshtein("chetwyndstapylton", "stapylton", 0.52941176);
        test_levenshtein("vanderburchgraeff", "burchgraeff", 0.64705882);
        test_levenshtein("manitouabeouich", "manithabehich", 0.73333333);
        test_levenshtein("decrocketagne", "crocketagni", 0.76923077);
        test_levenshtein("vannoorstrant", "juriaens", 0.15384615);
        test_levenshtein("twisletonwykehamfiennes", "fiennes", 0.30434783);
        test_levenshtein("hennikermajor", "henniker", 0.61538462);
        test_levenshtein("haakonsdatter", "haakonson", 0.53846154);
        test_levenshtein("aupry", "auprybertrand", 0.38461538);
        test_levenshtein("thorsteinsdottir", "thurstenson", 0.5625);
        test_levenshtein("grossnicklaus", "greenehouse", 0.30769231);
    }

    fn test_jaro(a: &str, b: &str, expected_result: f64) {
        assert!(
            (compute_jaro(a, b) - expected_result).abs() < THRESHOLD,
            "\"{}\", \"{}\" was computed as {}, expected {}",
            a,
            b,
            compute_jaro(a, b),
            expected_result
        );
    }

    #[test]
    fn jaro_edge_cases() {
        test_jaro("s", "a", 0.0);
        test_jaro("s", "s", 1.0);
        test_jaro("", "", 1.0);
        test_jaro("string", "", 0.0);
        test_jaro("", "string", 0.0);
    }

    #[test]
    fn jaro_test_cases() {
        test_jaro("phillips", "philips", 0.95833333);
        test_jaro("kelly", "kelley", 0.94444444);
        test_jaro("wood", "woods", 0.93333333);
        test_jaro("russell", "russel", 0.95238095);
        test_jaro("macdonald", "mcdonald", 0.96296296);
        test_jaro("hansen", "hanson", 0.88888889);
        test_jaro("gray", "grey", 0.83333333);
        test_jaro("petersen", "peterson", 0.91666667);
        test_jaro("chamberlain", "chamberlin", 0.96969697);
        test_jaro("blankenship", "blankinship", 0.87272727);
        test_jaro("dickinson", "dickenson", 0.92592593);
        test_jaro("mullins", "mullens", 0.9047619);
        test_jaro("olsen", "olson", 0.86666667);
        test_jaro("pennington", "penington", 0.85555556);
        test_jaro("cunningham", "cuningham", 0.92962963);
        test_jaro("gillespie", "gillispie", 0.88425926);
        test_jaro("callaway", "calloway", 0.86904762);
        test_jaro("mackay", "mckay", 0.87777778);
        test_jaro("christensen", "christenson", 0.93939394);
        test_jaro("mcallister", "mcalister", 0.96666667);
        test_jaro("cleveland", "cleaveland", 0.89259259);
        test_jaro("denison", "dennison", 0.86309524);
        test_jaro("livingston", "levingston", 0.8962963);
        test_jaro("hendrix", "hendricks", 0.84126984);
        test_jaro("stillwell", "stilwell", 0.9212963);
        test_jaro("schaefer", "schafer", 0.91071429);
        test_jaro("moseley", "mosley", 0.8968254);
        test_jaro("plantagenet", "lancaster", 0.68181818);
        test_jaro("horner", "homer", 0.73888889);
        test_jaro("whittington", "whitington", 0.9030303);
        test_jaro("featherstone", "featherston", 0.97222222);
        test_jaro("reeder", "reader", 0.82222222);
        test_jaro("scarborough", "scarbrough", 0.93636364);
        test_jaro("higginbotham", "higgenbotham", 0.94444444);
        test_jaro("flanagan", "flannagan", 0.87962963);
        test_jaro("debeauchamp", "beauchamp", 0.9023569);
        test_jaro("donovan", "donavan", 0.84920635);
        test_jaro("plantagenet", "gaunt", 0.62424242);
        test_jaro("reedy", "rudy", 0.78333333);
        test_jaro("egan", "eagan", 0.85);
        test_jaro("powhatan", "rolfe", 0.44166667);
        test_jaro("landen", "austrasia", 0.42592593);
        test_jaro("pelletier", "peltier", 0.87830688);
        test_jaro("arrowood", "arwood", 0.86111111);
        test_jaro("canterbury", "canterberry", 0.90606061);
        test_jaro("scotland", "canmore", 0.51190476);
        test_jaro("depercy", "percy", 0.83809524);
        test_jaro("rau", "raw", 0.77777778);
        test_jaro("powhatan", "daughter", 0.47222222);
        test_jaro("anjou", "jerusalem", 0.54074074);
        test_jaro("deferrers", "ferrers", 0.83068783);
        test_jaro("devermandois", "vermandois", 0.91111111);
        test_jaro("bainbridge", "bambridge", 0.85462963);
        test_jaro("zachary", "zackery", 0.80952381);
        test_jaro("witherspoon", "weatherspoon", 0.88080808);
        test_jaro("breckenridge", "brackenridge", 0.91414141);
        test_jaro("fenstermacher", "fenstermaker", 0.92094017);
        test_jaro("declermont", "clermont", 0.89166667);
        test_jaro("hetherington", "heatherington", 0.97435897);
        test_jaro("defalaise", "arletta", 0.58862434);
        test_jaro("demeschines", "meschin", 0.83116883);
        test_jaro("benningfield", "beningfield", 0.94191919);
        test_jaro("bretagne", "brittany", 0.75);
        test_jaro("beresford", "berrisford", 0.81296296);
        test_jaro("wydeville", "woodville", 0.85185185);
        test_jaro("mormaer", "thane", 0.56190476);
        test_jaro("fitzpiers", "piers", 0.43703704);
        test_jaro("decourtenay", "courtenay", 0.82828283);
        test_jaro("debadlesmere", "badlesmere", 0.77777778);
        test_jaro("dewarenne", "warenne", 0.78306878);
        test_jaro("deroet", "roet", 0.80555556);
        test_jaro("demeschines", "meschines", 0.79124579);
        test_jaro("normandy", "brittany", 0.66666667);
        test_jaro("garnsey", "guernsey", 0.75793651);
        test_jaro("aetheling", "exile", 0.53333333);
        test_jaro("barlowe", "almy", 0.5952381);
        test_jaro("haraldsdatter", "haraldsdotter", 0.94871795);
        test_jaro("macmurrough", "macmurchada", 0.75757576);
        test_jaro("falaise", "arletta", 0.52380952);
        test_jaro("deberkeley", "berkeley", 0.80833333);
        test_jaro("tourault", "archambault", 0.62651515);
        test_jaro("valliance", "pray", 0.4537037);
        test_jaro("fischbach", "fishback", 0.78902116);
        test_jaro("dechatellerault", "chatellerault", 0.87863248);
        test_jaro("trico", "rapalje", 0.44761905);
        test_jaro("hatherly", "hanford", 0.60119048);
        test_jaro("aquitaine", "eleanor", 0.5026455);
        test_jaro("devere", "vere", 0.72222222);
        test_jaro("coppedge", "coppage", 0.81349206);
        test_jaro("rockefeller", "rockafellow", 0.77651515);
        test_jaro("rubenstein", "rubinstein", 0.85925926);
        test_jaro("roet", "swynford", 0.0);
        test_jaro("bodenheimer", "bodenhamer", 0.86902357);
        test_jaro("dehauteville", "hauteville", 0.81111111);
        test_jaro("maugis", "miville", 0.53968254);
        test_jaro("fitzjohn", "fitzgeoffrey", 0.68055556);
        test_jaro("decrepon", "hardaknutsson", 0.43589744);
        test_jaro("deswynnerton", "swinnerton", 0.80925926);
        test_jaro("detaillefer", "angouleme", 0.6026936);
        test_jaro("fleitel", "flatel", 0.78253968);
        test_jaro("temperley", "timperley", 0.80092593);
        test_jaro("dentzer", "henckel", 0.61904762);
        test_jaro("deroet", "swynford", 0.43055556);
        test_jaro("daingerfield", "dangerfield", 0.88131313);
        test_jaro("selbee", "blott", 0.45555556);
        test_jaro("berkeley", "martiau", 0.42261905);
        test_jaro("cavendishbentinck", "bentinck", 0.50637255);
        test_jaro("mcmurrough", "leinster", 0.40833333);
        test_jaro("debraose", "briose", 0.81944444);
        test_jaro("reinschmidt", "cuntze", 0.33838384);
        test_jaro("vancouwenhoven", "couwenhoven", 0.77705628);
        test_jaro("fenstermaker", "fenstemaker", 0.91161616);
        test_jaro("kerrich", "keridge", 0.71428571);
        test_jaro("oberbroeckling", "oberbrockling", 0.97619048);
        test_jaro("fitzmaurice", "fitzmorris", 0.77878788);
        test_jaro("flowerdew", "yeardly", 0.58730159);
        test_jaro("shufflebotham", "shufflebottom", 0.8974359);
        test_jaro("demontdidier", "montdidier", 0.84444444);
        test_jaro("facteau", "facto", 0.79047619);
        test_jaro("aldin", "nalle", 0.6);
        test_jaro("helphenstine", "helphinstine", 0.88383838);
        test_jaro("fitzalan", "goushill", 0.41666667);
        test_jaro("riseborough", "roseborough", 0.83939394);
        test_jaro("gruffydd", "rhys", 0.58333333);
        test_jaro("hornberger", "homberger", 0.7712963);
        test_jaro("tattershall", "tatarsole", 0.70947571);
        test_jaro("taillefer", "angouleme", 0.62962963);
        test_jaro("goldhatch", "tritton", 0.41798942);
        test_jaro("sumarlidasson", "somerledsson", 0.81410256);
        test_jaro("cuvellier", "cuvilje", 0.78571429);
        test_jaro("amberg", "glatfelder", 0.51111111);
        test_jaro("ruel", "ruelle", 0.88888889);
        test_jaro("billung", "sachsen", 0.42857143);
        test_jaro("delazouche", "zouche", 0.7);
        test_jaro("springham", "springhorn", 0.82592593);
        test_jaro("deserres", "dessert", 0.7797619);
        test_jaro("gendre", "bourgery", 0.52777778);
        test_jaro("braconie", "brackhonge", 0.85833333);
        test_jaro("pleydellbouverie", "bouverie", 0.45833333);
        test_jaro("plamondon", "plomondon", 0.84259259);
        test_jaro("aubigny", "albini", 0.74603175);
        test_jaro("freemanmitford", "mitford", 0.43650794);
        test_jaro("fightmaster", "fight", 0.81818182);
        test_jaro("wye", "why", 0.55555556);
        test_jaro("clabough", "clabo", 0.875);
        test_jaro("lautzenheiser", "lautzenhiser", 0.9465812);
        test_jaro("tellico", "clan", 0.46428571);
        test_jaro("doublehead", "cornblossom", 0.52424242);
        test_jaro("landvatter", "merckle", 0.41428571);
        test_jaro("grimoult", "sedilot", 0.60714286);
        test_jaro("kluczykowski", "kluck", 0.80555556);
        test_jaro("moytoy", "oconostota", 0.48888889);
        test_jaro("steenbergen", "stenbergen", 0.86969697);
        test_jaro("wolfensberger", "wolfersberger", 0.86538462);
        test_jaro("lydecker", "leydecker", 0.83796296);
        test_jaro("mountbatten", "battenberg", 0.58484848);
        test_jaro("hawbaker", "hawbecker", 0.83664021);
        test_jaro("degrandison", "grandson", 0.82575758);
        test_jaro("ardouin", "badeau", 0.64285714);
        test_jaro("verdun", "ardenne", 0.66269841);
        test_jaro("riemenschneider", "reimenschneider", 0.97777778);
        test_jaro("wentworthfitzwilliam", "fitzwilliam", 0.78939394);
        test_jaro("vanschouwen", "cornelissen", 0.56969697);
        test_jaro("frederickse", "lubbertsen", 0.42121212);
        test_jaro("haraldsson", "forkbeard", 0.54444444);
        test_jaro("ingaldesthorpe", "ingoldsthrop", 0.87049062);
        test_jaro("blennerhassett", "blaverhasset", 0.81587302);
        test_jaro("dechow", "dago", 0.61111111);
        test_jaro("levere", "lavere", 0.75555556);
        test_jaro("denivelles", "itta", 0.45);
        test_jaro("decressingham", "cressingham", 0.91841492);
        test_jaro("esten", "eustance", 0.76666667);
        test_jaro("maves", "mebs", 0.63333333);
        test_jaro("deliercourt", "juillet", 0.56168831);
        test_jaro("auxerre", "argengau", 0.49007937);
        test_jaro("delisoures", "lisours", 0.9);
        test_jaro("muscoe", "hucklescott", 0.62929293);
        test_jaro("feese", "fuse", 0.67222222);
        test_jaro("laughinghouse", "lathinghouse", 0.86033411);
        test_jaro("decrocketagne", "crocketagne", 0.7972028);
        test_jaro("petitpas", "bugaret", 0.3452381);
        test_jaro("leatherbarrow", "letherbarrow", 0.89102564);
        test_jaro("goughcalthorpe", "calthorpe", 0.76984127);
        test_jaro("stooksbury", "stookesberry", 0.88333333);
        test_jaro("devalletort", "valletort", 0.86531987);
        test_jaro("duranceau", "duranso", 0.75661376);
        test_jaro("ordepowlett", "powlett", 0.78354978);
        test_jaro("featherstonhaugh", "featherstonehaugh", 0.98039216);
        test_jaro("hepburnstuartforbestrefusis", "trefusis", 0.56856261);
        test_jaro("minkrevicius", "minkavitch", 0.76111111);
        test_jaro("frande", "andersdotter", 0.61666667);
        test_jaro("alwyn", "joan", 0.48333333);
        test_jaro("landvatter", "varonica", 0.55);
        test_jaro("dewindsor", "fitzotho", 0.49074074);
        test_jaro("renkenberger", "rinkenberger", 0.82323232);
        test_jaro("volkertsen", "noorman", 0.57619048);
        test_jaro("bottenfield", "bottomfield", 0.84175084);
        test_jaro("decherleton", "cherlton", 0.86742424);
        test_jaro("sinquefield", "sinkfield", 0.83038721);
        test_jaro("courchesne", "courchaine", 0.825);
        test_jaro("humphrie", "umfery", 0.625);
        test_jaro("loignon", "longnon", 0.79365079);
        test_jaro("oesterle", "osterle", 0.81547619);
        test_jaro("evemy", "evering", 0.67619048);
        test_jaro("niquette", "nequette", 0.82142857);
        test_jaro("lemeunier", "daubigeon", 0.46296296);
        test_jaro("hartsvelder", "hartzfelder", 0.87878788);
        test_jaro("destroismaisons", "destrosmaisons", 0.93015873);
        test_jaro("warminger", "wanninger", 0.8042328);
        test_jaro("chetwyndstapylton", "stapylton", 0.47657952);
        test_jaro("fivekiller", "ghigau", 0.42222222);
        test_jaro("rochet", "garrigues", 0.51851852);
        test_jaro("leyendecker", "lyendecker", 0.83636364);
        test_jaro("roeloffse", "kierstede", 0.5462963);
        test_jaro("jarry", "rapin", 0.46666667);
        test_jaro("lawter", "lantersee", 0.7962963);
        test_jaro("requa", "regna", 0.73333333);
        test_jaro("devaloines", "volognes", 0.72777778);
        test_jaro("featherstonhaugh", "fetherstonbaugh", 0.88849206);
        test_jaro("sacherell", "searth", 0.66296296);
        test_jaro("coeffes", "forestier", 0.62328042);
        test_jaro("dewease", "duese", 0.70714286);
        test_jaro("neumeister", "newmaster", 0.77830688);
        test_jaro("delusignan", "lusigan", 0.85238095);
        test_jaro("hearnsberger", "harnsberger", 0.8510101);
        test_jaro("vanderburchgraeff", "burchgraeff", 0.76114082);
        test_jaro("tiptoft", "tybotot", 0.63095238);
        test_jaro("crepon", "forkbeard", 0.5);
        test_jaro("rugglesbrise", "brise", 0.50555556);
        test_jaro("brassier", "decheilus", 0.32407407);
        test_jaro("coningsby", "connyngesby", 0.78872054);
        test_jaro("ingaldesthorpe", "ingoldsthorp", 0.90079365);
        test_jaro("streitenberger", "strattenbarger", 0.76623377);
        test_jaro("manitouabeouich", "manithabehich", 0.82952603);
        test_jaro("jaeckler", "margaretha", 0.44722222);
        test_jaro("paulo", "campot", 0.57777778);
        test_jaro("essenmacher", "eunmaker", 0.6540404);
        test_jaro("decrocketagne", "crocketagni", 0.79277389);
        test_jaro("unruhe", "kornmann", 0.36111111);
        test_jaro("reidelberger", "rudelberger", 0.78080808);
        test_jaro("bradtmueller", "bradmiller", 0.8462963);
        test_jaro("schreckengast", "shreckengast", 0.91880342);
        test_jaro("fivekiller", "kingfisher", 0.67777778);
        test_jaro("reichenberger", "richenberger", 0.83547009);
        test_jaro("muttlebury", "mattleberry", 0.84242424);
        test_jaro("jobidon", "bidon", 0.77142857);
        test_jaro("badlesmere", "northampton", 0.46060606);
        test_jaro("oxier", "ockshire", 0.65833333);
        test_jaro("siebenthaler", "sevendollar", 0.69227994);
        test_jaro("vannoorstrant", "juriaens", 0.51923077);
        test_jaro("stautzenberger", "stantzenberger", 0.9010989);
        test_jaro("molandersmolandes", "molandes", 0.82352941);
        test_jaro("altstaetter", "allstetter", 0.83198653);
        test_jaro("moredock", "nedock", 0.75277778);
        test_jaro("bouslaugh", "baughlough", 0.68306878);
        test_jaro("schoenbachler", "schoenbaechler", 0.92490842);
        test_jaro("doors", "streypress", 0.36666667);
        test_jaro("andrieszen", "larens", 0.71111111);
        test_jaro("hughesdaeth", "daeth", 0.36060606);
        test_jaro("cullumbine", "colleunbine", 0.75909091);
        test_jaro("twisletonwykehamfiennes", "fiennes", 0.51055901);
        test_jaro("scherber", "sharver", 0.71309524);
        test_jaro("coerten", "harmens", 0.50793651);
        test_jaro("pitres", "fitzroger", 0.7037037);
        test_jaro("degloucester", "fitzroger", 0.50925926);
        test_jaro("sevestre", "delessart", 0.66018519);
        test_jaro("smelker", "schmelcher", 0.81904762);
        test_jaro("during", "shaumloffel", 0.41919192);
        test_jaro("otterlifter", "wawli", 0.52727273);
        test_jaro("wackerle", "weckerla", 0.77380952);
        test_jaro("manselpleydell", "pleydell", 0.73214286);
        test_jaro("schwabenlender", "schwabenlander", 0.92673993);
        test_jaro("thurner", "thur", 0.85714286);
        test_jaro("rauhuff", "rowhuff", 0.74285714);
        test_jaro("hennikermajor", "henniker", 0.87179487);
        test_jaro("depitres", "fitzroger", 0.64814815);
        test_jaro("hotzenbella", "hotzenpeller", 0.85606061);
        test_jaro("haakonsdatter", "haakonson", 0.77207977);
        test_jaro("martinvegue", "vegue", 0.43030303);
        test_jaro("alcombrack", "alkenbrack", 0.8);
        test_jaro("kirberger", "moelich", 0.33597884);
        test_jaro("fregia", "fruger", 0.69444444);
        test_jaro("braunberger", "bramberg", 0.83712121);
        test_jaro("katterheinrich", "katterhenrich", 0.95054945);
        test_jaro("bechtelheimer", "becktelheimer", 0.89316239);
        test_jaro("encke", "ink", 0.68888889);
        test_jaro("kettleborough", "kettleboro", 0.92307692);
        test_jaro("ardion", "rabouin", 0.71587302);
        test_jaro("wittelsbach", "palatine", 0.53787879);
        test_jaro("dechaux", "chapelain", 0.50529101);
        test_jaro("vancortenbosch", "cortenbosch", 0.83766234);
        test_jaro("swyersexey", "sexey", 0.65);
        test_jaro("deherville", "sohier", 0.52222222);
        test_jaro("coeffes", "fourestier", 0.6047619);
        test_jaro("kemeystynte", "tynte", 0.51313131);
        test_jaro("knutti", "margaritha", 0.34444444);
        test_jaro("boeckhout", "elswaerts", 0.48148148);
        test_jaro("vansintern", "sintern", 0.75714286);
        test_jaro("knatchbullhugessen", "hugessen", 0.40740741);
        test_jaro("aupry", "auprybertrand", 0.79487179);
        test_jaro("bigot", "guillebour", 0.52222222);
        test_jaro("thorsteinsdottir", "thurstenson", 0.68244949);
        test_jaro("schwinghammer", "swinghammer", 0.88811189);
        test_jaro("mickelborough", "mickleburrough", 0.87118437);
        test_jaro("mignon", "guiet", 0.41111111);
        test_jaro("tantaquidgeon", "quidgeon", 0.70512821);
        test_jaro("duyts", "satyrs", 0.58888889);
        test_jaro("cornelise", "esselsteyn", 0.61481481);
        test_jaro("gillington", "gullotine", 0.73068783);
        test_jaro("rogerstillstone", "tillstone", 0.71851852);
        test_jaro("voidy", "vedie", 0.62222222);
        test_jaro("smithdorrien", "dorrien", 0.76587302);
        test_jaro("groethausen", "grothouse", 0.87205387);
        test_jaro("grossnicklaus", "greenehouse", 0.55788656);
        test_jaro("wilmotsitwell", "sitwell", 0.56630037);
        test_jaro("boertgens", "harmense", 0.59351852);
        test_jaro("koetterhagen", "katterhagen", 0.81414141);
        test_jaro("berthelette", "barthelette", 0.80606061);
        test_jaro("schoettler", "brechting", 0.53148148);
        test_jaro("etringer", "thilges", 0.69047619);
        test_jaro("sigurdsson", "lodbrok", 0.46507937);
        test_jaro("deligny", "bidon", 0.56507937);
        test_jaro("winsofer", "hubbarde", 0.33333333);
        test_jaro("straatmaker", "stratenmaker", 0.84747475);
        test_jaro("ouderkerk", "heemstraat", 0.43333333);
        test_jaro("comalander", "cumberlander", 0.69722222);
    }

    fn test_jaro_winkler(a: &str, b: &str, expected_result: f64) {
        assert!(
            (compute_jaro_winkler(a, b) - expected_result).abs() < THRESHOLD,
            "\"{}\", \"{}\" was computed as {}, expected {}",
            a,
            b,
            compute_jaro_winkler(a, b),
            expected_result
        );
    }

    #[test]
    fn jaro_winkler_edge_cases() {
        test_jaro_winkler("s", "s", 1.0);
        test_jaro_winkler("", "", 1.0);
        test_jaro_winkler("string", "", 0.0);
        test_jaro_winkler("", "string", 0.0);
    }

    #[test]
    fn jaro_winkler_test_cases() {
        test_jaro_winkler("phillips", "philips", 0.975);
        test_jaro_winkler("kelly", "kelley", 0.96666667);
        test_jaro_winkler("matthews", "mathews", 0.97083333);
        test_jaro_winkler("wood", "woods", 0.96);
        test_jaro_winkler("hayes", "hays", 0.95333333);
        test_jaro_winkler("russell", "russel", 0.97142857);
        test_jaro_winkler("rogers", "rodgers", 0.96190476);
        test_jaro_winkler("hansen", "hanson", 0.93333333);
        test_jaro_winkler("gray", "grey", 0.86666667);
        test_jaro_winkler("petersen", "peterson", 0.95);
        test_jaro_winkler("myers", "myres", 0.94666667);
        test_jaro_winkler("snyder", "snider", 0.91111111);
        test_jaro_winkler("chamberlain", "chamberlin", 0.98181818);
        test_jaro_winkler("blankenship", "blankinship", 0.92363636);
        test_jaro_winkler("lloyd", "loyd", 0.94);
        test_jaro_winkler("byrd", "bird", 0.85);
        test_jaro_winkler("dickinson", "dickenson", 0.95555556);
        test_jaro_winkler("whitaker", "whittaker", 0.97777778);
        test_jaro_winkler("mullins", "mullens", 0.94285714);
        test_jaro_winkler("frye", "fry", 0.94166667);
        test_jaro_winkler("olsen", "olson", 0.90666667);
        test_jaro_winkler("pennington", "penington", 0.89888889);
        test_jaro_winkler("cunningham", "cuningham", 0.95074074);
        test_jaro_winkler("kirby", "kerby", 0.88);
        test_jaro_winkler("gillespie", "gillispie", 0.93055556);
        test_jaro_winkler("hewitt", "hewett", 0.92222222);
        test_jaro_winkler("lowery", "lowry", 0.96111111);
        test_jaro_winkler("callaway", "calloway", 0.92142857);
        test_jaro_winkler("melton", "milton", 0.9);
        test_jaro_winkler("mckenzie", "mckinzie", 0.90833333);
        test_jaro_winkler("macleod", "mcleod", 0.95714286);
        test_jaro_winkler("davenport", "devenport", 0.89583333);
        test_jaro_winkler("hutchison", "hutchinson", 0.95777778);
        test_jaro_winkler("mackay", "mckay", 0.89);
        test_jaro_winkler("christensen", "christenson", 0.96363636);
        test_jaro_winkler("bannister", "banister", 0.97407407);
        test_jaro_winkler("mcallister", "mcalister", 0.98);
        test_jaro_winkler("cleveland", "cleaveland", 0.92481481);
        test_jaro_winkler("denison", "dennison", 0.90416667);
        test_jaro_winkler("hendrix", "hendricks", 0.9047619);
        test_jaro_winkler("caudill", "candill", 0.92380952);
        test_jaro_winkler("stillwell", "stilwell", 0.95277778);
        test_jaro_winkler("klein", "kline", 0.89333333);
        test_jaro_winkler("mayo", "mays", 0.88333333);
        test_jaro_winkler("albright", "allbright", 0.97037037);
        test_jaro_winkler("macpherson", "mcpherson", 0.97);
        test_jaro_winkler("schaefer", "schafer", 0.94642857);
        test_jaro_winkler("hoskins", "haskins", 0.91428571);
        test_jaro_winkler("moseley", "mosley", 0.92777778);
        test_jaro_winkler("plantagenet", "lancaster", 0.68181818);
        test_jaro_winkler("kendrick", "kindrick", 0.925);
        test_jaro_winkler("horner", "homer", 0.79111111);
        test_jaro_winkler("waddell", "waddle", 0.93809524);
        test_jaro_winkler("whittington", "whitington", 0.94181818);
        test_jaro_winkler("featherstone", "featherston", 0.98333333);
        test_jaro_winkler("broughton", "braughton", 0.94074074);
        test_jaro_winkler("reeder", "reader", 0.85777778);
        test_jaro_winkler("stedman", "steadman", 0.9375);
        test_jaro_winkler("satterfield", "saterfield", 0.97878788);
        test_jaro_winkler("scarborough", "scarbrough", 0.96181818);
        test_jaro_winkler("devine", "divine", 0.84);
        test_jaro_winkler("macfarlane", "mcfarlane", 0.87);
        test_jaro_winkler("debeauchamp", "beauchamp", 0.9023569);
        test_jaro_winkler("albritton", "allbritton", 0.97333333);
        test_jaro_winkler("hutto", "hutts", 0.92);
        test_jaro_winkler("swafford", "swofford", 0.8952381);
        test_jaro_winkler("donovan", "donavan", 0.89444444);
        test_jaro_winkler("plantagenet", "gaunt", 0.62424242);
        test_jaro_winkler("rinehart", "rhinehart", 0.89166667);
        test_jaro_winkler("macarthur", "mcarthur", 0.92916667);
        test_jaro_winkler("hemingway", "hemmingway", 0.97666667);
        test_jaro_winkler("schumacher", "schumaker", 0.93777778);
        test_jaro_winkler("reedy", "rudy", 0.805);
        test_jaro_winkler("dietrich", "deitrich", 0.9625);
        test_jaro_winkler("egan", "eagan", 0.865);
        test_jaro_winkler("steiner", "stiner", 0.91746032);
        test_jaro_winkler("powhatan", "rolfe", 0.44166667);
        test_jaro_winkler("beeler", "buler", 0.765);
        test_jaro_winkler("landen", "austrasia", 0.42592593);
        test_jaro_winkler("pelletier", "peltier", 0.91481481);
        test_jaro_winkler("farnham", "farnum", 0.90952381);
        test_jaro_winkler("arrowood", "arwood", 0.88888889);
        test_jaro_winkler("canterbury", "canterberry", 0.94363636);
        test_jaro_winkler("livengood", "livingood", 0.94814815);
        test_jaro_winkler("scotland", "canmore", 0.51190476);
        test_jaro_winkler("anjou", "danjou", 0.94444444);
        test_jaro_winkler("stanfield", "standfield", 0.93555556);
        test_jaro_winkler("depercy", "percy", 0.83809524);
        test_jaro_winkler("weatherford", "wetherford", 0.97575758);
        test_jaro_winkler("baumgardner", "bumgardner", 0.91272727);
        test_jaro_winkler("rau", "raw", 0.82222222);
        test_jaro_winkler("powhatan", "daughter", 0.47222222);
        test_jaro_winkler("anjou", "jerusalem", 0.54074074);
        test_jaro_winkler("deferrers", "ferrers", 0.83068783);
        test_jaro_winkler("seagraves", "segraves", 0.93703704);
        test_jaro_winkler("jaeger", "jager", 0.90222222);
        test_jaro_winkler("ammerman", "amerman", 0.92857143);
        test_jaro_winkler("muncy", "munsey", 0.87555556);
        test_jaro_winkler("bainbridge", "bambridge", 0.8837037);
        test_jaro_winkler("morehouse", "moorehouse", 0.91407407);
        test_jaro_winkler("witherspoon", "weatherspoon", 0.89272727);
        test_jaro_winkler("breckenridge", "brackenridge", 0.93131313);
        test_jaro_winkler("giddings", "geddings", 0.88214286);
        test_jaro_winkler("hochstetler", "hostetler", 0.95151515);
        test_jaro_winkler("chivers", "chevers", 0.87936508);
        test_jaro_winkler("macaulay", "mcaulay", 0.87678571);
        test_jaro_winkler("fenstermacher", "fenstermaker", 0.9525641);
        test_jaro_winkler("hetherington", "heatherington", 0.97948718);
        test_jaro_winkler("defalaise", "arletta", 0.58862434);
        test_jaro_winkler("breckenridge", "breckinridge", 0.94848485);
        test_jaro_winkler("demeschines", "meschin", 0.83116883);
        test_jaro_winkler("killingsworth", "killingswort", 0.98461538);
        test_jaro_winkler("benningfield", "beningfield", 0.95934343);
        test_jaro_winkler("bretagne", "brittany", 0.8);
        test_jaro_winkler("stonebraker", "stonebreaker", 0.96515152);
        test_jaro_winkler("beresford", "berrisford", 0.86907407);
        test_jaro_winkler("yeo", "geo", 0.77777778);
        test_jaro_winkler("henninger", "heninger", 0.94490741);
        test_jaro_winkler("budgen", "bridgen", 0.86428571);
        test_jaro_winkler("mormaer", "thane", 0.56190476);
        test_jaro_winkler("braithwaite", "brathwaite", 0.93212121);
        test_jaro_winkler("belfield", "bellfield", 0.91574074);
        test_jaro_winkler("fitzpiers", "piers", 0.43703704);
        test_jaro_winkler("decourtenay", "courtenay", 0.82828283);
        test_jaro_winkler("teegarden", "teagarden", 0.90740741);
        test_jaro_winkler("deholand", "holand", 0.91666667);
        test_jaro_winkler("demowbray", "mowbray", 0.92592593);
        test_jaro_winkler("macnaughton", "mcnaughton", 0.94272727);
        test_jaro_winkler("dewarenne", "warenne", 0.78306878);
        test_jaro_winkler("deroet", "roet", 0.80555556);
        test_jaro_winkler("demeschines", "meschines", 0.79124579);
        test_jaro_winkler("normandy", "brittany", 0.66666667);
        test_jaro_winkler("brewington", "bruington", 0.91703704);
        test_jaro_winkler("garnsey", "guernsey", 0.78214286);
        test_jaro_winkler("aetheling", "exile", 0.53333333);
        test_jaro_winkler("barlowe", "almy", 0.5952381);
        test_jaro_winkler("mulholland", "mullholland", 0.95545455);
        test_jaro_winkler("beddingfield", "bedingfield", 0.98055556);
        test_jaro_winkler("couwenhoven", "covenhoven", 0.92484848);
        test_jaro_winkler("macquarrie", "mcquarrie", 0.90333333);
        test_jaro_winkler("haraldsdatter", "haraldsdotter", 0.96923077);
        test_jaro_winkler("seed", "leed", 0.83333333);
        test_jaro_winkler("pitsenbarger", "pittsenbarger", 0.98205128);
        test_jaro_winkler("macmurrough", "macmurchada", 0.85454545);
        test_jaro_winkler("falaise", "arletta", 0.52380952);
        test_jaro_winkler("deberkeley", "berkeley", 0.80833333);
        test_jaro_winkler("guernsey", "gurnsey", 0.89047619);
        test_jaro_winkler("tourault", "archambault", 0.62651515);
        test_jaro_winkler("valliance", "pray", 0.4537037);
        test_jaro_winkler("enfinger", "infinger", 0.86904762);
        test_jaro_winkler("fischbach", "fishback", 0.85231481);
        test_jaro_winkler("pelham", "pellum", 0.84444444);
        test_jaro_winkler("dechatellerault", "chatellerault", 0.87863248);
        test_jaro_winkler("trico", "rapalje", 0.44761905);
        test_jaro_winkler("hatherly", "hanford", 0.60119048);
        test_jaro_winkler("aquitaine", "eleanor", 0.5026455);
        test_jaro_winkler("devere", "vere", 0.72222222);
        test_jaro_winkler("coppedge", "coppage", 0.88809524);
        test_jaro_winkler("rockefeller", "rockafellow", 0.86590909);
        test_jaro_winkler("rubenstein", "rubinstein", 0.90148148);
        test_jaro_winkler("mcmurrough", "macmurrough", 0.97272727);
        test_jaro_winkler("roet", "swynford", 0.0);
        test_jaro_winkler("bodenheimer", "bodenhamer", 0.92141414);
        test_jaro_winkler("dehauteville", "hauteville", 0.81111111);
        test_jaro_winkler("jubinville", "jubenville", 0.92740741);
        test_jaro_winkler("decantilupe", "cantilupe", 0.93939394);
        test_jaro_winkler("kitteringham", "ketteringham", 0.92272727);
        test_jaro_winkler("maugis", "miville", 0.53968254);
        test_jaro_winkler("cornford", "comford", 0.85079365);
        test_jaro_winkler("alsobrook", "alsabrook", 0.91898148);
        test_jaro_winkler("villines", "valines", 0.83214286);
        test_jaro_winkler("fitzjohn", "fitzgeoffrey", 0.68055556);
        test_jaro_winkler("decrepon", "hardaknutsson", 0.43589744);
        test_jaro_winkler("deswynnerton", "swinnerton", 0.80925926);
        test_jaro_winkler("terriot", "terriau", 0.88571429);
        test_jaro_winkler("detaillefer", "angouleme", 0.6026936);
        test_jaro_winkler("fleitel", "flatel", 0.82603175);
        test_jaro_winkler("temperley", "timperley", 0.82083333);
        test_jaro_winkler("dentzer", "henckel", 0.61904762);
        test_jaro_winkler("provencher", "provancher", 0.91555556);
        test_jaro_winkler("deroet", "swynford", 0.43055556);
        test_jaro_winkler("arganbright", "argenbright", 0.95757576);
        test_jaro_winkler("vencill", "vincell", 0.82857143);
        test_jaro_winkler("daingerfield", "dangerfield", 0.90505051);
        test_jaro_winkler("selbee", "blott", 0.45555556);
        test_jaro_winkler("berkeley", "martiau", 0.42261905);
        test_jaro_winkler("cavendishbentinck", "bentinck", 0.50637255);
        test_jaro_winkler("mcmurrough", "leinster", 0.40833333);
        test_jaro_winkler("debraose", "briose", 0.81944444);
        test_jaro_winkler("turberville", "tuberville", 0.94909091);
        test_jaro_winkler("reinschmidt", "cuntze", 0.33838384);
        test_jaro_winkler("kember", "thember", 0.84920635);
        test_jaro_winkler("vancouwenhoven", "couwenhoven", 0.77705628);
        test_jaro_winkler("fenstermaker", "fenstemaker", 0.9469697);
        test_jaro_winkler("oberbroeckling", "oberbrockling", 0.98571429);
        test_jaro_winkler("hems", "herns", 0.82666667);
        test_jaro_winkler("fitzmaurice", "fitzmorris", 0.86727273);
        test_jaro_winkler("mannon", "manon", 0.91444444);
        test_jaro_winkler("peddicord", "petticord", 0.88148148);
        test_jaro_winkler("flowerdew", "yeardly", 0.58730159);
        test_jaro_winkler("shufflebotham", "shufflebottom", 0.93846154);
        test_jaro_winkler("facteau", "facto", 0.87428571);
        test_jaro_winkler("aldin", "nalle", 0.6);
        test_jaro_winkler("helphenstine", "helphinstine", 0.93030303);
        test_jaro_winkler("debesford", "besford", 0.87830688);
        test_jaro_winkler("fitzalan", "goushill", 0.41666667);
        test_jaro_winkler("riseborough", "roseborough", 0.85545455);
        test_jaro_winkler("gruffydd", "rhys", 0.58333333);
        test_jaro_winkler("hornberger", "homberger", 0.81703704);
        test_jaro_winkler("tattershall", "tatarsole", 0.796633);
        test_jaro_winkler("taillefer", "angouleme", 0.62962963);
        test_jaro_winkler("reierson", "rierson", 0.91964286);
        test_jaro_winkler("wrinkle", "rinkle", 0.95238095);
        test_jaro_winkler("goldhatch", "tritton", 0.41798942);
        test_jaro_winkler("sumarlidasson", "somerledsson", 0.83269231);
        test_jaro_winkler("amberg", "glatfelder", 0.51111111);
        test_jaro_winkler("raistrick", "rastrick", 0.9037037);
        test_jaro_winkler("bajolet", "bayol", 0.83238095);
        test_jaro_winkler("billung", "sachsen", 0.42857143);
        test_jaro_winkler("delazouche", "zouche", 0.7);
        test_jaro_winkler("springham", "springhorn", 0.89555556);
        test_jaro_winkler("deserres", "dessert", 0.84583333);
        test_jaro_winkler("gendre", "bourgery", 0.52777778);
        test_jaro_winkler("braconie", "brackhonge", 0.915);
        test_jaro_winkler("pleydellbouverie", "bouverie", 0.45833333);
        test_jaro_winkler("fricks", "frix", 0.825);
        test_jaro_winkler("plamondon", "plomondon", 0.87407407);
        test_jaro_winkler("aubigny", "albini", 0.77142857);
        test_jaro_winkler("freemanmitford", "mitford", 0.43650794);
        test_jaro_winkler("fightmaster", "fight", 0.89090909);
        test_jaro_winkler("wye", "why", 0.55555556);
        test_jaro_winkler("birtwistle", "bertwistle", 0.87333333);
        test_jaro_winkler("lautzenheiser", "lautzenhiser", 0.96794872);
        test_jaro_winkler("puntenney", "puntney", 0.92698413);
        test_jaro_winkler("demaranville", "demoranville", 0.93989899);
        test_jaro_winkler("tellico", "clan", 0.46428571);
        test_jaro_winkler("doublehead", "cornblossom", 0.52424242);
        test_jaro_winkler("landvatter", "merckle", 0.41428571);
        test_jaro_winkler("smy", "sury", 0.75);
        test_jaro_winkler("macvane", "mcvane", 0.90714286);
        test_jaro_winkler("grimoult", "sedilot", 0.60714286);
        test_jaro_winkler("walgrave", "waldegrave", 0.895);
        test_jaro_winkler("moytoy", "oconostota", 0.48888889);
        test_jaro_winkler("steenbergen", "stenbergen", 0.90878788);
        test_jaro_winkler("wolfensberger", "wolfersberger", 0.91923077);
        test_jaro_winkler("lydecker", "leydecker", 0.85416667);
        test_jaro_winkler("scheele", "schule", 0.84777778);
        test_jaro_winkler("mountbatten", "battenberg", 0.58484848);
        test_jaro_winkler("detalvas", "talvace", 0.7797619);
        test_jaro_winkler("zwiefelhofer", "zwifelhofer", 0.91691919);
        test_jaro_winkler("hawbaker", "hawbecker", 0.90198413);
        test_jaro_winkler("degrandison", "grandson", 0.82575758);
        test_jaro_winkler("ardouin", "badeau", 0.64285714);
        test_jaro_winkler("loughmiller", "laughmiller", 0.94545455);
        test_jaro_winkler("verdun", "ardenne", 0.66269841);
        test_jaro_winkler("jorisse", "joire", 0.87047619);
        test_jaro_winkler("wentworthfitzwilliam", "fitzwilliam", 0.78939394);
        test_jaro_winkler("cornforth", "comforth", 0.86931217);
        test_jaro_winkler("vanschouwen", "cornelissen", 0.56969697);
        test_jaro_winkler("fuselier", "fusillier", 0.88564815);
        test_jaro_winkler("frederickse", "lubbertsen", 0.42121212);
        test_jaro_winkler("haraldsson", "forkbeard", 0.54444444);
        test_jaro_winkler("fausnaugh", "fosnaugh", 0.81011905);
        test_jaro_winkler("aymard", "emard", 0.73888889);
        test_jaro_winkler("ingaldesthorpe", "ingoldsthrop", 0.90934343);
        test_jaro_winkler("blennerhassett", "blaverhasset", 0.85269841);
        test_jaro_winkler("ednywain", "edwyn", 0.84666667);
        test_jaro_winkler("aubigny", "daubigny", 0.95833333);
        test_jaro_winkler("hinderliter", "henderliter", 0.91545455);
        test_jaro_winkler("deroucy", "rouci", 0.79047619);
        test_jaro_winkler("dechow", "dago", 0.61111111);
        test_jaro_winkler("kenchington", "kinchington", 0.88545455);
        test_jaro_winkler("levere", "lavere", 0.78);
        test_jaro_winkler("denivelles", "itta", 0.45);
        test_jaro_winkler("delusignan", "lusignam", 0.85833333);
        test_jaro_winkler("decressingham", "cressingham", 0.91841492);
        test_jaro_winkler("austerfield", "osterfield", 0.90606061);
        test_jaro_winkler("esten", "eustance", 0.79);
        test_jaro_winkler("maves", "mebs", 0.63333333);
        test_jaro_winkler("wieneke", "wineke", 0.87301587);
        test_jaro_winkler("deliercourt", "juillet", 0.56168831);
        test_jaro_winkler("auxerre", "argengau", 0.49007937);
        test_jaro_winkler("beedell", "budell", 0.80428571);
        test_jaro_winkler("muscoe", "hucklescott", 0.62929293);
        test_jaro_winkler("feese", "fuse", 0.67222222);
        test_jaro_winkler("laughinghouse", "lathinghouse", 0.88826729);
        test_jaro_winkler("decrocketagne", "crocketagne", 0.7972028);
        test_jaro_winkler("petitpas", "bugaret", 0.3452381);
        test_jaro_winkler("leatherbarrow", "letherbarrow", 0.91282051);
        test_jaro_winkler("goughcalthorpe", "calthorpe", 0.76984127);
        test_jaro_winkler("stooksbury", "stookesberry", 0.93);
        test_jaro_winkler("leichleiter", "lechleiter", 0.92242424);
        test_jaro_winkler("devalletort", "valletort", 0.86531987);
        test_jaro_winkler("duranceau", "duranso", 0.85396825);
        test_jaro_winkler("ordepowlett", "powlett", 0.78354978);
        test_jaro_winkler("freudenberg", "frendenberg", 0.93424242);
        test_jaro_winkler("featherstonhaugh", "featherstonehaugh", 0.98823529);
        test_jaro_winkler("hepburnstuartforbestrefusis", "trefusis", 0.56856261);
        test_jaro_winkler("minkrevicius", "minkavitch", 0.85666667);
        test_jaro_winkler("stuedemann", "studeman", 0.92416667);
        test_jaro_winkler("frande", "andersdotter", 0.61666667);
        test_jaro_winkler("alwyn", "joan", 0.48333333);
        test_jaro_winkler("abendschon", "obenchain", 0.75555556);
        test_jaro_winkler("landvatter", "varonica", 0.55);
        test_jaro_winkler("dewindsor", "fitzotho", 0.49074074);
        test_jaro_winkler("renkenberger", "rinkenberger", 0.84090909);
        test_jaro_winkler("volkertsen", "noorman", 0.57619048);
        test_jaro_winkler("casaubon", "casobon", 0.86944444);
        test_jaro_winkler("decherleton", "cherlton", 0.86742424);
        test_jaro_winkler("karraker", "kanaker", 0.80634921);
        test_jaro_winkler("sinquefield", "sinkfield", 0.88127104);
        test_jaro_winkler("lycon", "laican", 0.73);
        test_jaro_winkler("cyphert", "seyphert", 0.75793651);
        test_jaro_winkler("humphrie", "umfery", 0.625);
        test_jaro_winkler("loignon", "longnon", 0.83492063);
        test_jaro_winkler("cletheroe", "clitheroe", 0.84074074);
        test_jaro_winkler("oesterle", "osterle", 0.83392857);
        test_jaro_winkler("evemy", "evering", 0.67619048);
        test_jaro_winkler("niquette", "nequette", 0.83928571);
        test_jaro_winkler("lemeunier", "daubigeon", 0.46296296);
        test_jaro_winkler("hartsvelder", "hartzfelder", 0.92727273);
        test_jaro_winkler("beiersdorf", "biersdorf", 0.93666667);
        test_jaro_winkler("destroismaisons", "destrosmaisons", 0.95809524);
        test_jaro_winkler("warminger", "wanninger", 0.84338624);
        test_jaro_winkler("demoleyns", "molines", 0.78571429);
        test_jaro_winkler("chetwyndstapylton", "stapylton", 0.47657952);
        test_jaro_winkler("woodville", "wydvile", 0.85714286);
        test_jaro_winkler("fivekiller", "ghigau", 0.42222222);
        test_jaro_winkler("rochet", "garrigues", 0.51851852);
        test_jaro_winkler("leyendecker", "lyendecker", 0.85272727);
        test_jaro_winkler("auringer", "oranger", 0.81349206);
        test_jaro_winkler("twelftree", "twelvetree", 0.91277778);
        test_jaro_winkler("roeloffse", "kierstede", 0.5462963);
        test_jaro_winkler("stalsworth", "stolsworth", 0.88740741);
        test_jaro_winkler("jarry", "rapin", 0.46666667);
        test_jaro_winkler("lawter", "lantersee", 0.83703704);
        test_jaro_winkler("andrewartha", "andrawartha", 0.90363636);
        test_jaro_winkler("requa", "regna", 0.78666667);
        test_jaro_winkler("devaloines", "volognes", 0.72777778);
        test_jaro_winkler("featherstonhaugh", "fetherstonbaugh", 0.91079365);
        test_jaro_winkler("sacherell", "searth", 0.66296296);
        test_jaro_winkler("peerenboom", "perenboom", 0.9437037);
        test_jaro_winkler("coeffes", "forestier", 0.62328042);
        test_jaro_winkler("dewease", "duese", 0.73642857);
        test_jaro_winkler("schackmann", "shackman", 0.9025);
        test_jaro_winkler("breidenbaugh", "bridenbaugh", 0.95353535);
        test_jaro_winkler("schollenberger", "shollenberger", 0.97857143);
        test_jaro_winkler("neumeister", "newmaster", 0.8226455);
        test_jaro_winkler("bettesworth", "betsworth", 0.90572391);
        test_jaro_winkler("demedici", "medici", 0.86111111);
        test_jaro_winkler("volkertsen", "holgersen", 0.82592593);
        test_jaro_winkler("delusignan", "lusigan", 0.85238095);
        test_jaro_winkler("elchert", "elkhart", 0.84761905);
        test_jaro_winkler("detaillebois", "taillebois", 0.87777778);
        test_jaro_winkler("fagelson", "feygelson", 0.85297619);
        test_jaro_winkler("burdeshaw", "burtashaw", 0.86296296);
        test_jaro_winkler("vanderburchgraeff", "burchgraeff", 0.76114082);
        test_jaro_winkler("tiptoft", "tybotot", 0.63095238);
        test_jaro_winkler("crepon", "forkbeard", 0.5);
        test_jaro_winkler("rugglesbrise", "brise", 0.50555556);
        test_jaro_winkler("grawbarger", "grauberger", 0.8775);
        test_jaro_winkler("brassier", "decheilus", 0.32407407);
        test_jaro_winkler("coningsby", "connyngesby", 0.85210438);
        test_jaro_winkler("barneycastle", "barnacastle", 0.92848485);
        test_jaro_winkler("degreystoke", "greystock", 0.83038721);
        test_jaro_winkler("streitenberger", "strattenbarger", 0.83636364);
        test_jaro_winkler("manitouabeouich", "manithabehich", 0.89771562);
        test_jaro_winkler("haddleton", "addleton", 0.96296296);
        test_jaro_winkler("trethurffe", "tretford", 0.83666667);
        test_jaro_winkler("jaeckler", "margaretha", 0.44722222);
        test_jaro_winkler("braeutigam", "braitigam", 0.89824074);
        test_jaro_winkler("bacorn", "bakehorn", 0.85555556);
        test_jaro_winkler("burkenbine", "birkinbine", 0.8425);
        test_jaro_winkler("paulo", "campot", 0.57777778);
        test_jaro_winkler("essenmacher", "eunmaker", 0.6540404);
        test_jaro_winkler("decrocketagne", "crocketagni", 0.79277389);
        test_jaro_winkler("unruhe", "kornmann", 0.36111111);
        test_jaro_winkler("baesemann", "baseman", 0.9026455);
        test_jaro_winkler("worchester", "worsestor", 0.84481481);
        test_jaro_winkler("reidelberger", "rudelberger", 0.80272727);
        test_jaro_winkler("bradtmueller", "bradmiller", 0.90777778);
        test_jaro_winkler("schreckengast", "shreckengast", 0.92692308);
        test_jaro_winkler("eisentrout", "isentrout", 0.92962963);
        test_jaro_winkler("fivekiller", "kingfisher", 0.67777778);
        test_jaro_winkler("jinright", "ginwright", 0.88425926);
        test_jaro_winkler("reichenberger", "richenberger", 0.85192308);
        test_jaro_winkler("langehennig", "laughennig", 0.89521886);
        test_jaro_winkler("muttlebury", "mattleberry", 0.85818182);
        test_jaro_winkler("cullumbine", "columbine", 0.86916667);
        test_jaro_winkler("badlesmere", "northampton", 0.46060606);
        test_jaro_winkler("oxier", "ockshire", 0.65833333);
        test_jaro_winkler("anway", "amvay", 0.76);
        test_jaro_winkler("wagenseller", "wagonseller", 0.91090909);
        test_jaro_winkler("siebenthaler", "sevendollar", 0.69227994);
        test_jaro_winkler("vannoorstrant", "juriaens", 0.51923077);
        test_jaro_winkler("grundvig", "gumdeig", 0.80178571);
        test_jaro_winkler("freudenberger", "frendenberger", 0.94465812);
        test_jaro_winkler("schinbeckler", "shinbeckler", 0.89318182);
        test_jaro_winkler("stautzenberger", "stantzenberger", 0.93076923);
        test_jaro_winkler("molandersmolandes", "molandes", 0.89411765);
        test_jaro_winkler("altstaetter", "allstetter", 0.86558923);
        test_jaro_winkler("moredock", "nedock", 0.75277778);
        test_jaro_winkler("bouslaugh", "baughlough", 0.68306878);
        test_jaro_winkler("schoenbachler", "schoenbaechler", 0.95494505);
        test_jaro_winkler("tetterton", "letterton", 0.84259259);
        test_jaro_winkler("korsing", "curring", 0.71428571);
        test_jaro_winkler("breckheimer", "brickheimer", 0.87151515);
        test_jaro_winkler("doors", "streypress", 0.36666667);
        test_jaro_winkler("flattum", "flattenn", 0.86785714);
        test_jaro_winkler("demontmorency", "montmorency", 0.94871795);
        test_jaro_winkler("andrieszen", "larens", 0.71111111);
        test_jaro_winkler("hughesdaeth", "daeth", 0.36060606);
        test_jaro_winkler("cullumbine", "colleunbine", 0.78318182);
        test_jaro_winkler("twisletonwykehamfiennes", "fiennes", 0.51055901);
        test_jaro_winkler("scherber", "sharver", 0.74178571);
        test_jaro_winkler("coerten", "harmens", 0.50793651);
        test_jaro_winkler("pitres", "fitzroger", 0.7037037);
        test_jaro_winkler("degloucester", "fitzroger", 0.50925926);
        test_jaro_winkler("sevestre", "delessart", 0.66018519);
        test_jaro_winkler("larzelere", "larzalere", 0.90555556);
        test_jaro_winkler("bargsley", "bayeley", 0.77047619);
        test_jaro_winkler("flockirth", "flogerth", 0.86388889);
        test_jaro_winkler("euteneier", "eutencies", 0.88253968);
        test_jaro_winkler("smelker", "schmelcher", 0.83714286);
        test_jaro_winkler("auchincloss", "anchencloss", 0.85757576);
        test_jaro_winkler("during", "shaumloffel", 0.41919192);
        test_jaro_winkler("arizmendi", "arismendez", 0.87814815);
        test_jaro_winkler("otterlifter", "wawli", 0.52727273);
        test_jaro_winkler("wackerle", "weckerla", 0.79642857);
        test_jaro_winkler("manselpleydell", "pleydell", 0.73214286);
        test_jaro_winkler("schwabenlender", "schwabenlander", 0.95604396);
        test_jaro_winkler("hemmesch", "hemish", 0.87361111);
        test_jaro_winkler("austerfield", "hosterfield", 0.87878788);
        test_jaro_winkler("deetherstone", "detherston", 0.92888889);
        test_jaro_winkler("rauhuff", "rowhuff", 0.76857143);
        test_jaro_winkler("yorek", "jarek", 0.73333333);
        test_jaro_winkler("hennikermajor", "henniker", 0.92307692);
        test_jaro_winkler("depitres", "fitzroger", 0.64814815);
        test_jaro_winkler("riedmueller", "reidmiller", 0.84878788);
        test_jaro_winkler("culm", "culin", 0.84833333);
        test_jaro_winkler("jonah", "joney", 0.81333333);
        test_jaro_winkler("heckingbottom", "hickingbottom", 0.92884615);
        test_jaro_winkler("schnorenberg", "schnorrenberg", 0.95128205);
        test_jaro_winkler("hotzenbella", "hotzenpeller", 0.91363636);
        test_jaro_winkler("theiring", "tyring", 0.7775);
        test_jaro_winkler("dieffenwierth", "diffenweirth", 0.93504274);
        test_jaro_winkler("haakonsdatter", "haakonson", 0.86324786);
        test_jaro_winkler("martinvegue", "vegue", 0.43030303);
        test_jaro_winkler("threapleton", "thrippleton", 0.88922559);
        test_jaro_winkler("kirberger", "moelich", 0.33597884);
        test_jaro_winkler("hinneschied", "henneschid", 0.88212121);
        test_jaro_winkler("palmermorewood", "morewood", 0.81547619);
        test_jaro_winkler("guicciardini", "guciardini", 0.84888889);
        test_jaro_winkler("fregia", "fruger", 0.69444444);
        test_jaro_winkler("braunberger", "bramberg", 0.88598485);
        test_jaro_winkler("katterheinrich", "katterhenrich", 0.97032967);
        test_jaro_winkler("bolibaugh", "bolebough", 0.8962963);
        test_jaro_winkler("bechtelheimer", "becktelheimer", 0.92521368);
        test_jaro_winkler("detibetot", "tybotot", 0.75661376);
        test_jaro_winkler("encke", "ink", 0.68888889);
        test_jaro_winkler("kettleborough", "kettleboro", 0.95384615);
        test_jaro_winkler("whittingstall", "whettingstall", 0.93675214);
        test_jaro_winkler("baggenstoss", "backenstoss", 0.9030303);
        test_jaro_winkler("ardion", "rabouin", 0.71587302);
        test_jaro_winkler("wittelsbach", "palatine", 0.53787879);
        test_jaro_winkler("dechaux", "chapelain", 0.50529101);
        test_jaro_winkler("vancortenbosch", "cortenbosch", 0.83766234);
        test_jaro_winkler("swyersexey", "sexey", 0.65);
        test_jaro_winkler("deherville", "sohier", 0.52222222);
        test_jaro_winkler("kaesemeyer", "kasemeyer", 0.88444444);
        test_jaro_winkler("righthouse", "wrighthouse", 0.96969697);
        test_jaro_winkler("coeffes", "fourestier", 0.6047619);
        test_jaro_winkler("kemeystynte", "tynte", 0.51313131);
        test_jaro_winkler("goenner", "toennes", 0.80952381);
        test_jaro_winkler("schattschneider", "schatschneider", 0.98666667);
        test_jaro_winkler("masonheimer", "masenheimer", 0.88757576);
        test_jaro_winkler("knutti", "margaritha", 0.34444444);
        test_jaro_winkler("stoughtenger", "stoutenger", 0.92666667);
        test_jaro_winkler("boeckhout", "elswaerts", 0.48148148);
        test_jaro_winkler("nohrenhold", "nornhold", 0.91333333);
        test_jaro_winkler("transdotter", "thrandsdotter", 0.92657343);
        test_jaro_winkler("vansintern", "sintern", 0.75714286);
        test_jaro_winkler("knatchbullhugessen", "hugessen", 0.40740741);
        test_jaro_winkler("deshayes", "dehais", 0.80222222);
        test_jaro_winkler("bornbach", "bomback", 0.82380952);
        test_jaro_winkler("aupry", "auprybertrand", 0.87692308);
        test_jaro_winkler("loewenhagen", "lowenhagen", 0.89575758);
        test_jaro_winkler("thorsteinsdottir", "thurstenson", 0.68244949);
        test_jaro_winkler("schwinghammer", "swinghammer", 0.8993007);
        test_jaro_winkler("mickelborough", "mickleburrough", 0.92271062);
        test_jaro_winkler("friedenberg", "freedenberg", 0.89818182);
        test_jaro_winkler("houmes", "hornnes", 0.7968254);
        test_jaro_winkler("mignon", "guiet", 0.41111111);
        test_jaro_winkler("mickelborough", "michelborough", 0.96410256);
        test_jaro_winkler("tantaquidgeon", "quidgeon", 0.70512821);
        test_jaro_winkler("duyts", "satyrs", 0.58888889);
        test_jaro_winkler("highcock", "hycock", 0.8375);
        test_jaro_winkler("cornelise", "esselsteyn", 0.61481481);
        test_jaro_winkler("schoenthaler", "schonthaler", 0.92878788);
        test_jaro_winkler("gillington", "gullotine", 0.75761905);
        test_jaro_winkler("rogerstillstone", "tillstone", 0.71851852);
        test_jaro_winkler("voidy", "vedie", 0.62222222);
        test_jaro_winkler("smithdorrien", "dorrien", 0.76587302);
        test_jaro_winkler("groethausen", "grothouse", 0.91043771);
        test_jaro_winkler("schmeeckle", "schmuckle", 0.88777778);
        test_jaro_winkler("grossnicklaus", "greenehouse", 0.55788656);
        test_jaro_winkler("wilmotsitwell", "sitwell", 0.56630037);
        test_jaro_winkler("boertgens", "harmense", 0.59351852);
        test_jaro_winkler("koetterhagen", "katterhagen", 0.83272727);
        test_jaro_winkler("berthelette", "barthelette", 0.82545455);
        test_jaro_winkler("schoettler", "brechting", 0.53148148);
        test_jaro_winkler("wescovich", "viscovich", 0.85185185);
        test_jaro_winkler("etringer", "thilges", 0.69047619);
        test_jaro_winkler("sigurdsson", "lodbrok", 0.46507937);
        test_jaro_winkler("deligny", "bidon", 0.56507937);
        test_jaro_winkler("winsofer", "hubbarde", 0.33333333);
        test_jaro_winkler("straatmaker", "stratenmaker", 0.90848485);
        test_jaro_winkler("warrenbuer", "warambour", 0.82888889);
        test_jaro_winkler("ouderkerk", "heemstraat", 0.43333333);
        test_jaro_winkler("comalander", "cumberlander", 0.69722222);
        test_jaro_winkler("holtzendorff", "holsendorf", 0.91833333);
        test_jaro_winkler("beirdneau", "birdno", 0.81666667);
    }
}
