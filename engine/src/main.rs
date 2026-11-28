use std::io::{stdin, stdout, BufReader};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(String::as_str) == Some("bench") {
        let depth = args.get(2).and_then(|d| d.parse().ok()).unwrap_or(8);
        let (nodes, nps) = vektr::bench::run(depth);
        println!("{nodes} nodes {nps} nps");
    } else if args.get(1).map(String::as_str) == Some("netcheck") {
        let net = vektr::nnue::Network::embedded();
        match vektr::nnue::netcheck(&net, vektr::nnue::DEFAULT_CHECK) {
            Ok(n) => println!("netcheck ok {n} positions"),
            Err(lines) => {
                for line in lines {
                    println!("{line}");
                }
                std::process::exit(1);
            }
        }
    } else if args.get(1).map(String::as_str) == Some("datagen") {
        std::process::exit(datagen(&args[2..]));
    } else {
        vektr::uci::run(BufReader::new(stdin()), stdout());
    }
}

const USAGE: &str = "usage: vektr datagen --out FILE --positions N [--threads T=1] [--nodes K=5000] \
[--seed S=1] [--net FILE] [--starts FILE]";

struct DatagenArgs {
    out: String,
    positions: u64,
    threads: usize,
    nodes: u64,
    seed: u64,
    net: Option<String>,
    starts: Option<String>,
}

fn parse_datagen(args: &[String]) -> Option<DatagenArgs> {
    let (mut out, mut positions, mut net, mut starts) = (None, None, None, None);
    let (mut threads, mut nodes, mut seed) = (1usize, 5000u64, 1u64);
    let mut it = args.iter();
    while let Some(flag) = it.next() {
        let value = it.next()?;
        match flag.as_str() {
            "--out" => out = Some(value.clone()),
            "--positions" => positions = Some(value.parse().ok()?),
            "--threads" => threads = value.parse().ok().filter(|&t| t > 0)?,
            "--nodes" => nodes = value.parse().ok().filter(|&n| n > 0)?,
            "--seed" => seed = value.parse().ok()?,
            "--net" => net = Some(value.clone()),
            "--starts" => starts = Some(value.clone()),
            _ => return None,
        }
    }
    Some(DatagenArgs { out: out?, positions: positions?, threads, nodes, seed, net, starts })
}

/// Runs `vektr datagen`; returns the exit status.
fn datagen(args: &[String]) -> i32 {
    let Some(a) = parse_datagen(args) else {
        eprintln!("{USAGE}");
        return 2;
    };
    match datagen_to_file(&a) {
        Ok(summary) => {
            println!(
                "datagen: {} games, {} positions, +{} ={} -{}",
                summary.games, summary.positions, summary.white_wins, summary.draws, summary.black_wins
            );
            0
        }
        Err(message) => {
            eprintln!("datagen: {message}");
            1
        }
    }
}

fn datagen_to_file(a: &DatagenArgs) -> Result<vektr::datagen::Summary, String> {
    use std::io::Write;

    let net = match &a.net {
        Some(path) => std::sync::Arc::new(vektr::nnue::Network::load(path)?),
        None => vektr::nnue::Network::embedded(),
    };
    let starts = match &a.starts {
        Some(path) => {
            let text = std::fs::read_to_string(path).map_err(|e| format!("cannot read {path}: {e}"))?;
            vektr::datagen::load_starts(&text)
        }
        None => Vec::new(),
    };
    let start_share = if starts.is_empty() { 0 } else { 10 };
    let settings = vektr::datagen::Settings { nodes: a.nodes, starts, start_share };

    let part = format!("{}.part", a.out);
    let file = std::fs::File::create(&part).map_err(|e| format!("cannot write {part}: {e}"))?;
    let mut writer = std::io::BufWriter::new(file);
    let result = vektr::datagen::run(&settings, net, a.positions, a.threads, a.seed, &mut writer)
        .and_then(|summary| writer.flush().map(|_| summary))
        .and_then(|summary| std::fs::rename(&part, &a.out).map(|_| summary));
    result.map_err(|e| {
        let _ = std::fs::remove_file(&part);
        format!("cannot write {}: {e}", a.out)
    })
}
