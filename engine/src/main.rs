use std::io::{stdin, stdout, BufReader};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(String::as_str) == Some("bench") {
        let depth = args.get(2).and_then(|d| d.parse().ok()).unwrap_or(8);
        let (nodes, nps) = vektr::bench::run(depth);
        println!("{nodes} nodes {nps} nps");
    } else {
        vektr::uci::run(BufReader::new(stdin()), stdout());
    }
}
