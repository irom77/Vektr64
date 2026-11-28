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
    } else {
        vektr::uci::run(BufReader::new(stdin()), stdout());
    }
}
