use vektr::{
    tt::{Bound, Entry, TT},
    MATE,
};

#[test]
fn stores_and_probes() {
    let mut t = TT::new(1);
    t.store(42, 0, Entry { mv: None, score: 17, depth: 5, bound: Bound::Exact });
    let e = t.probe(42, 0).unwrap();
    assert_eq!((e.score, e.depth, e.bound), (17, 5, Bound::Exact));
    assert!(t.probe(43, 0).is_none());
}

#[test]
fn mate_scores_are_rebased_on_ply() {
    let mut t = TT::new(1);
    // at ply 4 we found "mate 3 plies from here": score MATE - 7
    t.store(7, 4, Entry { mv: None, score: MATE - 7, depth: 9, bound: Bound::Exact });
    assert_eq!(t.probe(7, 10).unwrap().score, MATE - 13);
    t.store(8, 4, Entry { mv: None, score: -(MATE - 7), depth: 9, bound: Bound::Exact });
    assert_eq!(t.probe(8, 10).unwrap().score, -(MATE - 13));
}

#[test]
fn size_respects_the_budget() {
    let t = TT::new(16);
    assert!(t.size_bytes() <= 16 * 1024 * 1024 && t.size_bytes() > 4 * 1024 * 1024);
    assert!(TT::new(0).size_bytes() > 0);
}
