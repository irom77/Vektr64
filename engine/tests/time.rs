use vektr::time::{allocate, fixed, Clock};

fn c(r: i64, i: i64, m: Option<u32>) -> Clock {
    Clock { remaining_ms: r, inc_ms: i, moves_to_go: m }
}

#[test]
fn blitz_3_plus_2_opening() {
    assert_eq!(allocate(&c(180_000, 2_000, None), 100), (3_498, 13_992));
}

#[test]
fn low_on_time_with_increment() {
    assert_eq!(allocate(&c(1_000, 2_000, None), 100), (150, 450));
}

#[test]
fn moves_to_go_is_used() {
    assert_eq!(allocate(&c(60_000, 0, Some(10)), 0), (3_000, 12_000));
}

#[test]
fn zero_and_negative_clocks_give_one_ms() {
    assert_eq!(allocate(&c(0, 0, None), 100), (1, 1));
    assert_eq!(allocate(&c(-500, 2_000, None), 100), (1, 1));
    assert_eq!(allocate(&c(50, 0, None), 100), (1, 1));
}

#[test]
fn hard_never_exceeds_usable_time() {
    for r in [10, 100, 1_000, 10_000, 300_000] {
        for i in [0, 2_000, 3_000] {
            let (s, h) = allocate(&c(r, i, None), 100);
            assert!(s <= h && (h as i64) <= (r - 100).max(1));
        }
    }
}

#[test]
fn movetime() {
    assert_eq!(fixed(1_000, 100), (900, 900));
    assert_eq!(fixed(50, 100), (1, 1));
}
