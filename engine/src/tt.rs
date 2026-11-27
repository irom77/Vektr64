//! Transposition table: what the search already learned about a position.

use cozy_chess::{Move, Piece, Square};

use crate::{MATE, MAX_PLY};

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Bound {
    Exact,
    /// The true score is at least this (the search failed high).
    Lower,
    /// The true score is at most this (the search failed low).
    Upper,
}

#[derive(Clone, Copy)]
pub struct Entry {
    pub mv: Option<Move>,
    pub score: i32,
    pub depth: u8,
    pub bound: Bound,
}

#[derive(Clone, Copy)]
struct Slot {
    key: u64,
    score: i16,
    mv: u16,
    depth: u8,
    /// 0 = empty, otherwise `Bound` + 1.
    bound: u8,
}

const EMPTY: Slot = Slot { key: 0, score: 0, mv: 0, depth: 0, bound: 0 };
const MATE_BOUND: i32 = MATE - MAX_PLY as i32;

pub struct TT {
    slots: Vec<Slot>,
    mask: usize,
}

fn pack_move(mv: Option<Move>) -> u16 {
    match mv {
        None => 0,
        Some(m) => {
            let promo = match m.promotion {
                None => 0,
                Some(p) => p as u16 + 1,
            };
            // A real move never has from == to, so 0 is free to mean "none".
            (m.from as u16) | (m.to as u16) << 6 | promo << 12
        }
    }
}

fn unpack_move(bits: u16) -> Option<Move> {
    if bits == 0 {
        return None;
    }
    let promo = (bits >> 12) as usize;
    Some(Move {
        from: Square::index((bits & 63) as usize),
        to: Square::index((bits >> 6 & 63) as usize),
        promotion: if promo == 0 { None } else { Some(Piece::index(promo - 1)) },
    })
}

impl TT {
    /// A table using at most `mb` MiB, and never fewer than 1024 entries.
    pub fn new(mb: usize) -> TT {
        let budget = mb * 1024 * 1024 / std::mem::size_of::<Slot>();
        let mut count = 1024;
        while count * 2 <= budget {
            count *= 2;
        }
        TT { slots: vec![EMPTY; count], mask: count - 1 }
    }

    pub fn clear(&mut self) {
        self.slots.fill(EMPTY);
    }

    pub fn size_bytes(&self) -> usize {
        self.slots.len() * std::mem::size_of::<Slot>()
    }

    /// Looks up `hash`. Mate scores come back relative to `ply`.
    pub fn probe(&self, hash: u64, ply: usize) -> Option<Entry> {
        let slot = &self.slots[hash as usize & self.mask];
        if slot.bound == 0 || slot.key != hash {
            return None;
        }
        let mut score = slot.score as i32;
        if score >= MATE_BOUND {
            score -= ply as i32;
        } else if score <= -MATE_BOUND {
            score += ply as i32;
        }
        let bound = match slot.bound {
            1 => Bound::Exact,
            2 => Bound::Lower,
            _ => Bound::Upper,
        };
        Some(Entry { mv: unpack_move(slot.mv), score, depth: slot.depth, bound })
    }

    /// Stores an entry whose score is relative to `ply`. Always replaces.
    pub fn store(&mut self, hash: u64, ply: usize, e: Entry) {
        // Mate scores are kept as distance from this node, not from the root.
        let mut score = e.score;
        if score >= MATE_BOUND {
            score += ply as i32;
        } else if score <= -MATE_BOUND {
            score -= ply as i32;
        }
        let bound = match e.bound {
            Bound::Exact => 1,
            Bound::Lower => 2,
            Bound::Upper => 3,
        };
        self.slots[hash as usize & self.mask] =
            Slot { key: hash, score: score as i16, mv: pack_move(e.mv), depth: e.depth, bound };
    }
}
