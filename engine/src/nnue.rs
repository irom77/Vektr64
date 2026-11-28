//! Network inference: file loading, the incrementally updated accumulator and the
//! integer evaluation (AVX2 when the CPU has it, scalar otherwise; identical scores).
//!
//! The definition of the evaluation is shared with `trainer/src/vektr_trainer/vknn.py`.

use cozy_chess::{BitBoard, Board, Color, Piece};
use std::sync::{Arc, OnceLock};

pub const HIDDEN: usize = 128;
pub const INPUTS: usize = 768;
const QA: i32 = 255;
const QB: i64 = 64;
const SCALE: i64 = 400;
const MAX_FT: i32 = 505;
const MAX_OUT_W: i32 = 127;
const MAGIC: &[u8; 4] = b"VKNN";
const FILE_SIZE: usize = 12 + INPUTS * HIDDEN * 2 + HIDDEN * 2 + 2 * HIDDEN * 2 + 4;

pub const DEFAULT_NET: &[u8] = include_bytes!("../nets/default.vknn");
pub const DEFAULT_CHECK: &str = include_str!("../nets/default.check");

pub struct Network {
    ft_w: Vec<i16>, // INPUTS x HIDDEN, feature-major
    ft_b: [i16; HIDDEN],
    out_w: [i16; 2 * HIDDEN],
    out_b: i32,
}

fn u32_at(bytes: &[u8], pos: usize) -> u32 {
    u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap())
}

fn i16s(bytes: &[u8]) -> impl Iterator<Item = i16> + '_ {
    bytes.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]]))
}

impl Network {
    pub fn from_bytes(bytes: &[u8]) -> Result<Network, String> {
        if bytes.len() < 12 || &bytes[..4] != MAGIC {
            return Err("not a VKNN file".into());
        }
        let version = u32_at(bytes, 4);
        if version != 1 {
            return Err(format!("unsupported VKNN version {version}"));
        }
        let hidden = u32_at(bytes, 8);
        if hidden as usize != HIDDEN {
            return Err(format!("unsupported hidden size {hidden}"));
        }
        if bytes.len() != FILE_SIZE {
            return Err(format!("wrong file length {}, expected {FILE_SIZE}", bytes.len()));
        }
        let mut pos = 12;
        let ft_w: Vec<i16> = i16s(&bytes[pos..pos + INPUTS * HIDDEN * 2]).collect();
        pos += INPUTS * HIDDEN * 2;
        let mut ft_b = [0i16; HIDDEN];
        for (d, s) in ft_b.iter_mut().zip(i16s(&bytes[pos..pos + HIDDEN * 2])) {
            *d = s;
        }
        pos += HIDDEN * 2;
        let mut out_w = [0i16; 2 * HIDDEN];
        for (d, s) in out_w.iter_mut().zip(i16s(&bytes[pos..pos + 2 * HIDDEN * 2])) {
            *d = s;
        }
        pos += 2 * HIDDEN * 2;
        let out_b = u32_at(bytes, pos) as i32;
        if ft_w.iter().chain(ft_b.iter()).any(|&v| (v as i32).abs() > MAX_FT) {
            return Err(format!("feature weights outside +-{MAX_FT}"));
        }
        if out_w.iter().any(|&v| (v as i32).abs() > MAX_OUT_W) {
            return Err(format!("output weights outside +-{MAX_OUT_W}"));
        }
        Ok(Network { ft_w, ft_b, out_w, out_b })
    }

    pub fn load(path: &str) -> Result<Network, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("cannot read {path}: {e}"))?;
        Network::from_bytes(&bytes)
    }

    /// The compiled-in network, parsed once.
    pub fn embedded() -> Arc<Network> {
        static NET: OnceLock<Arc<Network>> = OnceLock::new();
        NET.get_or_init(|| Arc::new(Network::from_bytes(DEFAULT_NET).expect("embedded network is valid")))
            .clone()
    }

    fn row(&self, feature: usize) -> &[i16] {
        &self.ft_w[feature * HIDDEN..(feature + 1) * HIDDEN]
    }

    /// Score from the side to move's point of view, run-time dispatch to AVX2 or scalar.
    pub fn evaluate(&self, acc: &Accumulator, stm: Color) -> i32 {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                // SAFETY: AVX2 support was just detected.
                return unsafe { self.evaluate_avx2(acc, stm) };
            }
        }
        self.evaluate_scalar(acc, stm)
    }

    fn halves<'a>(acc: &'a Accumulator, stm: Color) -> (&'a [i16; HIDDEN], &'a [i16; HIDDEN]) {
        match stm {
            Color::White => (&acc.white, &acc.black),
            Color::Black => (&acc.black, &acc.white),
        }
    }

    fn finish(&self, sum: i32) -> i32 {
        let cp = ((sum as i64 / QA as i64 + self.out_b as i64) * SCALE) / (QA as i64 * QB);
        cp.clamp(-20000, 20000) as i32
    }

    pub fn evaluate_scalar(&self, acc: &Accumulator, stm: Color) -> i32 {
        let (us, them) = Self::halves(acc, stm);
        let mut sum = 0i32;
        for j in 0..HIDDEN {
            let u = (us[j] as i32).clamp(0, QA);
            let t = (them[j] as i32).clamp(0, QA);
            sum += u * u * self.out_w[j] as i32 + t * t * self.out_w[HIDDEN + j] as i32;
        }
        self.finish(sum)
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn evaluate_avx2(&self, acc: &Accumulator, stm: Color) -> i32 {
        use std::arch::x86_64::*;
        let (us, them) = Self::halves(acc, stm);
        let zero = _mm256_setzero_si256();
        let max = _mm256_set1_epi16(QA as i16);
        let mut total = _mm256_setzero_si256();
        for (half, acc_half) in [us, them].into_iter().enumerate() {
            for i in 0..HIDDEN / 16 {
                let v = _mm256_loadu_si256(acc_half.as_ptr().add(i * 16) as *const __m256i);
                let c = _mm256_min_epi16(_mm256_max_epi16(v, zero), max);
                let w = _mm256_loadu_si256(self.out_w.as_ptr().add(half * HIDDEN + i * 16) as *const __m256i);
                // 255 * 127 fits in 16 bits; madd with c then adds pairs in 32 bits.
                total = _mm256_add_epi32(total, _mm256_madd_epi16(_mm256_mullo_epi16(c, w), c));
            }
        }
        let mut lanes = [0i32; 8];
        _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, total);
        self.finish(lanes.iter().sum())
    }

    pub fn evaluate_board(&self, board: &Board) -> i32 {
        self.evaluate(&Accumulator::refresh(self, board), board.side_to_move())
    }
}

/// Feature index as White sees it, and as Black sees it.
fn features(color: Color, piece: Piece, square: usize) -> (usize, usize) {
    let c = color as usize;
    let p = piece as usize;
    (c * 384 + p * 64 + square, (1 - c) * 384 + p * 64 + (square ^ 56))
}

#[derive(Clone)]
#[repr(C, align(32))]
pub struct Accumulator {
    pub white: [i16; HIDDEN],
    pub black: [i16; HIDDEN],
}

fn add_row(half: &mut [i16; HIDDEN], row: &[i16]) {
    for (a, w) in half.iter_mut().zip(row) {
        *a = a.wrapping_add(*w);
    }
}

fn sub_row(half: &mut [i16; HIDDEN], row: &[i16]) {
    for (a, w) in half.iter_mut().zip(row) {
        *a = a.wrapping_sub(*w);
    }
}

impl Accumulator {
    pub fn refresh(net: &Network, board: &Board) -> Accumulator {
        let mut acc = Accumulator { white: net.ft_b, black: net.ft_b };
        for color in Color::ALL {
            for piece in Piece::ALL {
                for sq in board.colored_pieces(color, piece) {
                    let (w, b) = features(color, piece, sq as usize);
                    add_row(&mut acc.white, net.row(w));
                    add_row(&mut acc.black, net.row(b));
                }
            }
        }
        acc
    }

    /// Derive this accumulator from the parent's by comparing the two boards' piece sets,
    /// so every kind of move (castling, en passant, promotion) is handled the same way.
    pub fn update_from(&mut self, parent: &Accumulator, net: &Network, before: &Board, after: &Board) {
        self.white = parent.white;
        self.black = parent.black;
        for color in Color::ALL {
            for piece in Piece::ALL {
                let old: BitBoard = before.colored_pieces(color, piece);
                let new: BitBoard = after.colored_pieces(color, piece);
                for sq in old & !new {
                    let (w, b) = features(color, piece, sq as usize);
                    sub_row(&mut self.white, net.row(w));
                    sub_row(&mut self.black, net.row(b));
                }
                for sq in new & !old {
                    let (w, b) = features(color, piece, sq as usize);
                    add_row(&mut self.white, net.row(w));
                    add_row(&mut self.black, net.row(b));
                }
            }
        }
    }
}

/// Parse a FEN that may omit the move counters (as check-file lines do).
pub fn parse_fen(fen: &str) -> Result<Board, String> {
    let mut text = fen.trim().to_string();
    match text.split_whitespace().count() {
        4 => text.push_str(" 0 1"),
        5 => text.push_str(" 1"),
        _ => {}
    }
    Board::from_fen(&text, false).map_err(|e| format!("bad fen: {e}"))
}

/// Compare the network with a check file of `<fen>;<cp>` lines. Returns the number of
/// positions checked, or one line per disagreement.
pub fn netcheck(net: &Network, check: &str) -> Result<usize, Vec<String>> {
    let mut count = 0;
    let mut errors = Vec::new();
    for line in check.lines().filter(|l| !l.trim().is_empty()) {
        let Some((fen, expected)) = line.rsplit_once(';') else {
            errors.push(format!("malformed line: {line}"));
            continue;
        };
        let (Ok(board), Ok(expected)) = (parse_fen(fen), expected.trim().parse::<i32>()) else {
            errors.push(format!("malformed line: {line}"));
            continue;
        };
        count += 1;
        let got = net.evaluate_board(&board);
        // The file holds unclamped scores; the engine clamps to +-20000.
        let want = expected.clamp(-20000, 20000);
        if got != want {
            errors.push(format!("mismatch {fen}: engine {got} trainer {expected}"));
        }
    }
    if errors.is_empty() {
        Ok(count)
    } else {
        Err(errors)
    }
}
