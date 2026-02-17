use crate::arithmetic::{AEnc, ADec};
use crate::fnv::fnv;
use crate::lzp::LZP;
use crate::ppm::PPM;

const MAX_ORD: usize = 6;
// Models: 1 PPM + 7 orders + 3 skip/sparse + 1 word + 1 match = 13
const N_ORDER_MODELS: usize = MAX_ORD + 1;
const N_EXTRA_MODELS: usize = 5; // skip1, skip2, sparse, word, match
const N_BIT_MODELS: usize = N_ORDER_MODELS + N_EXTRA_MODELS;
const N_MODELS: usize = 1 + N_BIT_MODELS; // 13 total

const BIT_SCALE: u64 = 1 << 15;
const LR: f64 = 0.001;

// ── Residual NN correction ──
const HIDDEN: usize = 6;
const NN_LR: f64 = 0.01;

// ── Secondary Symbol Estimator (SSE) ──
const SSE_BINS: usize = 64;
const SSE_RATE: f64 = 0.005;

// ── Direct-mapped bit context table ──
const BIT_TABLE_BITS: usize = 24;
const BIT_TABLE_SIZE: usize = 1 << BIT_TABLE_BITS;
const BIT_TABLE_MASK: u32 = (BIT_TABLE_SIZE - 1) as u32;

// ── Lookup tables ──
const STRETCH_TABLE_SIZE: usize = 4096;
const SQUASH_TABLE_SIZE: usize = 4096;
const SQUASH_RANGE: f64 = 12.0;

static mut STRETCH_LUT: [f64; STRETCH_TABLE_SIZE + 1] = [0.0; STRETCH_TABLE_SIZE + 1];
static mut SQUASH_LUT: [f64; SQUASH_TABLE_SIZE + 1] = [0.0; SQUASH_TABLE_SIZE + 1];
static TABLES_INIT: std::sync::Once = std::sync::Once::new();

fn init_tables() {
    TABLES_INIT.call_once(|| {
        unsafe {
            for i in 0..=STRETCH_TABLE_SIZE {
                let mut p = i as f64 / STRETCH_TABLE_SIZE as f64;
                p = p.max(1e-4).min(1.0 - 1e-4);
                STRETCH_LUT[i] = (p / (1.0 - p)).ln();
            }
            for i in 0..=SQUASH_TABLE_SIZE {
                let x = (i as f64 / SQUASH_TABLE_SIZE as f64) * 2.0 * SQUASH_RANGE - SQUASH_RANGE;
                SQUASH_LUT[i] = 1.0 / (1.0 + (-x).exp());
            }
        }
    });
}

#[inline(always)]
fn stretch_fast(p: f64) -> f64 {
    let idx = (p * STRETCH_TABLE_SIZE as f64) as usize;
    let idx = idx.min(STRETCH_TABLE_SIZE);
    unsafe { *STRETCH_LUT.get_unchecked(idx) }
}

#[inline(always)]
fn squash_fast(x: f64) -> f64 {
    if x >= SQUASH_RANGE {
        return 1.0 - 1e-5;
    }
    if x <= -SQUASH_RANGE {
        return 1e-5;
    }
    let idx = ((x + SQUASH_RANGE) / (2.0 * SQUASH_RANGE) * SQUASH_TABLE_SIZE as f64) as usize;
    let idx = idx.min(SQUASH_TABLE_SIZE);
    unsafe { *SQUASH_LUT.get_unchecked(idx) }
}

/// Context mixer with 13 models, linear mixing + residual NN correction.
pub struct ContextMixer {
    mo: usize,
    pub(crate) ppm: PPM,
    pub(crate) lzp: LZP,
    hist: Vec<u8>,
    bit_table: Vec<[u16; 2]>,
    /// Running word hash (resets on space/newline)
    word_hash: u32,
    /// Linear mixer weights
    linear_w: [[f64; N_MODELS]; 8],
    nn_w1: [[[f64; N_MODELS]; HIDDEN]; 8],
    nn_b1: [[f64; HIDDEN]; 8],
    nn_w2: [[f64; HIDDEN]; 8],
    nn_b2: [f64; 8],
    /// SSE: adaptive probability refinement per (bit_pos, prob_bin)
    sse: [[f64; SSE_BINS]; 8],
}

impl ContextMixer {
    pub fn new(max_order: usize) -> Self {
        init_tables();

        let mut linear_w = [[0.0f64; N_MODELS]; 8];
        for bp in 0..8 {
            linear_w[bp][0] = 1.0;
        }

        let phi = 0.618033988749895f64;
        let scale = 0.1 / (N_MODELS as f64).sqrt();
        let mut nn_w1 = [[[0.0f64; N_MODELS]; HIDDEN]; 8];
        let mut nn_b1 = [[0.0f64; HIDDEN]; 8];
        for bp in 0..8 {
            for j in 0..HIDDEN {
                for i in 0..N_MODELS {
                    let seed = (bp * HIDDEN * N_MODELS + j * N_MODELS + i) as f64;
                    nn_w1[bp][j][i] = ((seed * phi).fract() - 0.5) * 2.0 * scale;
                }
                nn_b1[bp][j] = ((j as f64 * phi * 7.0).fract() - 0.5) * 0.05;
            }
        }

        Self {
            mo: max_order,
            ppm: PPM::new(max_order),
            lzp: LZP::new(),
            hist: Vec::new(),
            bit_table: vec![[0u16; 2]; BIT_TABLE_SIZE],
            word_hash: 0,
            linear_w,
            nn_w1,
            nn_b1,
            nn_w2: {
                let mut w = [[0.0f64; HIDDEN]; 8];
                for bp in 0..8 {
                    for j in 0..HIDDEN {
                        w[bp][j] = 0.15;
                    }
                }
                w
            },
            nn_b2: [0.0f64; 8],
            sse: {
                let mut s = [[0.0f64; SSE_BINS]; 8];
                for bp in 0..8 {
                    for bin in 0..SSE_BINS {
                        s[bp][bin] = (bin as f64 + 0.5) / SSE_BINS as f64;
                    }
                }
                s
            },
        }
    }

    pub fn with_default_order() -> Self {
        Self::new(MAX_ORD)
    }

    pub fn pretrain(&mut self, data: &[u8]) {
        self.ppm.pretrain(data);
        self.lzp.pretrain(data);

        for &byte in data {
            let mut node: u32 = 1;
            let n = self.hist.len();
            let max_order = std::cmp::min(self.mo + 1, n + 1);

            // Precompute byte-level hashes once per byte
            let (byte_bases, active) = self.precompute_byte_hashes(n, max_order);

            for bit_pos in 0..8u32 {
                let bit = ((byte >> (7 - bit_pos)) & 1) as usize;
                let node_part = node.wrapping_mul(2654435761);

                for m in 0..N_BIT_MODELS {
                    let h = if active[m] { byte_bases[m] ^ node_part } else { 0 };
                    let idx = (h & BIT_TABLE_MASK) as usize;
                    let entry = unsafe { self.bit_table.get_unchecked_mut(idx) };
                    entry[bit] = entry[bit].saturating_add(1);
                }
                node = node * 2 + bit as u32;
            }
            self.update_word_hash(byte);
            self.hist.push(byte);
        }

        for entry in self.bit_table.iter_mut() {
            entry[0] /= 2;
            entry[1] /= 2;
        }
    }

    /// Update word hash — reset on space/newline, accumulate otherwise.
    #[inline(always)]
    fn update_word_hash(&mut self, byte: u8) {
        if byte == 32 || byte == 10 {
            self.word_hash = 0;
        } else {
            self.word_hash = (self.word_hash ^ byte as u32).wrapping_mul(16777619);
        }
    }

    /// Precompute byte-level base hashes (constant across all 8 bit positions).
    /// Returns (base_hashes, active_mask). Final bit hash = base ^ node_part.
    #[inline(always)]
    fn precompute_byte_hashes(
        &self,
        n: usize,
        max_order: usize,
    ) -> ([u32; N_BIT_MODELS], [bool; N_BIT_MODELS]) {
        let mut base = [0u32; N_BIT_MODELS];
        let mut active = [false; N_BIT_MODELS];

        // Order models: base = fnv_hash * FNV_PRIME
        for order in 0..max_order {
            let byte_h = if order == 0 {
                0u32
            } else {
                fnv(&self.hist, n - order, n)
            };
            base[order] = byte_h.wrapping_mul(16777619);
            active[order] = true;
        }
        // Inactive orders: active stays false, hash stays 0

        let oe = N_ORDER_MODELS; // order_end

        // Skip-1: hash(byte[-1], byte[-3])
        if n >= 3 {
            let h = (self.hist[n - 1] as u32)
                .wrapping_mul(16777619)
                ^ (self.hist[n - 3] as u32).wrapping_mul(2654435761);
            base[oe] = h.wrapping_mul(16777619) ^ 0x12345678;
            active[oe] = true;
        }

        // Skip-2: hash(byte[-1], byte[-4])
        if n >= 4 {
            let h = (self.hist[n - 1] as u32)
                .wrapping_mul(16777619)
                ^ (self.hist[n - 4] as u32).wrapping_mul(2654435761);
            base[oe + 1] = h.wrapping_mul(16777619) ^ 0x23456789;
            active[oe + 1] = true;
        }

        // Sparse: hash(byte[-2], byte[-4])
        if n >= 4 {
            let h = (self.hist[n - 2] as u32)
                .wrapping_mul(16777619)
                ^ (self.hist[n - 4] as u32).wrapping_mul(2654435761);
            base[oe + 2] = h.wrapping_mul(16777619) ^ 0x3456789A;
            active[oe + 2] = true;
        }

        // Word context
        if self.word_hash != 0 {
            base[oe + 3] = self.word_hash.wrapping_mul(16777619) ^ 0x456789AB;
            active[oe + 3] = true;
        }

        // Match model
        if self.lzp.pred >= 0 && self.lzp.pred_len >= 4 {
            let len_bucket = if self.lzp.pred_len >= 16 {
                3u32
            } else if self.lzp.pred_len >= 8 {
                2u32
            } else {
                1u32
            };
            let h = (self.lzp.pred as u32)
                .wrapping_mul(16777619)
                ^ len_bucket.wrapping_mul(2654435761);
            base[oe + 4] = h.wrapping_mul(16777619) ^ 0x56789ABC;
            active[oe + 4] = true;
        }

        (base, active)
    }

    /// Combine precomputed byte bases with bit-tree node to get final hashes.
    #[inline(always)]
    fn make_bit_hashes(
        byte_bases: &[u32; N_BIT_MODELS],
        active: &[bool; N_BIT_MODELS],
        node: u32,
    ) -> [u32; N_BIT_MODELS] {
        let node_part = node.wrapping_mul(2654435761);
        let mut hashes = [0u32; N_BIT_MODELS];
        for i in 0..N_BIT_MODELS {
            if active[i] {
                hashes[i] = byte_bases[i] ^ node_part;
            }
        }
        hashes
    }

    #[inline(always)]
    fn ppm_bit_prob(ppm_cum: &[f64; 257], node: u32) -> f64 {
        let depth = (32 - node.leading_zeros()) - 1;
        let lo = ((node - (1 << depth)) << (8 - depth)) as usize;
        let hi = lo + (1usize << (8 - depth));
        let mid = (lo + hi) / 2;
        let p0 = unsafe { *ppm_cum.get_unchecked(mid) - *ppm_cum.get_unchecked(lo) };
        let p1 = unsafe { *ppm_cum.get_unchecked(hi) - *ppm_cum.get_unchecked(mid) };
        let total = p0 + p1;
        if total > 1e-10 {
            p1 / total
        } else {
            0.5
        }
    }

    #[inline(always)]
    fn bit_predict_direct(&self, h: u32) -> f64 {
        let entry = unsafe { self.bit_table.get_unchecked((h & BIT_TABLE_MASK) as usize) };
        let total = (entry[0] as u32 + entry[1] as u32) as f64;
        if total == 0.0 {
            0.5
        } else {
            (entry[1] as f64 + 0.5) / (total + 1.0)
        }
    }

    #[inline(always)]
    fn bit_update_direct(&mut self, h: u32, bit: usize) {
        let entry = unsafe { self.bit_table.get_unchecked_mut((h & BIT_TABLE_MASK) as usize) };
        entry[bit] = entry[bit].saturating_add(1);
        if entry[0] as u32 + entry[1] as u32 > 16 {
            entry[0] = (entry[0] + 1) >> 1;
            entry[1] = (entry[1] + 1) >> 1;
        }
    }

    #[inline(always)]
    fn forward(
        &self,
        bit_pos: usize,
        inputs: &[f64; N_MODELS],
    ) -> (f64, [f64; N_MODELS], [f64; HIDDEN]) {
        let mut stretched = [0.0f64; N_MODELS];
        for i in 0..N_MODELS {
            stretched[i] = stretch_fast(inputs[i]);
        }

        let w = &self.linear_w[bit_pos];
        let mut linear_logit = 0.0f64;
        for i in 0..N_MODELS {
            linear_logit += w[i] * stretched[i];
        }

        let mut hidden = [0.0f64; HIDDEN];
        for j in 0..HIDDEN {
            let mut sum = self.nn_b1[bit_pos][j];
            for i in 0..N_MODELS {
                sum += self.nn_w1[bit_pos][j][i] * stretched[i];
            }
            hidden[j] = squash_fast(sum);
        }
        let mut correction = self.nn_b2[bit_pos];
        for j in 0..HIDDEN {
            correction += self.nn_w2[bit_pos][j] * hidden[j];
        }

        let mixed = squash_fast(linear_logit + correction);
        (mixed, stretched, hidden)
    }

    #[inline(always)]
    fn backward(
        &mut self,
        bit_pos: usize,
        stretched: &[f64; N_MODELS],
        hidden: &[f64; HIDDEN],
        mixed: f64,
        target: f64,
    ) {
        let err = target - mixed;

        let w = &mut self.linear_w[bit_pos];
        for i in 0..N_MODELS {
            w[i] = (w[i] + LR * err * stretched[i]).max(-8.0).min(8.0);
        }

        for j in 0..HIDDEN {
            self.nn_w2[bit_pos][j] =
                (self.nn_w2[bit_pos][j] + NN_LR * err * hidden[j]).max(-4.0).min(4.0);
        }
        self.nn_b2[bit_pos] = (self.nn_b2[bit_pos] + NN_LR * err).max(-4.0).min(4.0);

        for j in 0..HIDDEN {
            let d_hidden = err * self.nn_w2[bit_pos][j] * hidden[j] * (1.0 - hidden[j]);
            for i in 0..N_MODELS {
                self.nn_w1[bit_pos][j][i] =
                    (self.nn_w1[bit_pos][j][i] + NN_LR * d_hidden * stretched[i])
                        .max(-4.0)
                        .min(4.0);
            }
            self.nn_b1[bit_pos][j] =
                (self.nn_b1[bit_pos][j] + NN_LR * d_hidden).max(-4.0).min(4.0);
        }
    }

    fn build_cum_cached(&self, order_hashes: &[u32; 7], max_order: usize) -> [f64; 257] {
        let match_byte = self.lzp.pred;
        let match_len = self.lzp.pred_len;
        let mut dist = self.ppm.distribution_f_cached(order_hashes, max_order);

        if match_byte >= 0 && match_len >= 4 {
            let lzp_w = (match_len as f64 * 0.01).min(0.25);
            let rest = 0.02 / 255.0;
            for b in 0..256 {
                if b as i32 == match_byte {
                    dist[b] = (1.0 - lzp_w) * dist[b] + lzp_w * 0.98;
                } else {
                    dist[b] = (1.0 - lzp_w) * dist[b] + lzp_w * rest;
                }
            }
        }

        let mut ppm_cum = [0.0f64; 257];
        for i in 0..256 {
            ppm_cum[i + 1] = ppm_cum[i] + dist[i];
        }
        ppm_cum
    }

    #[inline(always)]
    fn gather_preds(
        &self,
        node: u32,
        ppm_cum: &[f64; 257],
        hashes: &[u32; N_BIT_MODELS],
    ) -> [f64; N_MODELS] {
        let mut preds = [0.5f64; N_MODELS];
        preds[0] = Self::ppm_bit_prob(ppm_cum, node);
        for m in 0..N_BIT_MODELS {
            preds[1 + m] = self.bit_predict_direct(hashes[m]);
        }
        preds
    }

    pub fn encode_byte(&mut self, byte: u8, enc: &mut AEnc) {
        let n = self.hist.len();
        let max_order = std::cmp::min(self.mo + 1, n + 1);

        // Extract order hashes for PPM (shared computation)
        let mut order_hashes = [0u32; 7];
        for order in 0..max_order {
            order_hashes[order] = if order == 0 {
                0u32
            } else {
                fnv(&self.hist, n - order, n)
            };
        }

        let ppm_cum = self.build_cum_cached(&order_hashes, max_order);

        // Precompute byte-level hashes once (constant across all 8 bits)
        let (byte_bases, active) = self.precompute_byte_hashes(n, max_order);

        let mut node: u32 = 1;
        for bit_pos in 0..8 {
            let bit = (byte >> (7 - bit_pos)) & 1;
            let hashes = Self::make_bit_hashes(&byte_bases, &active, node);
            let preds = self.gather_preds(node, &ppm_cum, &hashes);

            let (mixed, stretched, hidden) = self.forward(bit_pos, &preds);

            // SSE refinement
            let bin_f = mixed * (SSE_BINS - 1) as f64;
            let bin = (bin_f as usize).min(SSE_BINS - 2);
            let frac = bin_f - bin as f64;
            let sse_p = self.sse[bit_pos][bin] * (1.0 - frac) + self.sse[bit_pos][bin + 1] * frac;
            let final_p = 0.7 * mixed + 0.3 * sse_p;

            let p1 = (final_p * BIT_SCALE as f64).round() as u64;
            let p1 = p1.max(1).min(BIT_SCALE - 1);
            enc.encode_bit(bit, p1, BIT_SCALE);

            self.backward(bit_pos, &stretched, &hidden, mixed, bit as f64);

            // SSE update
            let target = bit as f64;
            self.sse[bit_pos][bin] += SSE_RATE * (target - self.sse[bit_pos][bin]);
            self.sse[bit_pos][bin + 1] += SSE_RATE * (target - self.sse[bit_pos][bin + 1]);

            for m in 0..N_BIT_MODELS {
                self.bit_update_direct(hashes[m], bit as usize);
            }
            node = node * 2 + bit as u32;
        }
        self.ppm.update_cached(byte, &order_hashes, max_order);
        self.lzp.update(byte);
        self.update_word_hash(byte);
        self.hist.push(byte);
    }

    pub fn decode_byte(&mut self, dec: &mut ADec) -> u8 {
        let n = self.hist.len();
        let max_order = std::cmp::min(self.mo + 1, n + 1);

        // Extract order hashes for PPM (shared computation)
        let mut order_hashes = [0u32; 7];
        for order in 0..max_order {
            order_hashes[order] = if order == 0 {
                0u32
            } else {
                fnv(&self.hist, n - order, n)
            };
        }

        let ppm_cum = self.build_cum_cached(&order_hashes, max_order);

        // Precompute byte-level hashes once
        let (byte_bases, active) = self.precompute_byte_hashes(n, max_order);

        let mut node: u32 = 1;
        let mut byte_val: u8 = 0;
        for bit_pos in 0..8 {
            let hashes = Self::make_bit_hashes(&byte_bases, &active, node);
            let preds = self.gather_preds(node, &ppm_cum, &hashes);

            let (mixed, stretched, hidden) = self.forward(bit_pos, &preds);

            // SSE refinement
            let bin_f = mixed * (SSE_BINS - 1) as f64;
            let bin = (bin_f as usize).min(SSE_BINS - 2);
            let frac = bin_f - bin as f64;
            let sse_p = self.sse[bit_pos][bin] * (1.0 - frac) + self.sse[bit_pos][bin + 1] * frac;
            let final_p = 0.7 * mixed + 0.3 * sse_p;

            let p1 = (final_p * BIT_SCALE as f64).round() as u64;
            let p1 = p1.max(1).min(BIT_SCALE - 1);
            let bit = dec.decode_bit(p1, BIT_SCALE);

            self.backward(bit_pos, &stretched, &hidden, mixed, bit as f64);

            // SSE update
            let target = bit as f64;
            self.sse[bit_pos][bin] += SSE_RATE * (target - self.sse[bit_pos][bin]);
            self.sse[bit_pos][bin + 1] += SSE_RATE * (target - self.sse[bit_pos][bin + 1]);

            for m in 0..N_BIT_MODELS {
                self.bit_update_direct(hashes[m], bit as usize);
            }
            byte_val = (byte_val << 1) | bit;
            node = node * 2 + bit as u32;
        }
        self.ppm.update_cached(byte_val, &order_hashes, max_order);
        self.lzp.update(byte_val);
        self.update_word_hash(byte_val);
        self.hist.push(byte_val);
        byte_val
    }
}
