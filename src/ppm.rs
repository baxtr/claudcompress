use crate::arithmetic::{AEnc, ADec};
use crate::charfreq::CHAR_FREQ;
use crate::fnv::fnv;

const MAX_ORD: usize = 6;
const DISCOUNT: f64 = 0.85;

// ── Open-addressing hash table: u32 -> SymCounts ──

const EMPTY_KEY: u32 = u32::MAX;

/// Compact symbol-count map using flat Vec for cache-friendly iteration.
struct SymCounts {
    entries: Vec<(u8, u32)>,
}

impl SymCounts {
    #[inline]
    fn new() -> Self {
        Self {
            entries: Vec::new(), // zero alloc until first push
        }
    }

    #[inline]
    fn increment(&mut self, sym: u8) {
        for entry in self.entries.iter_mut() {
            if entry.0 == sym {
                entry.1 += 1;
                return;
            }
        }
        self.entries.push((sym, 1));
    }

    #[inline]
    fn total(&self) -> u32 {
        let mut s = 0u32;
        for &(_, c) in &self.entries {
            s += c;
        }
        s
    }

    #[inline]
    fn len(&self) -> usize {
        self.entries.len()
    }

    #[inline]
    fn iter(&self) -> std::slice::Iter<(u8, u32)> {
        self.entries.iter()
    }

    #[inline]
    fn values_mut(&mut self) -> impl Iterator<Item = &mut u32> {
        self.entries.iter_mut().map(|(_, c)| c)
    }
}

/// Open-addressing hash table with linear probing. u32 keys -> SymCounts values.
/// Keys and values in separate arrays: probing only touches the keys array
/// (4 bytes each, 16 keys per cache line), values accessed only on hit.
struct CtxTable {
    keys: Vec<u32>,
    vals: Vec<SymCounts>,
    mask: usize,
    len: usize,
}

impl CtxTable {
    fn new() -> Self {
        let cap = 64; // small initial size, grows as needed
        Self {
            keys: vec![EMPTY_KEY; cap],
            vals: (0..cap).map(|_| SymCounts::new()).collect(),
            mask: cap - 1,
            len: 0,
        }
    }

    #[inline(always)]
    fn fix_key(key: u32) -> u32 {
        if key == EMPTY_KEY { key.wrapping_sub(1) } else { key }
    }

    #[inline]
    fn get(&self, key: u32) -> Option<&SymCounts> {
        let key = Self::fix_key(key);
        let mut idx = (key as usize) & self.mask;
        loop {
            let k = unsafe { *self.keys.get_unchecked(idx) };
            if k == key {
                return Some(unsafe { self.vals.get_unchecked(idx) });
            }
            if k == EMPTY_KEY {
                return None;
            }
            idx = (idx + 1) & self.mask;
        }
    }

    #[inline]
    fn get_or_insert(&mut self, key: u32) -> &mut SymCounts {
        let key = Self::fix_key(key);
        if self.len * 2 >= self.mask + 1 {
            self.grow();
        }
        let mut idx = (key as usize) & self.mask;
        loop {
            let k = unsafe { *self.keys.get_unchecked(idx) };
            if k == key {
                return unsafe { self.vals.get_unchecked_mut(idx) };
            }
            if k == EMPTY_KEY {
                unsafe { *self.keys.get_unchecked_mut(idx) = key; }
                self.len += 1;
                return unsafe { self.vals.get_unchecked_mut(idx) };
            }
            idx = (idx + 1) & self.mask;
        }
    }

    fn grow(&mut self) {
        let new_cap = (self.mask + 1) * 2;
        let new_mask = new_cap - 1;
        let mut new_keys = vec![EMPTY_KEY; new_cap];
        let mut new_vals: Vec<SymCounts> = (0..new_cap).map(|_| SymCounts::new()).collect();

        for old_idx in 0..=self.mask {
            if self.keys[old_idx] != EMPTY_KEY {
                let k = self.keys[old_idx];
                let mut idx = (k as usize) & new_mask;
                loop {
                    if new_keys[idx] == EMPTY_KEY {
                        new_keys[idx] = k;
                        std::mem::swap(&mut new_vals[idx], &mut self.vals[old_idx]);
                        break;
                    }
                    idx = (idx + 1) & new_mask;
                }
            }
        }

        self.keys = new_keys;
        self.vals = new_vals;
        self.mask = new_mask;
    }

    fn values_mut(&mut self) -> impl Iterator<Item = &mut SymCounts> {
        self.keys
            .iter()
            .zip(self.vals.iter_mut())
            .filter(|(&k, _)| k != EMPTY_KEY)
            .map(|(_, v)| v)
    }
}

// ── PPM Model ──

/// PPM model with Kneser-Ney smoothing (no escapes — all orders interpolated).
pub struct PPM {
    mo: usize,
    /// ctx[order] maps context_hash -> symbol counts
    ctx: Vec<CtxTable>,
    hist: Vec<u8>,
    /// Set after pretrain: unigram frequencies from training data
    base_freq: Option<[u32; 256]>,
}

impl PPM {
    pub fn new(max_order: usize) -> Self {
        let mut ctx = Vec::with_capacity(max_order + 1);
        for _ in 0..=max_order {
            ctx.push(CtxTable::new());
        }
        Self {
            mo: max_order,
            ctx,
            hist: Vec::new(),
            base_freq: None,
        }
    }

    pub fn with_default_order() -> Self {
        Self::new(MAX_ORD)
    }

    fn hash(&self, order: usize) -> Option<u32> {
        let n = self.hist.len();
        if order > n {
            return None;
        }
        if order == 0 {
            return Some(0);
        }
        Some(fnv(&self.hist, n - order, n))
    }

    pub(crate) fn update(&mut self, byte: u8) {
        for order in 0..=self.mo {
            let h = match self.hash(order) {
                Some(h) => h,
                None => continue,
            };
            let d = self.ctx[order].get_or_insert(h);
            d.increment(byte);
        }
        self.hist.push(byte);
    }

    /// Update context tables using pre-computed order hashes.
    pub(crate) fn update_cached(&mut self, byte: u8, order_hashes: &[u32; 7], max_order: usize) {
        let limit = std::cmp::min(self.mo + 1, max_order);
        for order in 0..limit {
            let h = order_hashes[order];
            let d = self.ctx[order].get_or_insert(h);
            d.increment(byte);
        }
        self.hist.push(byte);
    }

    pub fn pretrain(&mut self, data: &[u8]) {
        for &b in data {
            self.update(b);
        }
        // Compute pretrain unigram for order -1 base distribution
        let mut base = [1u32; 256];
        for &b in data {
            base[b as usize] += 2;
        }
        self.base_freq = Some(base);

        // Dampen pretrain counts
        for order in 0..=self.mo {
            for d in self.ctx[order].values_mut() {
                for count in d.values_mut() {
                    *count = std::cmp::max(1, isqrt(*count));
                }
            }
        }
    }

    fn class_base(&self) -> [u32; 256] {
        let mut base = [1u32; 256];
        let n = self.hist.len();
        if n == 0 {
            return base;
        }
        let prev = self.hist[n - 1];

        if prev >= 129 {
            base[32] = 150;
            base[44] = 40;
            base[46] = 40;
            base[39] = 15;
            base[10] = 15;
            base[59] = 5;
            base[58] = 5;
            base[45] = 8;
            base[33] = 3;
            base[63] = 3;
        } else if (97..=122).contains(&prev) {
            for b in 97..=122 {
                base[b] = 40;
            }
            base[32] = 120;
            base[44] = 25;
            base[46] = 25;
            base[39] = 15;
            base[45] = 8;
            base[10] = 10;
            for b in 129..=255 {
                base[b] = 5;
            }
        } else if prev == 32 {
            for b in 129..=255 {
                base[b] = 60;
            }
            base[128] = 40;
            for b in 97..=122 {
                base[b] = 25;
            }
            for b in 65..=90 {
                base[b] = 15;
            }
            base[34] = 5;
        } else if prev == 46 || prev == 33 || prev == 63 {
            base[32] = 200;
            base[10] = 50;
        } else if prev == 44 {
            base[32] = 200;
        } else if prev == 10 {
            base[10] = 30;
            for b in 129..=255 {
                base[b] = 25;
            }
            base[128] = 40;
            for b in 65..=90 {
                base[b] = 20;
            }
        } else if prev == 128 {
            for b in 129..=255 {
                base[b] = 80;
            }
        } else if (65..=90).contains(&prev) {
            for b in 97..=122 {
                base[b] = 80;
            }
        }

        base
    }

    /// Compute KN-smoothed float distribution over all 256 bytes.
    pub(crate) fn distribution_f(&self) -> [f64; 256] {
        let mut hashes = [0u32; 7];
        let n = self.hist.len();
        let max_order = std::cmp::min(self.mo + 1, n + 1);
        for order in 0..max_order {
            hashes[order] = match self.hash(order) {
                Some(h) => h,
                None => 0,
            };
        }
        self.distribution_f_cached(&hashes, max_order)
    }

    /// Compute KN-smoothed float distribution using pre-computed order hashes.
    pub(crate) fn distribution_f_cached(
        &self,
        order_hashes: &[u32; 7],
        max_order: usize,
    ) -> [f64; 256] {
        let base_arr: &[u32; 256] = match &self.base_freq {
            Some(bf) => bf,
            None => &CHAR_FREQ,
        };
        let class_base = self.class_base();

        let mut mixed = [0u32; 256];
        let mut freq_total: u64 = 0;
        for b in 0..256 {
            mixed[b] = base_arr[b] + class_base[b];
            freq_total += mixed[b] as u64;
        }
        let inv_freq_total = 1.0 / freq_total as f64;

        let mut dist = [0.0f64; 256];
        for b in 0..256 {
            dist[b] = mixed[b] as f64 * inv_freq_total;
        }

        let limit = std::cmp::min(self.mo + 1, max_order);
        for order in 0..limit {
            let h = order_hashes[order];
            let d = match self.ctx[order].get(h) {
                Some(d) => d,
                None => continue,
            };
            let c_total = d.total();
            if c_total == 0 {
                continue;
            }
            let n_unique = d.len() as f64;
            let inv_c_total = 1.0 / c_total as f64;
            let lam = DISCOUNT * n_unique * inv_c_total;

            let mut new_dist = [0.0f64; 256];
            for b in 0..256 {
                new_dist[b] = lam * dist[b];
            }
            for &(sym, count) in d.iter() {
                let direct = (count as f64 - DISCOUNT).max(0.0) * inv_c_total;
                new_dist[sym as usize] += direct;
            }
            dist = new_dist;
        }

        let eps = 0.10f64;
        let inv_safety_total = inv_freq_total;
        for b in 0..256 {
            dist[b] = (1.0 - eps) * dist[b] + eps * mixed[b] as f64 * inv_safety_total;
        }

        dist
    }

    /// Compute distribution with LZP mixing, returns integer counts for arithmetic coding.
    fn distribution(&self, match_byte: i32, match_len: i32) -> [u32; 256] {
        let mut dist = self.distribution_f();

        if match_byte >= 0 && match_len >= 4 {
            let lzp_w = (match_len as f64 * 0.04).min(0.65);
            let rest = 0.02 / 255.0;
            for b in 0..256 {
                if b as i32 == match_byte {
                    dist[b] = (1.0 - lzp_w) * dist[b] + lzp_w * 0.98;
                } else {
                    dist[b] = (1.0 - lzp_w) * dist[b] + lzp_w * rest;
                }
            }
        }

        let mut counts = [0u32; 256];
        for b in 0..256 {
            counts[b] = std::cmp::max(1, (dist[b] * 65536.0).round() as u32);
        }
        counts
    }

    pub fn encode_byte(&mut self, byte: u8, enc: &mut AEnc, match_byte: i32, match_len: i32) {
        let counts = self.distribution(match_byte, match_len);
        let mut cum: u64 = 0;
        let mut cl: u64 = 0;
        let mut ch: u64 = 0;
        for b in 0..256 {
            if b as u8 == byte {
                cl = cum;
                ch = cum + counts[b] as u64;
            }
            cum += counts[b] as u64;
        }
        enc.encode(cl, ch, cum);
        self.update(byte);
    }

    pub fn decode_byte(&mut self, dec: &mut ADec, match_byte: i32, match_len: i32) -> u8 {
        let counts = self.distribution(match_byte, match_len);
        let mut cum_arr = [0u64; 257];
        for b in 0..256 {
            cum_arr[b + 1] = cum_arr[b] + counts[b] as u64;
        }
        let total = cum_arr[256];
        let idx = dec.decode(&cum_arr, total);
        self.update(idx as u8);
        idx as u8
    }
}

/// Integer square root (floor of sqrt).
fn isqrt(val: u32) -> u32 {
    (val as f64).sqrt() as u32
}
