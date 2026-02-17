use rustc_hash::FxHashMap;
use crate::fnv::fnv;

/// LZP longest-match predictor.
/// table[hash] = pos where pos is the position of the byte that FOLLOWED the hashed context.
#[derive(Clone)]
pub struct LZP {
    pub hist: Vec<u8>,
    table: FxHashMap<u32, usize>,
    pub pred: i32,
    pub pred_len: i32,
}

impl LZP {
    pub fn new() -> Self {
        Self {
            hist: Vec::new(),
            table: FxHashMap::default(),
            pred: -1,
            pred_len: 0,
        }
    }

    pub fn update(&mut self, byte: u8) {
        let n = self.hist.len();
        // Store: for context hist[n-ctx_len..n], the following byte will be at position n
        let max_ctx = if n + 1 < 25 { n + 1 } else { 25 };
        for ctx_len in 3..max_ctx {
            let h = fnv(&self.hist, n - ctx_len, n);
            self.table.insert(h, n);
        }
        self.hist.push(byte);

        // Find match for the NEXT byte
        self.pred = -1;
        self.pred_len = 0;
        let n = self.hist.len();
        let start_ctx = if n < 24 { n } else { 24 };
        for ctx_len in (3..=start_ctx).rev() {
            let h = fnv(&self.hist, n - ctx_len, n);
            let pos = match self.table.get(&h) {
                Some(&p) => p,
                None => continue,
            };
            if pos >= n || pos < ctx_len {
                continue;
            }
            // Verify: stored context hist[pos-ctx_len..pos] == current hist[n-ctx_len..n]
            let mut ok = true;
            for j in 0..ctx_len {
                if self.hist[pos - ctx_len + j] != self.hist[n - ctx_len + j] {
                    ok = false;
                    break;
                }
            }
            if !ok {
                continue;
            }
            self.pred = self.hist[pos] as i32;
            self.pred_len = ctx_len as i32;
            return;
        }
    }

    pub fn pretrain(&mut self, data: &[u8]) {
        for &b in data {
            self.update(b);
        }
    }
}
