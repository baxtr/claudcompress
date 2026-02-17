use crate::bitio::{BitReader, BitWriter};

const PREC: u32 = 32;
const WHOLE: u64 = 1u64 << PREC;
const HALF: u64 = WHOLE >> 1;
const QTR: u64 = WHOLE >> 2;
const THREE_Q: u64 = 3 * QTR;

/// Arithmetic encoder (32-bit precision).
pub struct AEnc<'a> {
    w: &'a mut BitWriter,
    lo: u64,
    hi: u64,
    pend: u32,
}

impl<'a> AEnc<'a> {
    pub fn new(writer: &'a mut BitWriter) -> Self {
        Self {
            w: writer,
            lo: 0,
            hi: WHOLE - 1,
            pend: 0,
        }
    }

    pub fn encode(&mut self, cl: u64, ch: u64, total: u64) {
        let r = self.hi - self.lo + 1;
        self.hi = self.lo + (r * ch) / total - 1;
        self.lo = self.lo + (r * cl) / total;
        self.renorm();
    }

    pub fn finish(&mut self) {
        self.pend += 1;
        if self.lo < QTR {
            self.emit(0);
        } else {
            self.emit(1);
        }
    }

    fn renorm(&mut self) {
        loop {
            if self.hi < HALF {
                self.emit(0);
            } else if self.lo >= HALF {
                self.emit(1);
                self.lo -= HALF;
                self.hi -= HALF;
            } else if self.lo >= QTR && self.hi < THREE_Q {
                self.pend += 1;
                self.lo -= QTR;
                self.hi -= QTR;
            } else {
                break;
            }
            self.lo <<= 1;
            self.hi = (self.hi << 1) | 1;
        }
    }

    pub fn encode_bit(&mut self, bit: u8, p1: u64, scale: u64) {
        let threshold = scale - p1;
        if bit != 0 {
            self.encode(threshold, scale, scale);
        } else {
            self.encode(0, threshold, scale);
        }
    }

    fn emit(&mut self, bit: u8) {
        self.w.write(bit);
        for _ in 0..self.pend {
            self.w.write(1 - bit);
        }
        self.pend = 0;
    }
}

/// Arithmetic decoder (32-bit precision).
pub struct ADec<'a> {
    r: BitReader<'a>,
    lo: u64,
    hi: u64,
    val: u64,
}

impl<'a> ADec<'a> {
    pub fn new(mut reader: BitReader<'a>) -> Self {
        let mut val: u64 = 0;
        for _ in 0..PREC {
            val = (val << 1) | reader.read() as u64;
        }
        Self {
            r: reader,
            lo: 0,
            hi: WHOLE - 1,
            val,
        }
    }

    pub fn decode(&mut self, cum: &[u64], total: u64) -> usize {
        let r = self.hi - self.lo + 1;
        let scaled = ((self.val - self.lo + 1) * total - 1) / r;
        // Binary search for the symbol
        let mut lo_s: usize = 0;
        let mut hi_s: usize = cum.len() - 2;
        while lo_s < hi_s {
            let mid = (lo_s + hi_s) >> 1;
            if cum[mid + 1] <= scaled {
                lo_s = mid + 1;
            } else {
                hi_s = mid;
            }
        }
        let idx = lo_s;
        self.hi = self.lo + (r * cum[idx + 1]) / total - 1;
        self.lo = self.lo + (r * cum[idx]) / total;
        self.renorm();
        idx
    }

    pub fn decode_bit(&mut self, p1: u64, scale: u64) -> u8 {
        let r = self.hi - self.lo + 1;
        let threshold = scale - p1;
        let scaled = ((self.val - self.lo + 1) * scale - 1) / r;
        if scaled < threshold {
            self.hi = self.lo + (r * threshold) / scale - 1;
            self.renorm();
            0
        } else {
            self.lo = self.lo + (r * threshold) / scale;
            self.renorm();
            1
        }
    }

    fn renorm(&mut self) {
        loop {
            if self.hi < HALF {
                // pass
            } else if self.lo >= HALF {
                self.val -= HALF;
                self.lo -= HALF;
                self.hi -= HALF;
            } else if self.lo >= QTR && self.hi < THREE_Q {
                self.val -= QTR;
                self.lo -= QTR;
                self.hi -= QTR;
            } else {
                break;
            }
            self.lo <<= 1;
            self.hi = (self.hi << 1) | 1;
            self.val = (self.val << 1) | self.r.read() as u64;
        }
    }
}
