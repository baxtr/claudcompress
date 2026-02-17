/// Bit writer — MSB-first, accumulates into a byte buffer.
pub struct BitWriter {
    pub buf: Vec<u8>,
    cur: u8,
    pos: i8,
}

impl BitWriter {
    pub fn new() -> Self {
        Self {
            buf: Vec::new(),
            cur: 0,
            pos: 7,
        }
    }

    #[inline]
    pub fn write(&mut self, bit: u8) {
        if bit != 0 {
            self.cur |= 1 << self.pos;
        }
        self.pos -= 1;
        if self.pos < 0 {
            self.buf.push(self.cur);
            self.cur = 0;
            self.pos = 7;
        }
    }

    pub fn data(&mut self) -> &[u8] {
        if self.pos < 7 {
            self.buf.push(self.cur);
            self.cur = 0;
            self.pos = 7;
        }
        &self.buf
    }
}

/// Bit reader — MSB-first, reads from a byte slice.
pub struct BitReader<'a> {
    d: &'a [u8],
    bi: usize,
    bp: i8,
}

impl<'a> BitReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self {
            d: data,
            bi: 0,
            bp: 7,
        }
    }

    #[inline]
    pub fn read(&mut self) -> u8 {
        if self.bi >= self.d.len() {
            return 0;
        }
        let bit = (self.d[self.bi] >> self.bp) & 1;
        self.bp -= 1;
        if self.bp < 0 {
            self.bp = 7;
            self.bi += 1;
        }
        bit
    }
}
