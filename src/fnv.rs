/// FNV-1a 32-bit hash over a byte slice range.
#[inline]
pub fn fnv(d: &[u8], s: usize, e: usize) -> u32 {
    let mut h: u32 = 2166136261;
    for i in s..e {
        h = (h ^ d[i] as u32).wrapping_mul(16777619);
    }
    h
}
