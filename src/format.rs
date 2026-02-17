pub const MAGIC: &[u8; 4] = b"QICM";
pub const FMT_V7: u16 = 7;
pub const FMT_V8: u16 = 8;
/// Header: 4 bytes magic + 2 bytes version (LE) + 4 bytes preprocessed length (LE) = 10 bytes
pub const HEADER_SIZE: usize = 10;

pub fn write_header(version: u16, preprocessed_len: u32) -> Vec<u8> {
    let mut hdr = Vec::with_capacity(HEADER_SIZE);
    hdr.extend_from_slice(MAGIC);
    hdr.extend_from_slice(&version.to_le_bytes());
    hdr.extend_from_slice(&preprocessed_len.to_le_bytes());
    hdr
}

pub fn read_header(data: &[u8]) -> Result<(u16, u32), String> {
    if data.len() < HEADER_SIZE {
        return Err("Data too short for QICM header".into());
    }
    if &data[0..4] != MAGIC {
        return Err("Not a QICM file".into());
    }
    let ver = u16::from_le_bytes([data[4], data[5]]);
    if ver != FMT_V7 && ver != FMT_V8 {
        return Err(format!("Unsupported version {ver}"));
    }
    let orig_len = u32::from_le_bytes([data[6], data[7], data[8], data[9]]);
    Ok((ver, orig_len))
}
