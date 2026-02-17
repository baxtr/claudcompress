pub const MAGIC: &[u8; 4] = b"QICM";
pub const FMT_V7: u16 = 7;
pub const FMT_V8: u16 = 8;
pub const FMT_V9: u16 = 9;
/// Header: 4 bytes magic + 2 bytes version (LE) + 4 bytes preprocessed length (LE) = 10 bytes
pub const HEADER_SIZE: usize = 10;

pub fn write_header(version: u16, preprocessed_len: u32) -> Vec<u8> {
    let mut hdr = Vec::with_capacity(HEADER_SIZE);
    hdr.extend_from_slice(MAGIC);
    hdr.extend_from_slice(&version.to_le_bytes());
    hdr.extend_from_slice(&preprocessed_len.to_le_bytes());
    hdr
}

/// V9 header: magic(4) + version(2) + total_preproc_len(4) + num_blocks(2) + per-block (preproc_len(4) + compressed_len(4))
pub fn write_header_v9(total_preproc_len: u32, block_sizes: &[(u32, u32)]) -> Vec<u8> {
    let num_blocks = block_sizes.len() as u16;
    let hdr_size = 12 + 8 * block_sizes.len();
    let mut hdr = Vec::with_capacity(hdr_size);
    hdr.extend_from_slice(MAGIC);
    hdr.extend_from_slice(&FMT_V9.to_le_bytes());
    hdr.extend_from_slice(&total_preproc_len.to_le_bytes());
    hdr.extend_from_slice(&num_blocks.to_le_bytes());
    for &(preproc_len, compressed_len) in block_sizes {
        hdr.extend_from_slice(&preproc_len.to_le_bytes());
        hdr.extend_from_slice(&compressed_len.to_le_bytes());
    }
    hdr
}

/// Parse V9 header. Returns (total_preproc_len, vec of (preproc_len, compressed_len)).
pub fn read_header_v9(data: &[u8]) -> Result<(u32, Vec<(u32, u32)>), String> {
    if data.len() < 12 {
        return Err("Data too short for V9 header".into());
    }
    let num_blocks = u16::from_le_bytes([data[10], data[11]]) as usize;
    let needed = 12 + 8 * num_blocks;
    if data.len() < needed {
        return Err("Data too short for V9 block metadata".into());
    }
    let total_len = u32::from_le_bytes([data[6], data[7], data[8], data[9]]);
    let mut blocks = Vec::with_capacity(num_blocks);
    for i in 0..num_blocks {
        let off = 12 + i * 8;
        let preproc_len = u32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]]);
        let compressed_len = u32::from_le_bytes([data[off + 4], data[off + 5], data[off + 6], data[off + 7]]);
        blocks.push((preproc_len, compressed_len));
    }
    Ok((total_len, blocks))
}

/// Header size for V9 format given number of blocks.
pub fn header_size_v9(num_blocks: usize) -> usize {
    12 + 8 * num_blocks
}

pub fn read_header(data: &[u8]) -> Result<(u16, u32), String> {
    if data.len() < HEADER_SIZE {
        return Err("Data too short for QICM header".into());
    }
    if &data[0..4] != MAGIC {
        return Err("Not a QICM file".into());
    }
    let ver = u16::from_le_bytes([data[4], data[5]]);
    if ver != FMT_V7 && ver != FMT_V8 && ver != FMT_V9 {
        return Err(format!("Unsupported version {ver}"));
    }
    let orig_len = u32::from_le_bytes([data[6], data[7], data[8], data[9]]);
    Ok((ver, orig_len))
}
