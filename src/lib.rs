pub mod fnv;
pub mod dict;
pub mod pretrain;
pub mod charfreq;
pub mod bitio;
pub mod arithmetic;
pub mod ppm;
pub mod lzp;
pub mod mixer;
pub mod format;

use bitio::{BitWriter, BitReader};
use arithmetic::{AEnc, ADec};
use mixer::ContextMixer;
use ppm::PPM;
use lzp::LZP;

pub fn quantum_compress(text: &str) -> Vec<u8> {
    quantum_compress_threads(text, 0)
}

pub fn quantum_compress_threads(text: &str, threads: usize) -> Vec<u8> {
    let raw = text.as_bytes();
    let data = dict::preprocess(text);
    let n = data.len();
    let orig_size = raw.len();

    let pretrain_data = dict::preprocess(pretrain::PRETRAIN);

    // Determine thread count
    let num_threads = if threads > 0 {
        threads
    } else {
        std::thread::available_parallelism().map(|p| p.get()).unwrap_or(1)
    };

    // Use single-threaded V8 for small files or 1 thread
    let min_block = 65536;
    let num_blocks = if num_threads <= 1 || n < min_block * 2 {
        1
    } else {
        std::cmp::min(num_threads, n / min_block)
    };

    if num_blocks == 1 {
        return compress_single(&data, &pretrain_data, orig_size);
    }

    // Split data into blocks
    let block_size = n / num_blocks;
    let mut blocks: Vec<&[u8]> = Vec::with_capacity(num_blocks);
    for i in 0..num_blocks {
        let start = i * block_size;
        let end = if i == num_blocks - 1 { n } else { (i + 1) * block_size };
        blocks.push(&data[start..end]);
    }

    eprintln!("  Compressing with {} threads ({} blocks)...", num_blocks, num_blocks);

    // Pretrain one mixer, then clone for each block
    let mut base_cm = ContextMixer::with_default_order();
    base_cm.pretrain(&pretrain_data);

    // Compress blocks in parallel
    let compressed_blocks: Vec<Vec<u8>> = std::thread::scope(|s| {
        let handles: Vec<_> = blocks.iter().map(|block| {
            let mut cm = base_cm.clone();
            s.spawn(move || {
                let mut bw = BitWriter::new();
                {
                    let mut enc = AEnc::new(&mut bw);
                    for &byte in block.iter() {
                        cm.encode_byte(byte, &mut enc);
                    }
                    enc.finish();
                }
                bw.data().to_vec()
            })
        }).collect();

        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    // Build V9 output
    let block_sizes: Vec<(u32, u32)> = blocks.iter().zip(compressed_blocks.iter())
        .map(|(blk, comp)| (blk.len() as u32, comp.len() as u32))
        .collect();

    let mut result = format::write_header_v9(n as u32, &block_sizes);
    for comp in &compressed_blocks {
        result.extend_from_slice(comp);
    }

    eprintln!("\r  Compressing: 100%");
    eprintln!(
        "  {} \u{2192} {} bytes ({:.1}%)",
        orig_size,
        result.len(),
        result.len() as f64 * 100.0 / orig_size as f64
    );
    result
}

fn compress_single(data: &[u8], pretrain_data: &[u8], orig_size: usize) -> Vec<u8> {
    let n = data.len();
    let mut cm = ContextMixer::with_default_order();
    cm.pretrain(pretrain_data);

    let mut bw = BitWriter::new();
    {
        let mut enc = AEnc::new(&mut bw);
        let step = std::cmp::max(1, n / 20);
        for i in 0..n {
            if i % step == 0 {
                eprint!("\r  Compressing: {}%", i * 100 / n);
            }
            cm.encode_byte(data[i], &mut enc);
        }
        enc.finish();
    }

    let compressed = bw.data();
    let mut result = format::write_header(format::FMT_V8, n as u32);
    result.extend_from_slice(compressed);
    eprintln!("\r  Compressing: 100%");
    eprintln!(
        "  {} \u{2192} {} bytes ({:.1}%)",
        orig_size,
        result.len(),
        result.len() as f64 * 100.0 / orig_size as f64
    );
    result
}

pub fn quantum_decompress(data: &[u8]) -> Result<String, String> {
    quantum_decompress_threads(data, 0)
}

pub fn quantum_decompress_threads(data: &[u8], threads: usize) -> Result<String, String> {
    let (version, _orig_len) = format::read_header(data)?;

    let pretrain_data = dict::preprocess(pretrain::PRETRAIN);

    let result = match version {
        format::FMT_V7 => {
            let orig_len = _orig_len as usize;
            let br = BitReader::new(&data[format::HEADER_SIZE..]);
            decompress_v7(&pretrain_data, orig_len, br)
        }
        format::FMT_V8 => {
            let orig_len = _orig_len as usize;
            let br = BitReader::new(&data[format::HEADER_SIZE..]);
            decompress_v8(&pretrain_data, orig_len, br)
        }
        format::FMT_V9 => {
            decompress_v9(data, &pretrain_data, threads)?
        }
        _ => return Err(format!("Unsupported version {version}")),
    };

    eprintln!("\r  Decompressing: 100%    ");
    Ok(dict::unpreprocess(&result))
}

fn decompress_v7(pretrain_data: &[u8], orig_len: usize, br: BitReader) -> Vec<u8> {
    let mut ppm = PPM::with_default_order();
    ppm.pretrain(pretrain_data);
    let mut lzp = LZP::new();
    lzp.pretrain(pretrain_data);

    let mut dec = ADec::new(br);
    let mut result = Vec::with_capacity(orig_len);
    let step = std::cmp::max(1, orig_len / 20);
    for i in 0..orig_len {
        if i % step == 0 {
            eprint!("\r  Decompressing: {}%", i * 100 / orig_len);
        }
        let byte = ppm.decode_byte(&mut dec, lzp.pred, lzp.pred_len);
        result.push(byte);
        lzp.update(byte);
    }
    result
}

fn decompress_v8(pretrain_data: &[u8], orig_len: usize, br: BitReader) -> Vec<u8> {
    let mut cm = ContextMixer::with_default_order();
    cm.pretrain(pretrain_data);

    let mut dec = ADec::new(br);
    let mut result = Vec::with_capacity(orig_len);
    let step = std::cmp::max(1, orig_len / 20);
    for i in 0..orig_len {
        if i % step == 0 {
            eprint!("\r  Decompressing: {}%", i * 100 / orig_len);
        }
        let byte = cm.decode_byte(&mut dec);
        result.push(byte);
    }
    result
}

fn decompress_v9(data: &[u8], pretrain_data: &[u8], _threads: usize) -> Result<Vec<u8>, String> {
    let (_total_len, block_meta) = format::read_header_v9(data)?;
    let num_blocks = block_meta.len();
    let hdr_size = format::header_size_v9(num_blocks);

    // Calculate offsets for each block's compressed data
    let mut block_offsets = Vec::with_capacity(num_blocks);
    let mut offset = hdr_size;
    for &(_preproc_len, compressed_len) in &block_meta {
        block_offsets.push(offset);
        offset += compressed_len as usize;
    }

    eprintln!("  Decompressing {} blocks in parallel...", num_blocks);

    // Pretrain one mixer, clone for each block
    let mut base_cm = ContextMixer::with_default_order();
    base_cm.pretrain(pretrain_data);

    let decoded_blocks: Vec<Vec<u8>> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..num_blocks).map(|i| {
            let mut cm = base_cm.clone();
            let block_start = block_offsets[i];
            let compressed_len = block_meta[i].1 as usize;
            let preproc_len = block_meta[i].0 as usize;
            let block_data = &data[block_start..block_start + compressed_len];
            s.spawn(move || {
                let br = BitReader::new(block_data);
                let mut dec = ADec::new(br);
                let mut result = Vec::with_capacity(preproc_len);
                for _ in 0..preproc_len {
                    let byte = cm.decode_byte(&mut dec);
                    result.push(byte);
                }
                result
            })
        }).collect();

        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    // Concatenate blocks in order
    let total: usize = decoded_blocks.iter().map(|b| b.len()).sum();
    let mut result = Vec::with_capacity(total);
    for block in decoded_blocks {
        result.extend_from_slice(&block);
    }
    Ok(result)
}
