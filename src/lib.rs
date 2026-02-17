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
    let raw = text.as_bytes();
    let data = dict::preprocess(text);
    let n = data.len();
    let orig_size = raw.len();

    let pretrain_data = dict::preprocess(pretrain::PRETRAIN);
    let mut cm = ContextMixer::with_default_order();
    cm.pretrain(&pretrain_data);

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
    let (version, orig_len) = format::read_header(data)?;
    let orig_len = orig_len as usize;

    let pretrain_data = dict::preprocess(pretrain::PRETRAIN);
    let br = BitReader::new(&data[format::HEADER_SIZE..]);

    let result = match version {
        format::FMT_V7 => decompress_v7(&pretrain_data, orig_len, br),
        format::FMT_V8 => decompress_v8(&pretrain_data, orig_len, br),
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
