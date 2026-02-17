use crate::dict::CAP_MARKER;

const fn build_char_freq() -> [u32; 256] {
    let mut f = [1u32; 256];

    // Common ASCII characters in English text (relative frequencies)
    f[32] = 700; f[101] = 390; f[116] = 280; f[97] = 250; f[111] = 230;
    f[105] = 215; f[110] = 210; f[115] = 195; f[104] = 185; f[114] = 183;
    f[100] = 130; f[108] = 120; f[99] = 85; f[117] = 85; f[109] = 75;
    f[119] = 72; f[102] = 70; f[103] = 61; f[121] = 59; f[112] = 55;
    f[98] = 45; f[118] = 30; f[107] = 22; f[106] = 5; f[120] = 5;
    f[113] = 3; f[122] = 2; f[10] = 50; f[44] = 40; f[46] = 40;
    f[84] = 30; f[65] = 25; f[73] = 20; f[83] = 20; f[87] = 15;
    f[66] = 12; f[67] = 12; f[68] = 12; f[69] = 12; f[70] = 12;
    f[71] = 10; f[72] = 10; f[76] = 10; f[77] = 10; f[78] = 10;
    f[79] = 10; f[80] = 10; f[82] = 10; f[74] = 5; f[75] = 5;
    f[81] = 3; f[85] = 8; f[86] = 5; f[88] = 2; f[89] = 5;
    f[90] = 2; f[39] = 8; f[45] = 8; f[34] = 5; f[40] = 3;
    f[41] = 3; f[48] = 3; f[49] = 5; f[50] = 3; f[51] = 3;
    f[52] = 3; f[53] = 3; f[54] = 3; f[55] = 3; f[56] = 3;
    f[57] = 3; f[58] = 5; f[59] = 3;

    // Capitalize marker
    f[CAP_MARKER as usize] = 30;

    // Word tokens (most common first): max(3, 60 - i) for i in 0..127
    let mut i = 0usize;
    while i < 127 {
        let v = if 60 > i as u32 { 60 - i as u32 } else { 0 };
        f[129 + i] = if v >= 3 { v } else { 3 };
        i += 1;
    }

    f
}

pub const CHAR_FREQ: [u32; 256] = build_char_freq();
