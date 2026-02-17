use std::collections::HashMap;
use std::sync::LazyLock;

pub const CAP_MARKER: u8 = 128;

pub const DICT: &[&str] = &[
    "the", "and", "have", "that", "of", "they", "be", "to",
    "with", "in", "was", "for", "this", "which", "from", "would",
    "not", "there", "she", "he", "their", "his", "are", "it",
    "were", "you", "will", "had", "but", "other", "is", "make",
    "said", "when", "about", "more", "them", "been", "one", "could",
    "what", "state", "her", "as", "all", "time", "on", "say",
    "than", "who", "these", "through", "years", "at", "first", "can",
    "into", "by", "before", "because", "only", "think", "year", "some",
    "we", "man", "take", "him", "out", "come", "should", "after",
    "people", "do", "has", "know", "like", "then", "different", "between",
    "did", "great", "work", "made", "or", "such", "where", "being",
    "little", "give", "over", "another", "most", "even", "find", "become",
    "also", "against", "found", "new", "many", "those", "called", "must",
    "look", "without", "number", "place", "world", "back", "still", "an",
    "long", "see", "use", "get", "much", "its", "well", "down",
    "follow", "during", "any", "just", "under", "right", "thing",
];

static DMAP: LazyLock<HashMap<&'static str, usize>> = LazyLock::new(|| {
    DICT.iter().enumerate().map(|(i, w)| (*w, i)).collect()
});

pub fn preprocess(text: &str) -> Vec<u8> {
    let mut result = Vec::new();
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    let mut i = 0;
    while i < n {
        if chars[i].is_alphabetic() {
            let j_start = i;
            while i < n && chars[i].is_alphabetic() {
                i += 1;
            }
            let word: String = chars[j_start..i].iter().collect();
            let lower = word.to_lowercase();
            if let Some(&idx) = DMAP.get(lower.as_str()) {
                if word == lower {
                    result.push(129 + idx as u8);
                } else {
                    let first_upper = chars[j_start].is_uppercase();
                    let rest_match = word[chars[j_start].len_utf8()..] == lower[chars[j_start].to_lowercase().next().unwrap().len_utf8()..];
                    if first_upper && rest_match {
                        result.push(CAP_MARKER);
                        result.push(129 + idx as u8);
                    } else {
                        result.extend_from_slice(word.as_bytes());
                    }
                }
            } else {
                result.extend_from_slice(word.as_bytes());
            }
        } else {
            // Non-alphabetic: emit as UTF-8 byte(s)
            // Python does ord(text[i]) which only works for single-byte chars
            // For ASCII text this is always a single byte
            let ch = chars[i];
            if (ch as u32) < 128 {
                result.push(ch as u8);
            } else {
                let mut buf = [0u8; 4];
                let s = ch.encode_utf8(&mut buf);
                result.extend_from_slice(s.as_bytes());
            }
            i += 1;
        }
    }
    result
}

pub fn unpreprocess(data: &[u8]) -> String {
    let mut parts = String::new();
    let mut cap_next = false;
    for &b in data {
        if b == CAP_MARKER {
            cap_next = true;
            continue;
        }
        if b >= 129 {
            let word = DICT[(b - 129) as usize];
            if cap_next {
                let mut chars = word.chars();
                if let Some(first) = chars.next() {
                    for c in first.to_uppercase() {
                        parts.push(c);
                    }
                    parts.push_str(chars.as_str());
                }
                cap_next = false;
            } else {
                parts.push_str(word);
            }
        } else {
            cap_next = false;
            parts.push(b as char);
        }
    }
    parts
}
