use crate::errors::{TextResult, invalid};
use rivet_data::random::RandomStream;

const COMMON_WORDS: &[&str] = &[
    "the", "a", "is", "it", "and", "to", "of", "in", "for", "with",
];
const PADDING: &[&str] = &["[PAD]", "[UNK]", "[MASK]"];
const KEYBOARD: [&[u8]; 26] = [
    b"qsz", b"vghn", b"xdfv", b"serfcx", b"wrds", b"dgvcr", b"fhbvd", b"gjnb", b"uokj", b"hkmni",
    b"jloim", b"kop", b"njkl", b"bhjm", b"iplk", b"ol", b"was", b"etfd", b"awdxz", b"rygf",
    b"yijh", b"cfgbx", b"qesa", b"zsdcv", b"tuhg", b"asx",
];

/// Raw-text perturbations from the NLP robustness evaluation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TextNoiseKind {
    Clean,
    CharDelete,
    CharInsert,
    CharSwap,
    CharSubstitute,
    WordDelete,
    WordSwap,
    WordInsert,
    Truncate50,
    Truncate80,
    PadNoise,
    Uppercase,
    Lowercase,
    Spelling,
}

const KINDS: [(TextNoiseKind, &str); 14] = {
    use TextNoiseKind::*;
    [
        (Clean, "clean"),
        (CharDelete, "char_delete"),
        (CharInsert, "char_insert"),
        (CharSwap, "char_swap"),
        (CharSubstitute, "char_substitute"),
        (WordDelete, "word_delete"),
        (WordSwap, "word_swap"),
        (WordInsert, "word_insert"),
        (Truncate50, "truncate_50"),
        (Truncate80, "truncate_80"),
        (PadNoise, "pad_noise"),
        (Uppercase, "uppercase"),
        (Lowercase, "lowercase"),
        (Spelling, "spelling"),
    ]
};

impl std::str::FromStr for TextNoiseKind {
    type Err = crate::TextError;

    fn from_str(name: &str) -> TextResult<Self> {
        KINDS
            .iter()
            .find_map(|&(kind, key)| (name == key).then_some(kind))
            .ok_or_else(|| invalid(format!("unknown text noise: {name}")))
    }
}

/// A validated perturbation. Count-based operations apply at least one edit
/// when applicable, including at ratio zero, following the evaluation script.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TextNoise {
    kind: TextNoiseKind,
    ratio: f64,
}

// Construction excludes NaN.
impl Eq for TextNoise {}

impl TextNoise {
    pub fn new(kind: TextNoiseKind, ratio: f64) -> TextResult<Self> {
        if !(0.0..=1.0).contains(&ratio) {
            return Err(invalid("text noise ratio must be in [0, 1]"));
        }
        Ok(Self { kind, ratio })
    }

    pub const fn kind(self) -> TextNoiseKind {
        self.kind
    }

    pub const fn ratio(self) -> f64 {
        self.ratio
    }

    pub(crate) fn name(self) -> &'static str {
        KINDS.iter().find(|(kind, _)| *kind == self.kind).unwrap().1
    }

    pub fn apply(self, text: &str, rng: &mut RandomStream) -> TextResult<String> {
        use TextNoiseKind::*;
        match self.kind {
            Clean => Ok(text.to_owned()),
            Lowercase => Ok(text.to_lowercase()),
            Uppercase => {
                let mut output = String::with_capacity(text.len());
                for ch in text.chars() {
                    if rng.next_f64() < self.ratio {
                        output.extend(ch.to_uppercase());
                    } else {
                        output.push(ch);
                    }
                }
                Ok(output)
            }
            CharDelete | CharInsert | CharSwap | CharSubstitute => self.apply_chars(text, rng),
            _ => self.apply_words(text, rng),
        }
    }

    fn apply_chars(self, text: &str, rng: &mut RandomStream) -> TextResult<String> {
        use TextNoiseKind::*;
        let mut chars: Vec<char> = text.chars().collect();
        let count = self.edit_count(chars.len());
        match self.kind {
            CharDelete => delete(&mut chars, count, rng)?,
            CharInsert => {
                for _ in 0..count {
                    let index = rng.gen_range_usize(0..chars.len() + 1)?;
                    let letter = b'a' + rng.gen_range_usize(0..26)? as u8;
                    chars.insert(index, char::from(letter));
                }
            }
            CharSwap => swap_adjacent(&mut chars, count, rng)?,
            CharSubstitute => {
                let mut candidates: Vec<_> = chars
                    .iter()
                    .enumerate()
                    .filter_map(|(index, ch)| {
                        let mut lower = ch.to_lowercase();
                        let letter = lower.next()?;
                        (letter.is_ascii_lowercase() && lower.next().is_none())
                            .then_some((index, letter))
                    })
                    .collect();
                sample_prefix(&mut candidates, count, rng)?;
                for &(index, letter) in candidates.iter().take(count) {
                    let neighbors = KEYBOARD[letter as usize - 'a' as usize];
                    chars[index] = char::from(neighbors[rng.gen_range_usize(0..neighbors.len())?]);
                }
            }
            _ => unreachable!(),
        }
        Ok(chars.into_iter().collect())
    }

    fn apply_words(self, text: &str, rng: &mut RandomStream) -> TextResult<String> {
        use TextNoiseKind::*;
        // Python str.split() also treats the ASCII information separators as whitespace.
        let mut words: Vec<_> = text
            .split(|ch: char| ch.is_whitespace() || (28..=31).contains(&(ch as u32)))
            .filter(|word| !word.is_empty())
            .collect();
        let count = self.edit_count(words.len());
        match self.kind {
            WordDelete | WordSwap if words.len() < 2 => return Ok(text.to_owned()),
            WordDelete => {
                let count = count.min(words.len() - 1);
                delete(&mut words, count, rng)?;
            }
            WordSwap => swap_adjacent(&mut words, count, rng)?,
            WordInsert | PadNoise => {
                let pool = if self.kind == WordInsert {
                    COMMON_WORDS
                } else {
                    PADDING
                };
                for _ in 0..count {
                    let index = rng.gen_range_usize(0..words.len() + 1)?;
                    words.insert(index, pool[rng.gen_range_usize(0..pool.len())?]);
                }
            }
            Truncate50 => words.truncate((words.len() / 2).max(1)),
            Truncate80 => words.truncate((words.len() / 5).max(1)),
            Spelling => {
                for word in &mut words {
                    if let Some(replacement) = misspelling(&word.to_lowercase())
                        && rng.next_f64() < self.ratio
                    {
                        *word = replacement;
                    }
                }
            }
            _ => unreachable!(),
        }
        Ok(words.join(" "))
    }

    fn edit_count(self, len: usize) -> usize {
        ((len as f64 * self.ratio) as usize).max(1)
    }
}

// Partial Fisher–Yates samples uniformly without replacement.
fn sample_prefix<T>(values: &mut [T], count: usize, rng: &mut RandomStream) -> TextResult<()> {
    for index in 0..count.min(values.len()) {
        let selected = rng.gen_range_usize(index..values.len())?;
        values.swap(index, selected);
    }
    Ok(())
}

fn delete<T>(values: &mut Vec<T>, count: usize, rng: &mut RandomStream) -> TextResult<()> {
    let count = count.min(values.len());
    let mut indices: Vec<_> = (0..values.len()).collect();
    sample_prefix(&mut indices, count, rng)?;
    let mut removed = vec![false; values.len()];
    for &index in &indices[..count] {
        removed[index] = true;
    }
    let mut keep = removed.into_iter();
    values.retain(|_| !keep.next().unwrap());
    Ok(())
}

fn swap_adjacent<T>(values: &mut [T], count: usize, rng: &mut RandomStream) -> TextResult<()> {
    if values.len() >= 2 {
        for _ in 0..count {
            let index = rng.gen_range_usize(0..values.len() - 1)?;
            values.swap(index, index + 1);
        }
    }
    Ok(())
}

fn misspelling(word: &str) -> Option<&'static str> {
    Some(match word {
        "the" => "teh",
        "that" => "thta",
        "with" => "wiht",
        "have" => "haev",
        "this" => "tihs",
        "from" => "form",
        "would" => "woudl",
        "there" | "their" => "thier",
        "what" => "whta",
        "about" => "abotu",
        "which" => "whcih",
        "when" => "wneh",
        "make" => "maek",
        "like" => "lik e",
        "time" => "tim e",
        "very" => "vre y",
        "good" => "goo d",
        _ => return None,
    })
}
