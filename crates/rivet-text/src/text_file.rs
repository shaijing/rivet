use crate::TextSample;
use crate::errors::{TextResult, invalid};
use rivet_data::dataset::source::{Dataset, SourceCapabilities, validate_indices};
use rivet_data::errors::DataResult;
use std::io::{self, Read};
use std::ops::Range;
use std::path::Path;
use std::sync::Arc;

/// The unit of text represented by one dataset row.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TextSampleBy {
    #[default]
    Line,
    Paragraph,
    Document,
}

/// Encoding of the input bytes. Decoding always produces Unicode text.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TextFileEncoding {
    #[default]
    Utf8,
    Utf8Sig,
    /// Detect byte order from the UTF-16 BOM and consume it.
    Utf16,
    Utf16Le,
    Utf16Be,
    Ascii,
    Latin1,
}

impl std::str::FromStr for TextFileEncoding {
    type Err = crate::TextError;

    fn from_str(name: &str) -> TextResult<Self> {
        match name.to_ascii_lowercase().replace('_', "-").as_str() {
            "utf-8" | "utf8" => Ok(Self::Utf8),
            "utf-8-sig" => Ok(Self::Utf8Sig),
            "utf-16" | "utf16" => Ok(Self::Utf16),
            "utf-16-le" | "utf-16le" => Ok(Self::Utf16Le),
            "utf-16-be" | "utf-16be" => Ok(Self::Utf16Be),
            "ascii" | "us-ascii" => Ok(Self::Ascii),
            "latin-1" | "latin1" | "iso-8859-1" => Ok(Self::Latin1),
            _ => Err(invalid(format!("unsupported text encoding: {name}"))),
        }
    }
}

/// How the reader handles malformed input bytes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TextEncodingErrors {
    #[default]
    Strict,
    Replace,
    Ignore,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TextFileOptions {
    pub sample_by: TextSampleBy,
    /// Preserve trailing newlines in line mode. Other modes retain their
    /// internal newlines independently of this flag.
    pub keep_linebreaks: bool,
    pub encoding: TextFileEncoding,
    pub encoding_errors: TextEncodingErrors,
}

#[derive(Debug, Default)]
struct TextFileRows {
    documents: Vec<String>,
    rows: Vec<(usize, Range<usize>)>,
}

/// An eager decoded text source with indexed access to its rows.
///
/// Each decoded document is stored once. Reads return owned text for the
/// requested row indices, including repeats, in caller order.
#[derive(Clone, Debug)]
pub struct TextFileDataset {
    data: Arc<TextFileRows>,
}

impl TextFileDataset {
    /// Read one document from a local file or any other byte reader.
    pub fn from_reader(reader: impl Read, options: TextFileOptions) -> TextResult<Self> {
        let mut data = TextFileRows::default();
        data.push(read_document(reader, options)?, options);
        Ok(Self {
            data: Arc::new(data),
        })
    }

    /// Read local files in the supplied order. Row boundaries never span files.
    pub fn open_files(
        paths: impl IntoIterator<Item = impl AsRef<Path>>,
        options: TextFileOptions,
    ) -> TextResult<Self> {
        let mut data = TextFileRows::default();
        for path in paths {
            let path = path.as_ref();
            let document = std::fs::File::open(path)
                .and_then(|file| read_document(file, options))
                .map_err(|error| {
                    io::Error::new(error.kind(), format!("{}: {error}", path.display()))
                })?;
            data.push(document, options);
        }
        if data.documents.is_empty() {
            return Err(invalid("at least one text file is required"));
        }
        Ok(Self {
            data: Arc::new(data),
        })
    }

    pub(crate) fn sample_at(&self, index: usize) -> TextSample {
        let (document, range) = &self.data.rows[index];
        TextSample::new(&self.data.documents[*document][range.clone()])
    }
}

impl TextFileRows {
    fn push(&mut self, text: String, options: TextFileOptions) {
        let document = self.documents.len();
        let mut start = 0;
        match options.sample_by {
            TextSampleBy::Line => {
                for line in text.split_inclusive('\n') {
                    let visible = if options.keep_linebreaks {
                        line
                    } else {
                        line.strip_suffix('\n').unwrap_or(line)
                    };
                    self.rows.push((document, start..start + visible.len()));
                    start += line.len();
                }
            }
            TextSampleBy::Paragraph => {
                for (end, _) in text.match_indices("\n\n") {
                    if end > start {
                        self.rows.push((document, start..end));
                    }
                    start = end + 2;
                }
                if start < text.len() {
                    self.rows.push((document, start..text.len()));
                }
            }
            TextSampleBy::Document => self.rows.push((document, 0..text.len())),
        }
        self.documents.push(text);
    }
}

fn read_document(mut reader: impl Read, options: TextFileOptions) -> io::Result<String> {
    let mut bytes = Vec::new();
    reader.read_to_end(&mut bytes)?;
    let errors = options.encoding_errors;
    let text = match options.encoding {
        TextFileEncoding::Utf8 => decode_utf8(bytes, errors)?,
        TextFileEncoding::Utf8Sig => {
            const BOM: &[u8] = &[0xef, 0xbb, 0xbf];
            if BOM.starts_with(&bytes) {
                String::new()
            } else {
                if bytes.starts_with(BOM) {
                    bytes.drain(..BOM.len());
                }
                decode_utf8(bytes, errors)?
            }
        }
        TextFileEncoding::Utf16 => match bytes.as_slice() {
            [0xff, 0xfe, rest @ ..] => decode_utf16(rest, u16::from_le_bytes, errors)?,
            [0xfe, 0xff, rest @ ..] => decode_utf16(rest, u16::from_be_bytes, errors)?,
            [] | [_] => decode_utf16(&bytes, u16::from_le_bytes, errors)?,
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "UTF-16 input is missing a BOM",
                ));
            }
        },
        TextFileEncoding::Utf16Le => decode_utf16(&bytes, u16::from_le_bytes, errors)?,
        TextFileEncoding::Utf16Be => decode_utf16(&bytes, u16::from_be_bytes, errors)?,
        TextFileEncoding::Ascii => {
            let mut output = String::with_capacity(bytes.len());
            for (offset, byte) in bytes.into_iter().enumerate() {
                if byte.is_ascii() {
                    output.push(char::from(byte));
                } else {
                    malformed(&mut output, errors, || {
                        format!("invalid ASCII byte at offset {offset}")
                    })?;
                }
            }
            output
        }
        TextFileEncoding::Latin1 => bytes.into_iter().map(char::from).collect(),
    };

    // Match Python's text-mode universal newlines before splitting records.
    if text.contains('\r') {
        Ok(text.replace("\r\n", "\n").replace('\r', "\n"))
    } else {
        Ok(text)
    }
}

fn decode_utf8(bytes: Vec<u8>, errors: TextEncodingErrors) -> io::Result<String> {
    match String::from_utf8(bytes) {
        Ok(text) => Ok(text),
        Err(error) => match errors {
            TextEncodingErrors::Strict => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                error.utf8_error(),
            )),
            TextEncodingErrors::Replace => {
                Ok(String::from_utf8_lossy(error.as_bytes()).into_owned())
            }
            TextEncodingErrors::Ignore => Ok(error
                .as_bytes()
                .utf8_chunks()
                .map(|chunk| chunk.valid())
                .collect()),
        },
    }
}

fn decode_utf16(
    bytes: &[u8],
    decode_unit: fn([u8; 2]) -> u16,
    errors: TextEncodingErrors,
) -> io::Result<String> {
    let (units, tail) = bytes.as_chunks::<2>();
    let mut chars = char::decode_utf16(units.iter().copied().map(decode_unit)).peekable();
    let mut output = String::with_capacity(units.len());
    let mut has_trailing_byte = !tail.is_empty();
    while let Some(ch) = chars.next() {
        match ch {
            Ok(ch) => output.push(ch),
            Err(error) => {
                let surrogate = error.unpaired_surrogate();
                if (0xd800..=0xdbff).contains(&surrogate) && chars.peek().is_none() {
                    // Python's codec groups a final high surrogate and an incomplete unit.
                    has_trailing_byte = false;
                }
                malformed(&mut output, errors, || {
                    format!("unpaired UTF-16 surrogate U+{surrogate:04X}")
                })?;
            }
        }
    }
    if has_trailing_byte {
        malformed(&mut output, errors, || "truncated UTF-16 code unit".into())?;
    }
    Ok(output)
}

fn malformed(
    output: &mut String,
    errors: TextEncodingErrors,
    message: impl FnOnce() -> String,
) -> io::Result<()> {
    match errors {
        TextEncodingErrors::Strict => {
            return Err(io::Error::new(io::ErrorKind::InvalidData, message()));
        }
        TextEncodingErrors::Replace => output.push(char::REPLACEMENT_CHARACTER),
        TextEncodingErrors::Ignore => {}
    }
    Ok(())
}

impl Dataset for TextFileDataset {
    type Item = TextSample;

    fn len(&self) -> usize {
        self.data.rows.len()
    }

    fn capabilities(&self) -> SourceCapabilities {
        SourceCapabilities {
            batched_reads: true,
            parallel_reads: true,
            ..Default::default()
        }
    }

    fn get_many(&self, indices: &[usize]) -> DataResult<Vec<TextSample>> {
        validate_indices(indices, self.len())?;
        Ok(indices.iter().map(|&index| self.sample_at(index)).collect())
    }
}
