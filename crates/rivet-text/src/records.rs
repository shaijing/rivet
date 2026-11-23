use crate::errors::{TextResult, invalid};
use crate::{TextColumns, TextFileDataset, TextFileOptions, TextSample};
use calamine::{Data, Reader, open_workbook_auto_from_rs};
use flate2::read::MultiGzDecoder;
use rivet_data::dataset::memory::MemoryDataset;
use rivet_data::dataset::source::{Dataset, SourceCapabilities};
use rivet_data::errors::DataResult;
use serde_json::Value;
use std::fs::File;
use std::io::{self, BufReader, Cursor, Read};
use std::path::Path;

#[derive(Clone, Debug, Default)]
pub struct TextRecordsOptions {
    /// Columns to project from structured records. Defaults to the `text` column.
    pub columns: TextColumns,
    /// Excel worksheet name. The first worksheet is used by default.
    pub sheet: Option<String>,
    /// Sampling and decoding settings for plain-text files and archive members.
    pub text: TextFileOptions,
}

/// Indexed text rows loaded from structured files and TAR archives.
#[derive(Debug)]
pub struct TextRecordsDataset {
    rows: MemoryDataset<TextSample>,
}

impl TextRecordsDataset {
    pub fn open(path: impl AsRef<Path>, options: TextRecordsOptions) -> TextResult<Self> {
        Self::open_files([path], options)
    }

    /// Load inputs in caller order. Archive members retain their archive order.
    pub fn open_files(
        paths: impl IntoIterator<Item = impl AsRef<Path>>,
        options: TextRecordsOptions,
    ) -> TextResult<Self> {
        let mut rows = Vec::new();
        let mut paths = paths.into_iter().peekable();
        if paths.peek().is_none() {
            return Err(invalid("at least one data file is required"));
        }
        for path in paths {
            let path = path.as_ref();
            let result = (|| {
                let format = Format::from_path(path)
                    .ok_or_else(|| invalid("unsupported data file extension"))?;
                let reader = BufReader::new(File::open(path)?);
                match format {
                    Format::Tar => read_tar(reader, &options, &mut rows),
                    Format::TarGz => {
                        let mut decoder = MultiGzDecoder::new(reader);
                        read_tar(&mut decoder, &options, &mut rows)?;
                        // TAR stops at its end marker, which can precede the gzip checksum.
                        io::copy(&mut decoder, &mut io::sink())?;
                        Ok(())
                    }
                    _ => read_records(format, reader, &options, &mut rows),
                }
            })();
            result.map_err(|error| invalid(format!("{}: {error}", path.display())))?;
        }
        Ok(Self {
            rows: MemoryDataset::new(rows),
        })
    }
}

impl Dataset for TextRecordsDataset {
    type Item = TextSample;

    fn len(&self) -> usize {
        self.rows.len()
    }

    fn capabilities(&self) -> SourceCapabilities {
        self.rows.capabilities()
    }

    fn get_many(&self, indices: &[usize]) -> DataResult<Vec<TextSample>> {
        self.rows.get_many(indices)
    }
}

#[derive(Clone, Copy)]
enum Format {
    Json,
    Csv(u8),
    Excel,
    Text,
    Tar,
    TarGz,
}

impl Format {
    fn from_path(path: &Path) -> Option<Self> {
        let name = path.file_name()?.to_str()?.to_ascii_lowercase();
        if name.ends_with(".tar.gz") || name.ends_with(".tgz") {
            return Some(Self::TarGz);
        }
        match name.rsplit_once('.')?.1 {
            "json" | "jsonl" | "ndjson" => Some(Self::Json),
            "csv" => Some(Self::Csv(b',')),
            "tsv" => Some(Self::Csv(b'\t')),
            "xls" | "xlsx" | "xlsm" | "xlsb" | "ods" => Some(Self::Excel),
            "txt" => Some(Self::Text),
            "tar" => Some(Self::Tar),
            _ => None,
        }
    }
}

fn read_tar(
    reader: impl Read,
    options: &TextRecordsOptions,
    rows: &mut Vec<TextSample>,
) -> TextResult<()> {
    let mut archive = tar::Archive::new(reader);
    let mut found = false;
    for entry in archive.entries()? {
        let entry = entry?;
        if !entry.header().entry_type().is_file() {
            continue;
        }
        let path = entry.path()?.into_owned();
        let Some(format) = Format::from_path(&path) else {
            continue;
        };
        if matches!(format, Format::Tar | Format::TarGz) {
            continue;
        }
        found = true;
        read_records(format, entry, options, rows)
            .map_err(|error| invalid(format!("archive member {}: {error}", path.display())))?;
    }
    if !found {
        return Err(invalid("archive contains no supported text or table files"));
    }
    Ok(())
}

fn read_records(
    format: Format,
    reader: impl Read,
    options: &TextRecordsOptions,
    rows: &mut Vec<TextSample>,
) -> TextResult<()> {
    match format {
        Format::Json => read_json(reader, &options.columns, rows),
        Format::Csv(delimiter) => read_csv(reader, delimiter, &options.columns, rows),
        Format::Excel => read_excel(reader, options, rows),
        Format::Text => {
            let source = TextFileDataset::from_reader(reader, options.text)?;
            rows.reserve(source.len());
            for index in 0..source.len() {
                rows.push(source.sample_at(index));
            }
            Ok(())
        }
        Format::Tar | Format::TarGz => {
            unreachable!("archive dispatch happens at the source boundary")
        }
    }
}

fn read_json(
    reader: impl Read,
    columns: &TextColumns,
    rows: &mut Vec<TextSample>,
) -> TextResult<()> {
    let mut index = 0;
    for value in serde_json::Deserializer::from_reader(reader).into_iter::<Value>() {
        let value = value.map_err(|error| invalid(format!("JSON: {error}")))?;
        match value {
            Value::Array(records) => {
                for record in records {
                    index += 1;
                    rows.push(json_record(&record, columns, index)?);
                }
            }
            record => {
                index += 1;
                rows.push(json_record(&record, columns, index)?);
            }
        }
    }
    Ok(())
}

fn json_record(record: &Value, columns: &TextColumns, row: usize) -> TextResult<TextSample> {
    let record = record
        .as_object()
        .ok_or_else(|| invalid(format!("JSON record {row} must be an object")))?;
    let text = |name: &str| {
        record
            .get(name)
            .and_then(Value::as_str)
            .map(str::to_owned)
            .ok_or_else(|| cell_error(row, name, "a string"))
    };
    Ok(TextSample {
        text: text(&columns.text)?,
        text_pair: columns.text_pair.as_deref().map(text).transpose()?,
        label: columns
            .label
            .as_deref()
            .map(|name| {
                record
                    .get(name)
                    .and_then(Value::as_i64)
                    .ok_or_else(|| cell_error(row, name, "an int64"))
            })
            .transpose()?,
    })
}

struct ColumnIndices<'a> {
    text: (usize, &'a str),
    text_pair: Option<(usize, &'a str)>,
    label: Option<(usize, &'a str)>,
}

impl<'a> ColumnIndices<'a> {
    fn resolve(headers: &[&str], columns: &'a TextColumns) -> TextResult<Self> {
        let find = |name: &'a str| {
            let mut matches = headers
                .iter()
                .enumerate()
                .filter(|(_, header)| **header == name);
            let (index, _) = matches
                .next()
                .ok_or_else(|| invalid(format!("missing column {name:?}")))?;
            if matches.next().is_some() {
                return Err(invalid(format!("duplicate column {name:?}")));
            }
            Ok((index, name))
        };
        Ok(Self {
            text: find(&columns.text)?,
            text_pair: columns.text_pair.as_deref().map(find).transpose()?,
            label: columns.label.as_deref().map(find).transpose()?,
        })
    }
}

fn read_csv(
    reader: impl Read,
    delimiter: u8,
    columns: &TextColumns,
    rows: &mut Vec<TextSample>,
) -> TextResult<()> {
    let mut csv = csv::ReaderBuilder::new()
        .delimiter(delimiter)
        .from_reader(reader);
    let headers = csv
        .headers()
        .map_err(|error| invalid(format!("CSV/TSV header: {error}")))?;
    let indices = ColumnIndices::resolve(&headers.iter().collect::<Vec<_>>(), columns)?;
    for (index, result) in csv.records().enumerate() {
        let record = result.map_err(|error| invalid(format!("CSV/TSV: {error}")))?;
        let row = index + 2;
        rows.push(TextSample {
            text: record[indices.text.0].to_owned(),
            text_pair: indices.text_pair.map(|(i, _)| record[i].to_owned()),
            label: indices
                .label
                .map(|(i, name)| {
                    record[i]
                        .parse()
                        .map_err(|_| cell_error(row, name, "an int64"))
                })
                .transpose()?,
        });
    }
    Ok(())
}

fn read_excel(
    mut reader: impl Read,
    options: &TextRecordsOptions,
    rows: &mut Vec<TextSample>,
) -> TextResult<()> {
    let mut bytes = Vec::new();
    reader.read_to_end(&mut bytes)?;
    let mut workbook = open_workbook_auto_from_rs(Cursor::new(bytes))
        .map_err(|error| invalid(format!("Excel workbook: {error}")))?;
    let sheet = options
        .sheet
        .clone()
        .or_else(|| workbook.sheet_names().into_iter().next())
        .ok_or_else(|| invalid("workbook has no worksheets"))?;
    let range = workbook
        .worksheet_range(&sheet)
        .map_err(|error| invalid(format!("worksheet {sheet:?}: {error}")))?;
    let mut records = range.rows();
    let header = records
        .next()
        .ok_or_else(|| invalid(format!("worksheet {sheet:?} has no header row")))?;
    let headers = header
        .iter()
        .map(|cell| match cell {
            Data::String(name) => name.as_str(),
            _ => "",
        })
        .collect::<Vec<_>>();
    let columns = &options.columns;
    let indices = ColumnIndices::resolve(&headers, columns)?;
    let first_row = range.start().map_or(0, |(row, _)| row as usize);
    for (index, record) in records.enumerate() {
        if record.iter().all(|cell| matches!(cell, Data::Empty)) {
            continue;
        }
        let row = first_row + index + 2;
        let text = |i: usize, name: &str| match &record[i] {
            Data::String(value) => Ok(value.clone()),
            Data::Empty => Ok(String::new()),
            _ => Err(cell_error(row, name, "text")),
        };
        let label = indices
            .label
            .map(|(i, name)| {
                let error = || cell_error(row, name, "an int64");
                match &record[i] {
                    Data::Int(value) => Ok(*value),
                    Data::Float(value)
                        if value.fract() == 0.0
                            && *value >= i64::MIN as f64
                            && *value < -(i64::MIN as f64) =>
                    {
                        Ok(*value as i64)
                    }
                    Data::String(value) => value.parse().map_err(|_| error()),
                    _ => Err(error()),
                }
            })
            .transpose()?;
        rows.push(TextSample {
            text: text(indices.text.0, indices.text.1)?,
            text_pair: indices
                .text_pair
                .map(|(i, name)| text(i, name))
                .transpose()?,
            label,
        });
    }
    Ok(())
}

fn cell_error(row: usize, name: &str, expected: &str) -> crate::TextError {
    invalid(format!("row {row}, column {name:?}: expected {expected}"))
}
