use std::fs::{self, File};
use std::io::{self, Read, Seek};
use std::path::{Path, PathBuf};

use crate::errors::{DataResult, invalid_argument};

pub(super) fn safe_component(name: &str) -> bool {
    if matches!(name, "" | "." | "..")
        || name.contains(['/', '\\', ':', '\0'])
        || name.ends_with([' ', '.'])
    {
        return false;
    }
    let stem = name
        .split('.')
        .next()
        .unwrap_or_default()
        .trim_end()
        .to_ascii_uppercase();
    !matches!(
        stem.as_str(),
        "CON" | "PRN" | "AUX" | "NUL" | "CONIN$" | "CONOUT$"
    ) && !stem
        .strip_prefix("COM")
        .or_else(|| stem.strip_prefix("LPT"))
        .is_some_and(|suffix| {
            matches!(
                suffix,
                "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9" | "¹" | "²" | "³"
            )
        })
}

fn target(root: &Path, name: &str, format: &str) -> DataResult<PathBuf> {
    let mut path = root.to_owned();
    let name = name.trim_end_matches('/');
    for part in name.split('/') {
        if part == "." {
            continue;
        }
        if !safe_component(part) {
            return Err(invalid_argument(format!(
                "unsafe {format} member path: {name:?}"
            )));
        }
        path.push(part);
    }
    Ok(path)
}

pub(super) fn unpack(artifact: &Path, root: &Path) -> DataResult<bool> {
    let mut input = File::open(artifact)?;
    if let Ok(mut zip) = zip::ZipArchive::new(&mut input) {
        fs::create_dir(root)?;
        for index in 0..zip.len() {
            let mut member = zip.by_index(index).map_err(io::Error::other)?;
            let name = member.name().map_err(io::Error::other)?;
            let path = target(root, &name, "ZIP")?;
            if member.is_symlink() {
                return Err(invalid_argument(format!(
                    "ZIP member is a symbolic link: {:?}",
                    name
                )));
            }
            if member.is_dir() {
                fs::create_dir_all(path)?;
            } else {
                fs::create_dir_all(path.parent().unwrap())?;
                io::copy(&mut member, &mut File::create_new(path)?)?;
            }
        }
        return Ok(true);
    }

    input.rewind()?;
    let mut magic = [0; 6];
    let count = input.read(&mut magic)?;
    input.rewind()?;
    let mut input: Box<dyn Read> = match &magic[..count] {
        magic if magic.starts_with(&[0x1f, 0x8b]) => {
            Box::new(flate2::read::MultiGzDecoder::new(input))
        }
        magic if magic.starts_with(b"BZh") => Box::new(bzip2::read::MultiBzDecoder::new(input)),
        [0xfd, b'7', b'z', b'X', b'Z', 0] => Box::new(lzma_rust2::XzReader::new(input, true)),
        _ => Box::new(input),
    };
    let mut header = Vec::with_capacity(512);
    if input.by_ref().take(512).read_to_end(&mut header).is_err() || header.len() < 512 {
        return Ok(false);
    }
    let looks_like_tar = header[257..262] == *b"ustar";
    if header.iter().all(|&byte| byte == 0) {
        let mut end = [0; 512];
        if input.read_exact(&mut end).is_ok() && end.iter().all(|&byte| byte == 0) {
            fs::create_dir(root)?;
            return Ok(true);
        }
        return Ok(false);
    }
    let mut tar = tar::Archive::new(header.as_slice().chain(input));
    let mut entries = tar.entries()?;
    let first = match entries.next() {
        Some(Ok(entry)) => entry,
        Some(Err(err)) if looks_like_tar => return Err(err.into()),
        _ => return Ok(false),
    };
    fs::create_dir(root)?;
    for member in std::iter::once(Ok(first)).chain(entries) {
        let mut member = member?;
        let name = member.path()?;
        let name = name
            .to_str()
            .ok_or_else(|| invalid_argument("TAR member path is not UTF-8"))?;
        let path = target(root, name, "TAR")?;
        let kind = member.header().entry_type();
        if kind.is_dir() {
            fs::create_dir_all(path)?;
        } else if kind.is_file() {
            fs::create_dir_all(path.parent().unwrap())?;
            io::copy(&mut member, &mut File::create(path)?)?;
        } else {
            return Err(invalid_argument(format!(
                "unsupported TAR member: {name:?}"
            )));
        }
    }
    Ok(true)
}

/// Expand a compressed single file into the preparation workspace.
pub(super) fn decompress_file(source: &Path, root: &Path) -> DataResult<Option<PathBuf>> {
    let Some(extension) = source.extension().and_then(|ext| ext.to_str()) else {
        return Ok(None);
    };
    let destination = root.join(source.file_stem().unwrap_or_default());
    if !super::tabular::is_tabular_file(&destination) {
        return Ok(None);
    }
    let mut reader: Box<dyn Read> = match extension.to_ascii_lowercase().as_str() {
        "gz" => Box::new(flate2::read::MultiGzDecoder::new(File::open(source)?)),
        "bz2" => Box::new(bzip2::read::MultiBzDecoder::new(File::open(source)?)),
        "xz" => Box::new(lzma_rust2::XzReader::new(File::open(source)?, true)),
        _ => return Ok(None),
    };
    fs::create_dir(root)?;
    io::copy(&mut reader, &mut File::create_new(&destination)?)?;
    Ok(Some(destination))
}
