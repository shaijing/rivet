use std::collections::HashSet;
use std::fs::{self, File};
use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

use percent_encoding::percent_decode_str;
use reqwest::blocking::{Client, RequestBuilder, Response};
use serde::Deserialize;
use url::Url;
use walkdir::WalkDir;

use super::{archive, parse_url};
use crate::errors::{DataResult, invalid_argument};

pub(super) fn git_url(location: &str) -> DataResult<String> {
    let location = if location.contains("://") {
        location.to_owned()
    } else {
        let valid = location.split_once('/').is_some_and(|(owner, repo)| {
            [owner, repo].iter().all(|part| {
                !matches!(*part, "" | "." | "..") && !part.contains(['/', '\\', '?', '#'])
            })
        });
        if !valid {
            return Err(invalid_argument(
                "Git source must be owner/repo or a Git URL",
            ));
        }
        format!("https://github.com/{location}")
    };
    let url = parse_url(&location)?;
    match url.scheme() {
        "file" if url.host().is_none() && url.path().starts_with('/') => Ok(location),
        "https" | "http" | "ssh" | "git" if url.host().is_some() => Ok(location),
        _ => Err(invalid_argument("unsupported Git URL")),
    }
}

pub(super) fn git(url: &str, destination: &Path, revision: Option<&str>) -> DataResult<()> {
    let run = |command: &mut Command| -> DataResult<()> {
        let status = command
            .env(
                "GIT_ALLOW_PROTOCOL",
                if url.starts_with("file:") {
                    "file"
                } else {
                    "http:https:ssh:git"
                },
            )
            .status()?;
        if !status.success() {
            return Err(io::Error::other(format!("git exited with {status}")).into());
        }
        Ok(())
    };
    let checkout = || {
        let mut command = Command::new("git");
        command.arg("-C").arg(destination);
        command
    };
    let head = if let Some(revision) = revision {
        run(Command::new("git").arg("init").arg(destination))?;
        run(checkout().args(["remote", "add", "origin", url]))?;
        run(checkout().args(["fetch", "--depth=1", "--", "origin", revision]))?;
        "FETCH_HEAD"
    } else {
        run(Command::new("git")
            .args(["clone", "--depth=1", "--no-checkout", "--", url])
            .arg(destination))?;
        "HEAD"
    };
    run(checkout().args(["checkout", "--detach", head]))?;
    fs::remove_dir_all(destination.join(".git"))?;
    let root = destination.canonicalize()?;
    for entry in WalkDir::new(destination) {
        let entry = entry.map_err(io::Error::from)?;
        if entry.file_type().is_symlink()
            && !entry
                .path()
                .canonicalize()
                .is_ok_and(|path| path.starts_with(&root))
        {
            return Err(invalid_argument(format!(
                "unsafe Git symbolic link: {}",
                entry.path().display()
            )));
        }
    }
    Ok(())
}

fn client() -> DataResult<Client> {
    Client::builder()
        .timeout(Duration::from_secs(60))
        .connect_timeout(Duration::from_secs(60))
        .user_agent(concat!("rivet/", env!("CARGO_PKG_VERSION")))
        .build()
        .map_err(|err| io::Error::other(err.without_url()).into())
}

fn send(request: RequestBuilder) -> DataResult<Response> {
    request
        .send()
        .and_then(Response::error_for_status)
        .map_err(|err| io::Error::other(err.without_url()).into())
}

fn save(mut response: Response, path: &Path) -> DataResult<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    io::copy(&mut response, &mut File::create_new(path)?)?;
    Ok(())
}

pub(super) fn http(source: &str, destination: &Path, extract: bool) -> DataResult<PathBuf> {
    let url = parse_url(source)?;
    let segment = url.path().rsplit('/').next().unwrap_or_default();
    let decoded = percent_decode_str(segment).decode_utf8_lossy();
    let name = decoded
        .rsplit('/')
        .find(|part| !part.is_empty())
        .unwrap_or_default();
    let name = if archive::safe_component(name) {
        name
    } else {
        "download"
    };
    let artifact = destination.join(name);
    save(send(client()?.get(url))?, &artifact)?;

    if extract {
        let unpacked = destination.with_file_name("unpacked");
        if archive::unpack(&artifact, &unpacked)? {
            fs::remove_dir_all(destination)?;
            fs::rename(unpacked, destination)?;
            return Ok(destination.to_owned());
        }
    }
    Ok(artifact)
}

#[derive(Deserialize)]
struct RepoInfo {
    sha: String,
}

#[derive(Deserialize)]
struct TreeEntry {
    #[serde(rename = "type")]
    kind: String,
    path: String,
}

fn hf_token() -> Option<String> {
    if std::env::var("HF_HUB_DISABLE_IMPLICIT_TOKEN").is_ok_and(|value| {
        matches!(
            value.to_ascii_uppercase().as_str(),
            "1" | "ON" | "YES" | "TRUE"
        )
    }) {
        return None;
    }
    let clean = |value: String| {
        let value = value.replace(['\r', '\n'], "").trim().to_owned();
        (!value.is_empty()).then_some(value)
    };
    ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"]
        .into_iter()
        .find_map(|key| std::env::var(key).ok().and_then(clean))
        .or_else(|| {
            let token = std::env::var_os("HF_TOKEN_PATH")
                .map(PathBuf::from)
                .or_else(|| {
                    let home = std::env::var_os("HF_HOME").map(PathBuf::from).or_else(|| {
                        std::env::var_os("XDG_CACHE_HOME")
                            .map(PathBuf::from)
                            .or_else(|| dirs::home_dir().map(|path| path.join(".cache")))
                            .map(|path| path.join("huggingface"))
                    })?;
                    Some(home.join("token"))
                })?;
            fs::read_to_string(token).ok().and_then(clean)
        })
}

fn at(endpoint: &Url, parts: &[&str]) -> Url {
    let mut url = endpoint.clone();
    url.path_segments_mut()
        .unwrap()
        .pop_if_empty()
        .extend(parts.iter().copied());
    url
}

pub(super) fn hugging_face(
    repo: &str,
    revision: Option<&str>,
    destination: &Path,
) -> DataResult<()> {
    let repo: Vec<_> = repo.split('/').collect();
    if repo.len() > 2
        || !repo.iter().all(|part| {
            !matches!(*part, "" | "." | "..")
                && part
                    .bytes()
                    .all(|c| c.is_ascii_alphanumeric() || matches!(c, b'.' | b'-' | b'_'))
        })
    {
        return Err(invalid_argument(
            "invalid Hugging Face dataset repository ID",
        ));
    }
    if std::env::var("HF_HUB_OFFLINE").is_ok_and(|value| {
        matches!(
            value.to_ascii_uppercase().as_str(),
            "1" | "ON" | "YES" | "TRUE"
        )
    }) {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            "Hugging Face Hub is offline and the dataset is absent from the Rivet cache",
        )
        .into());
    }
    let endpoint = parse_url(
        &std::env::var("HF_ENDPOINT").unwrap_or_else(|_| "https://huggingface.co".into()),
    )?;
    if !matches!(endpoint.scheme(), "http" | "https") || endpoint.host().is_none() {
        return Err(invalid_argument("invalid Hugging Face endpoint"));
    }
    let client = client()?;
    let token = hf_token();
    let get = |url: Url| {
        let request = client.get(url);
        match &token {
            Some(token) => request.bearer_auth(token),
            None => request,
        }
    };
    let mut api = vec!["api", "datasets"];
    api.extend(&repo);
    let revision = revision.unwrap_or("main");
    let mut info = api.clone();
    info.extend(["revision", revision]);
    let info: RepoInfo = send(get(at(&endpoint, &info)))?
        .json()
        .map_err(io::Error::other)?;
    if info.sha.len() != 40 || !info.sha.bytes().all(|c| c.is_ascii_hexdigit()) {
        return Err(io::Error::other("Hugging Face returned an invalid commit hash").into());
    }

    let mut resolve = vec!["datasets"];
    resolve.extend(&repo);
    resolve.extend(["resolve", &info.sha]);
    api.extend(["tree", &info.sha]);
    let mut page = at(&endpoint, &api);
    page.query_pairs_mut()
        .append_pair("recursive", "true")
        .append_pair("expand", "false");
    fs::create_dir(destination)?;
    let mut visited = HashSet::new();
    loop {
        if !visited.insert(page.clone()) {
            return Err(io::Error::other("Hugging Face returned a repeated tree page").into());
        }
        let response = send(get(page))?;
        let next = response
            .headers()
            .get_all(reqwest::header::LINK)
            .iter()
            .filter_map(|value| value.to_str().ok())
            .flat_map(|value| value.split(','))
            .find_map(|value| {
                let (target, params) = value.trim().split_once('>')?;
                params
                    .split(';')
                    .any(|part| matches!(part.trim(), "rel=\"next\"" | "rel=next"))
                    .then(|| target.strip_prefix('<'))
                    .flatten()
            })
            .map(|target| response.url().join(target))
            .transpose()
            .map_err(invalid_argument)?;
        let entries: Vec<TreeEntry> = response.json().map_err(io::Error::other)?;
        for entry in entries.into_iter().filter(|entry| entry.kind == "file") {
            let parts: Vec<_> = entry.path.split('/').collect();
            if !parts.iter().all(|part| archive::safe_component(part)) {
                return Err(invalid_argument(format!(
                    "unsafe Hugging Face file path: {:?}",
                    entry.path
                )));
            }
            let mut file = resolve.clone();
            file.extend(parts);
            save(
                send(get(at(&endpoint, &file)))?,
                &destination.join(entry.path),
            )?;
        }
        let Some(next) = next else { break };
        if endpoint.origin() != next.origin() {
            return Err(io::Error::other("Hugging Face tree pagination changed origin").into());
        }
        page = next;
    }
    Ok(())
}
