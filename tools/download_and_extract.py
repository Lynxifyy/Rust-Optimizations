#!/usr/bin/env python3
"""Download Minecraft Bedrock APKs and extract lib/arm64-v8a/libminecraftpe.so.

mcpedl.org does not serve the APK from the page URL directly. The chain is:

    GET  /minecraft-pe-<ver>-apk/          -> HTML with a hidden `file_id`
    POST /show_file.php  file_id=<id>      -> HTML with `DOWNLOAD_URL = "<apk>"`
    GET  <apk>                             -> application/vnd.android.package-archive

The APKs are ~380 MB each and the `.so` inside is stored deflated, so this script
downloads the whole APK to disk and then extracts just the one member with
`zipfile` (no need to unpack the rest). Use `--keep-apk` to keep the archive; by
default the APK is deleted once the `.so` is out, because three APKs plus three
`.so` files is ~2.3 GB.

Usage:
    python3 tools/download_and_extract.py --versions 26.52 26.60 26.62 --out work
    python3 tools/download_and_extract.py --resolve-only 26.62   # just print URLs
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

MCPEDL_BASE = "https://mcpedl.org"
SO_MEMBER = "lib/arm64-v8a/libminecraftpe.so"
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

# The versions this project targets, and the mcpedl page slug for each. The
# `file_id` is resolved from the page at run time, not hardcoded, because mcpedl
# changes it when a build is re-uploaded.
DEFAULT_VERSIONS = ("26.52", "26.60", "26.62")


class DownloadError(RuntimeError):
    """A step of the download chain failed in a way the caller should see."""


def version_to_slug(version: str) -> str:
    """`26.60` -> `minecraft-pe-26-60-apk`."""
    return "minecraft-pe-" + version.replace(".", "-") + "-apk"


def version_to_key(version: str) -> str:
    """`26.60` -> `26_60` (a filename-safe key)."""
    return version.replace(".", "_")


def _get(url: str, timeout: int = 60) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def _post(url: str, data: dict[str, str], timeout: int = 60) -> bytes:
    encoded = urllib.parse.urlencode(data).encode()
    request = urllib.request.Request(url, data=encoded, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def find_file_id(page_html: str) -> str:
    """Pull the hidden `file_id` out of a version page's download form."""
    match = re.search(r'name="file_id"\s+value="(\d+)"', page_html)
    if not match:
        raise DownloadError("no file_id found on the page")
    return match.group(1)


def find_download_url(show_file_html: str) -> str:
    """Pull `DOWNLOAD_URL = "..."` out of the show_file.php response."""
    match = re.search(r'DOWNLOAD_URL\s*=\s*"(https?://[^"]+)"', show_file_html)
    if not match:
        raise DownloadError("no DOWNLOAD_URL found in the show_file response")
    return match.group(1)


def resolve_apk_url(version: str, timeout: int = 60) -> tuple[str, str]:
    """Resolve a version to `(file_id, apk_url)` through the two-step chain."""
    slug = version_to_slug(version)
    page = _get(f"{MCPEDL_BASE}/{slug}/", timeout=timeout).decode("utf-8", "replace")
    file_id = find_file_id(page)
    show = _post(
        f"{MCPEDL_BASE}/show_file.php", {"file_id": file_id}, timeout=timeout
    ).decode("utf-8", "replace")
    apk_url = find_download_url(show)
    return file_id, apk_url


def download_file(url: str, dest: Path, *, timeout: int = 120) -> tuple[int, str]:
    """Streams `url` to `dest`. Returns `(bytes, sha256)`."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    total = 0
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response, dest.open(
        "wb"
    ) as out:
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
            digest.update(chunk)
            total += len(chunk)
    return total, digest.hexdigest()


def extract_so(apk_path: Path, dest: Path) -> int:
    """Extracts `lib/arm64-v8a/libminecraftpe.so` from `apk_path` to `dest`.

    Returns the extracted size in bytes. The APK is a ZIP; the `.so` is stored
    deflated, so `ZipFile.open` inflates it as it streams.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(apk_path) as archive:
        try:
            member = archive.getinfo(SO_MEMBER)
        except KeyError as exc:
            names = [n for n in archive.namelist() if n.endswith("libminecraftpe.so")]
            raise DownloadError(
                f"{SO_MEMBER} not in {apk_path.name}; found instead: {names}"
            ) from exc
        with archive.open(member) as src, dest.open("wb") as out:
            shutil.copyfileobj(src, out, length=1 << 20)
    return dest.stat().st_size


@dataclass
class ExtractedVersion:
    version: str
    file_id: str
    apk_url: str
    apk_bytes: int
    apk_sha256: str
    so_path: str
    so_bytes: int


def process_version(
    version: str,
    out_dir: Path,
    *,
    keep_apk: bool,
    resolve_only: bool,
) -> ExtractedVersion | None:
    print(f"[{version}] resolving download chain...", flush=True)
    file_id, apk_url = resolve_apk_url(version)
    print(f"[{version}] file_id={file_id} url={apk_url}", flush=True)
    if resolve_only:
        return None

    apk_path = out_dir / "apk" / f"minecraft-{version_to_key(version)}.apk"
    so_path = out_dir / "so" / f"libminecraftpe-{version_to_key(version)}.so"

    if apk_path.exists():
        print(f"[{version}] reusing {apk_path}", flush=True)
        apk_bytes = apk_path.stat().st_size
        digest = hashlib.sha256()
        with apk_path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        apk_sha256 = digest.hexdigest()
    else:
        print(f"[{version}] downloading {apk_url} ...", flush=True)
        apk_bytes, apk_sha256 = download_file(apk_url, apk_path)
        print(f"[{version}] downloaded {apk_bytes} bytes", flush=True)

    print(f"[{version}] extracting {SO_MEMBER} ...", flush=True)
    so_bytes = extract_so(apk_path, so_path)
    print(f"[{version}] extracted {so_bytes} bytes -> {so_path}", flush=True)

    if not keep_apk:
        apk_path.unlink(missing_ok=True)

    return ExtractedVersion(
        version=version,
        file_id=file_id,
        apk_url=apk_url,
        apk_bytes=apk_bytes,
        apk_sha256=apk_sha256,
        so_path=str(so_path),
        so_bytes=so_bytes,
    )


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--versions",
        nargs="+",
        default=list(DEFAULT_VERSIONS),
        help="versions to fetch (default: 26.52 26.60 26.62)",
    )
    parser.add_argument("--out", default="work", help="output directory")
    parser.add_argument(
        "--keep-apk", action="store_true", help="keep the downloaded APK archives"
    )
    parser.add_argument(
        "--resolve-only",
        action="store_true",
        help="resolve and print the APK URLs without downloading",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[ExtractedVersion] = []
    failures: list[str] = []
    for version in args.versions:
        try:
            result = process_version(
                version,
                out_dir,
                keep_apk=args.keep_apk,
                resolve_only=args.resolve_only,
            )
            if result is not None:
                results.append(result)
        except (DownloadError, urllib.error.URLError, OSError, zipfile.BadZipFile) as exc:
            print(f"[{version}] FAILED: {exc}", file=sys.stderr, flush=True)
            failures.append(version)

    if results:
        manifest = out_dir / "downloads.json"
        manifest.write_text(json.dumps([asdict(r) for r in results], indent=2) + "\n")
        print(f"wrote {manifest}")

    if failures:
        print(f"failed versions: {failures}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
