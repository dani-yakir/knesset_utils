"""Fetch the current SQLite mirror from a GitHub Release and swap it into place.

The rebuild job (`.github/workflows/regenerate.yml`) publishes each finished mirror as a
zstd-compressed asset on a moving `latest` release. The HTTP server calls
`ensure_mirror()` on boot and, optionally, from a background thread, to keep its
local copy current.

Design points:
- A public mirror repo needs no credentials; a private one uses `cfg.github_token`.
- Freshness is tracked by a small marker file next to the DB (`<db>.release`)
  holding `<release id>:<asset id>:<asset updated_at>`. If it matches the live
  release, nothing is downloaded.
- The new file is written to `<db>.part` in the *same directory* and moved onto
  `db_path` with `os.replace()` (atomic on one filesystem). `server/tools.py`
  opens a fresh read-only connection per call, so a swap between calls is safe.
- Only one DB is ever kept. Temp files orphaned by a killed download are swept
  before the next one starts, so they cannot pile up on a persistent disk.
- A transient failure never deletes or truncates a working local DB.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path

import httpx

from knesset_utils.server.config import ServerConfig

log = logging.getLogger("knesset_utils.mirror")

_API = "https://api.github.com"
_CHUNK = 1 << 20


def _headers(cfg: ServerConfig, *, octet: bool = False) -> dict[str, str]:
    headers = {
        "Accept": "application/octet-stream" if octet else "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if cfg.github_token:
        headers["Authorization"] = f"Bearer {cfg.github_token}"
    return headers


def _get_release(cfg: ServerConfig) -> dict:
    url = f"{_API}/repos/{cfg.mirror_repo}/releases/tags/{cfg.mirror_release_tag}"
    resp = httpx.get(url, headers=_headers(cfg), timeout=30.0, follow_redirects=True)
    resp.raise_for_status()
    return resp.json()


def _pick_asset(release: dict, name: str) -> dict:
    for asset in release.get("assets", []):
        if asset.get("name") == name:
            return asset
    raise RuntimeError(f"release {release.get('tag_name')!r} has no asset {name!r}")


def _marker_path(db_path: Path) -> Path:
    return db_path.with_name(db_path.name + ".release")


def _marker(release: dict, asset: dict) -> str:
    return f"{release.get('id')}:{asset.get('id')}:{asset.get('updated_at')}"


def _part_path(db_path: Path) -> Path:
    return db_path.with_name(db_path.name + ".part")


def _sweep_leftovers(db_path: Path) -> None:
    """Delete temp files left behind by a download that was killed mid-flight.

    A SIGKILL (redeploy, OOM) skips the `finally` in `_download_and_swap`, and on
    a persistent disk a DB-sized orphan then eats the headroom every later swap
    needs. Only one download runs at a time, so anything matching is stale.
    Also covers the randomly named `tmp*.zst` / `tmp*.sqlite.part` files that
    earlier versions created.
    """
    part = _part_path(db_path)
    for path in db_path.parent.iterdir():
        legacy = path.name.startswith("tmp") and path.name.endswith((".zst", ".sqlite.part"))
        if path == part or legacy:
            try:
                size = path.stat().st_size
                path.unlink()
                log.warning("removed leftover temp file %s (%d bytes)", path, size)
            except OSError:
                log.exception("could not remove leftover temp file %s", path)


def _decompressor(window_log: int):
    import zstandard

    if window_log:
        return zstandard.ZstdDecompressor(max_window_size=1 << window_log)
    return zstandard.ZstdDecompressor()


def _download_and_swap(cfg: ServerConfig, release: dict, asset: dict) -> None:
    db_path = cfg.db_path
    db_path.parent.mkdir(parents=True, exist_ok=True)
    _sweep_leftovers(db_path)

    if cfg.github_token:
        url = f"{_API}/repos/{cfg.mirror_repo}/releases/assets/{asset['id']}"
        dl_headers = _headers(cfg, octet=True)
    else:
        url = asset["browser_download_url"]
        dl_headers = {}

    # Decompress straight off the socket: the compressed asset never touches the
    # disk, so the peak is the old DB plus the new one.
    part = _part_path(db_path)
    dobj = _decompressor(cfg.zstd_long_window_log).decompressobj()
    try:
        with httpx.stream("GET", url, headers=dl_headers, timeout=None, follow_redirects=True) as resp:
            resp.raise_for_status()
            with open(part, "wb") as f:
                for chunk in resp.iter_bytes(_CHUNK):
                    f.write(dobj.decompress(chunk))
        if not dobj.eof:
            raise RuntimeError(f"asset {asset.get('name')!r} ended before the zstd frame did")
        os.replace(part, db_path)  # atomic on the same filesystem
        _marker_path(db_path).write_text(_marker(release, asset))
    finally:
        try:
            os.unlink(part)
        except OSError:
            pass


def ensure_mirror(cfg: ServerConfig) -> None:
    """Download the mirror if the local copy is missing or older than the release.

    No-op when `cfg.mirror_repo` is unset. Never raises if a usable DB already
    exists locally -- it just logs and keeps serving the old data.
    """
    if not cfg.mirror_repo:
        return
    try:
        release = _get_release(cfg)
        asset = _pick_asset(release, cfg.mirror_asset)
        want = _marker(release, asset)
        marker_file = _marker_path(cfg.db_path)
        have = marker_file.read_text().strip() if marker_file.exists() else None
        if cfg.db_path.exists() and have == want:
            log.info("mirror up to date (%s)", want)
            return
        log.info("fetching mirror %s -> %s", want, cfg.db_path)
        _download_and_swap(cfg, release, asset)
        log.info("mirror ready: %s (%d bytes)", cfg.db_path, cfg.db_path.stat().st_size)
    except Exception:
        if cfg.db_path.exists():
            log.exception("mirror refresh failed; keeping existing DB")
            return
        raise


def start_refresh_thread(cfg: ServerConfig) -> threading.Thread:
    """Spawn a daemon thread that re-checks the release every `cfg.refresh_interval` seconds."""
    interval = max(cfg.refresh_interval, 60)

    def _loop() -> None:
        while True:
            time.sleep(interval)
            try:
                ensure_mirror(cfg)
            except Exception:
                log.exception("background mirror refresh crashed; will retry next tick")

    thread = threading.Thread(target=_loop, name="mirror-refresh", daemon=True)
    thread.start()
    return thread
