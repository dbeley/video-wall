#!/usr/bin/env python3
"""video-wall — Single-window video wall with pygame viewer + audio."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import glob
import json
import math
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

# Suppress pygame compile-time warnings
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
warnings.filterwarnings("ignore", message=".*avx2.*", category=RuntimeWarning)

import pygame  # noqa: E402 — late import after env setup

MIN_FILE_SIZE = 50 * 1024  # 50 KB — skip torrent stubs / incomplete dls
DEFAULT_EXTS = (".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm")
MAX_TILES = 16
TITLE = "Video Wall"
VIDEO_FPS = 30
# Cap on frames discarded at once when catching up after a stall.
MAX_CATCHUP_FRAMES = 60
# Max time to wait for a tile's audio to start playing before showing its video
# anyway (keeps a broken/muted tile from staying black forever).
AUDIO_START_TIMEOUT = 3.0
# ffplay status line, e.g. "  12.34 M-A:  0.000 fd=...". The first number is
# the master (audio) clock in seconds; `nan` lines don't match.
_AUDIO_POS_RE = re.compile(rb"^\s*([-\d.]+)\s+M-A:")


def _pipe_max_size() -> int:
    try:
        with open("/proc/sys/fs/pipe-max-size") as fh:
            return int(fh.read().strip())
    except (OSError, ValueError):
        return 1 << 20


# -------- Models --------


@dataclass(frozen=True)
class VideoMeta:
    duration: float | None
    has_audio: bool
    has_video: bool
    video_codec: str | None


@dataclass
class HwAccelConfig:
    mode: str | None = None
    global_opts: dict[str, str] = field(default_factory=dict)

    @property
    def enabled(self) -> bool:
        return self.mode is not None


@dataclass
class PipelineConfig:
    tiles: list["TileState"]
    rows: int
    cols: int
    cell_w: int
    cell_h: int
    loop: bool
    volume: float
    hwaccel: HwAccelConfig
    verbose: bool
    no_audio: bool
    audio_mode: str
    audio_tile: int | None
    audio_rate: int
    border: int
    fullscreen: bool


@dataclass
class TileState:
    path: Path
    seek: float

    @property
    def metadata(self) -> VideoMeta:
        return get_metadata(self.path)


# -------- Caching & ffprobe --------

_metadata_cache: dict[Path, VideoMeta] = {}


def need(binname: str) -> None:
    if not shutil.which(binname):
        raise SystemExit(f"Missing '{binname}' in PATH.")


def run_cmd(argv: list[str]) -> str:
    try:
        return subprocess.check_output(argv, stderr=subprocess.STDOUT).decode(
            "utf-8", "ignore"
        )
    except subprocess.CalledProcessError as exc:
        print(
            f"[warn] command {' '.join(argv)} failed with code {exc.returncode}",
            file=sys.stderr,
        )
        return ""
    except Exception as exc:
        print(f"[warn] command {' '.join(argv)} failed: {exc}", file=sys.stderr)
        return ""


def get_metadata(path: Path) -> VideoMeta:
    meta = _metadata_cache.get(path)
    if meta is not None:
        return meta

    out = run_cmd(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=index,codec_type,codec_name",
            "-of",
            "json",
            str(path),
        ]
    )

    duration: float | None = None
    has_audio = False
    has_video = False
    vcodec: str | None = None
    if out:
        try:
            data = json.loads(out)
            fmt = data.get("format") or {}
            dur_str = fmt.get("duration")
            if dur_str:
                duration = float(dur_str)
            for s in data.get("streams") or []:
                ctype = (s.get("codec_type") or "").lower()
                if ctype == "audio":
                    has_audio = True
                if ctype == "video":
                    has_video = True
                    if vcodec is None:
                        name = s.get("codec_name")
                        if isinstance(name, str) and name:
                            vcodec = name.lower()
        except Exception as exc:
            print(
                f"[warn] failed to parse ffprobe output for {path}: {exc}",
                file=sys.stderr,
            )

    meta = VideoMeta(
        duration=duration,
        has_audio=has_audio,
        has_video=has_video,
        video_codec=vcodec,
    )
    _metadata_cache[path] = meta
    return meta


def is_valid_video(path: Path) -> bool:
    """Cheap cached validity: returns True if ffprobe sees a video stream
    AND the file is large enough (filters torrent stubs)."""
    try:
        if not get_metadata(path).has_video:
            return False
        # Skip suspiciously small files (incomplete torrents, stubs)
        try:
            if path.stat().st_size < MIN_FILE_SIZE:
                return False
        except OSError:
            return False
        return True
    except Exception:
        return False


# -------- Seek helpers --------


def normalize_seek(path: Path, offset: float, loop: bool) -> float:
    meta = get_metadata(path)
    dur = meta.duration or 0.0
    if dur > 0:
        if loop:
            offset = offset % dur
        max_seek = max(0.0, dur - 3.0)
        offset = min(offset, max_seek)
    return max(0.0, offset)


def random_seek(path: Path, loop: bool, start_pct: float, end_pct: float) -> float:
    dur = get_metadata(path).duration or 0.0
    if dur <= 0:
        return 0.0
    raw = random.uniform(start_pct, end_pct) * dur
    return normalize_seek(path, raw, loop)


# -------- Pool & grid --------


def pick_videos(
    pool: list[Path], count: int, exts: tuple[str, ...], seed: int | None
) -> list[Path]:
    if seed is not None:
        random.seed(seed)

    candidates = list(pool)
    random.shuffle(candidates)

    def _validate(p: Path) -> tuple[Path, bool]:
        return p, is_valid_video(p)

    selected: list[Path] = []
    checked = 0
    workers = min(8, len(candidates))
    with ThreadPoolExecutor(max_workers=workers) as pool_exec:
        futures = {
            pool_exec.submit(_validate, p): p
            for p in candidates[:min(len(candidates), count * 20)]
        }
        for fut in as_completed(futures):
            p, ok = fut.result()
            checked += 1
            if checked % 5 == 0 or checked <= count:
                print(
                    f"\rValidating: {checked}/{len(futures)} checked, "
                    f"{len(selected)}/{count} found",
                    end="", file=sys.stderr, flush=True,
                )
            if ok:
                selected.append(p)
                if len(selected) >= count:
                    pool_exec.shutdown(wait=False, cancel_futures=True)
                    break

    print(file=sys.stderr)  # newline after progress

    if len(selected) < count:
        raise SystemExit(
            f"Need at least {count} valid videos with extensions {exts}, "
            f"only found {len(selected)}/{count} after checking {checked} candidates."
        )
    return selected


def parse_grid(grid_str: str | None) -> tuple[int, int] | None:
    """Parse 'NxM' grid string. Returns (rows, cols) or None."""
    if not grid_str:
        return None
    parts = grid_str.lower().split("x")
    if len(parts) != 2:
        raise SystemExit(f"Invalid grid format '{grid_str}' — use NxM (e.g. 3x3, 2x4).")
    try:
        r, c = int(parts[0]), int(parts[1])
    except ValueError:
        raise SystemExit(f"Invalid grid format '{grid_str}' — use NxM (e.g. 3x3, 2x4).")
    if r < 1 or c < 1:
        raise SystemExit("Grid rows and columns must be at least 1.")
    if r * c > MAX_TILES:
        raise SystemExit(f"Grid {r}x{c} = {r*c} tiles exceeds max {MAX_TILES}.")
    return r, c


def compute_grid(n: int, grid: tuple[int, int] | None) -> tuple[int, int]:
    """Compute (rows, cols) for n tiles. If grid is given, use it explicitly."""
    if grid:
        return grid
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    return rows, cols


# -------- Hardware acceleration --------


def list_hwaccels() -> set[str]:
    out = run_cmd(["ffmpeg", "-hide_banner", "-hwaccels"])
    accels: set[str] = set()
    for line in out.splitlines():
        s = line.strip().lower()
        if s and not s.startswith("hardware acceleration"):
            accels.add(s)
    return accels


def find_vaapi_device() -> str | None:
    for path in sorted(glob.glob("/dev/dri/renderD*")):
        if os.access(path, os.R_OK | os.W_OK):
            return path
    return None


def _cuda_runtime_available() -> bool:
    try:
        import ctypes.util

        return ctypes.util.find_library("cuda") is not None
    except Exception:
        return False


def choose_hwaccel(pref: str | None = None) -> HwAccelConfig:
    if pref and pref.lower() in ("off", "none"):
        return HwAccelConfig()

    p = (pref or "auto").lower()

    if p == "cuda":
        if _cuda_runtime_available():
            return HwAccelConfig(mode="cuda")
        print(
            "[warn] --hwaccel cuda: CUDA runtime not available, "
            "falling back to software",
            file=sys.stderr,
        )
        return HwAccelConfig()

    if p == "vaapi":
        dev = find_vaapi_device()
        if dev:
            return HwAccelConfig(mode="vaapi", global_opts={"-vaapi_device": dev})
        print(
            "[warn] --hwaccel vaapi: no VAAPI render node found",
            file=sys.stderr,
        )
        return HwAccelConfig()

    accels = list_hwaccels()
    if "cuda" in accels and _cuda_runtime_available():
        return HwAccelConfig(mode="cuda")
    if "vaapi" in accels:
        dev = find_vaapi_device()
        if dev:
            return HwAccelConfig(mode="vaapi", global_opts={"-vaapi_device": dev})
    return HwAccelConfig()


# -------- ffmpeg command builders --------

# Codecs for which hardware decode is skipped (hardware scale filter would then
# be applied to software frames and fail).
_SOFTWARE_ONLY_CODECS = {"mpeg4", "msmpeg4v3", "mpeg1video"}


def _wants_hw(cfg: PipelineConfig, tile: TileState) -> bool:
    codec = (tile.metadata.video_codec or "").lower()
    return cfg.hwaccel.enabled and codec not in _SOFTWARE_ONLY_CODECS


def _tile_filter_ops(cfg: PipelineConfig, tile: TileState) -> list[str]:
    """Per-tile scale/pad/fps chain: fill the cell, preserve aspect, black bars."""
    ops: list[str] = []

    if _wants_hw(cfg, tile) and cfg.hwaccel.mode == "cuda":
        ops.append(
            f"scale_cuda={cfg.cell_w}:{cfg.cell_h}:"
            f"force_original_aspect_ratio=decrease"
        )
        ops += ["hwdownload", "format=nv12"]
    elif _wants_hw(cfg, tile) and cfg.hwaccel.mode == "vaapi":
        ops.append(
            f"scale_vaapi=w={cfg.cell_w}:h={cfg.cell_h}:"
            f"force_original_aspect_ratio=decrease"
        )
        ops += ["hwdownload", "format=nv12"]
    else:
        ops.append(
            f"scale={cfg.cell_w}:{cfg.cell_h}:"
            f"force_original_aspect_ratio=decrease"
        )

    ops.append(f"pad={cfg.cell_w}:{cfg.cell_h}:(ow-iw)/2:(oh-ih)/2:black")
    ops.append("format=yuv420p")
    ops.append(f"fps={VIDEO_FPS}")
    return ops


def build_tile_video_cmd(cfg: PipelineConfig, index: int) -> tuple[list[str], int, int]:
    """ffmpeg command for ONE tile: cell-sized raw RGB frames to stdout.

    One process per tile means replacing/seeking a tile never touches the
    other tiles' decoders (no whole-wall restart, no playback reset).
    """
    tile = cfg.tiles[index]

    args = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "info" if cfg.verbose else "error",
    ]
    for k, v in cfg.hwaccel.global_opts.items():
        args += [k, v]

    if cfg.loop:
        args += ["-stream_loop", "-1"]
    if _wants_hw(cfg, tile) and cfg.hwaccel.mode == "cuda":
        args += ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
    elif _wants_hw(cfg, tile) and cfg.hwaccel.mode == "vaapi":
        args += ["-hwaccel", "vaapi", "-hwaccel_output_format", "vaapi"]

    args += ["-ss", f"{tile.seek:.3f}", "-i", str(tile.path.absolute())]
    args += ["-an", "-vf", ",".join(_tile_filter_ops(cfg, tile))]
    args += ["-pix_fmt", "rgb24", "-f", "rawvideo", "-"]
    return args, cfg.cell_w, cfg.cell_h


def build_tile_audio_cmd(cfg: PipelineConfig, index: int) -> list[str] | None:
    """ffmpeg command for ONE tile's audio → WAV to stdout (None when muted).

    Each tile gets its own ffmpeg→ffplay chain, so changing a tile restarts
    only that tile's audio and the others keep playing. The OS/sound server
    mixes the per-tile streams (``--volume`` is applied per stream).
    """
    if cfg.no_audio:
        return None
    tile = cfg.tiles[index]
    if not tile.metadata.has_audio:
        return None

    if cfg.audio_mode == "one":
        with_audio = [i for i, t in enumerate(cfg.tiles) if t.metadata.has_audio]
        chosen = cfg.audio_tile if cfg.audio_tile in with_audio else with_audio[0]
        if index != chosen:
            return None

    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "info" if cfg.verbose else "error",
        "-ss", f"{tile.seek:.3f}",
        "-i", str(tile.path.absolute()),
        "-vn",
        "-af", f"aresample=async=1:min_hard_comp=0.100,volume={cfg.volume}",
        "-c:a", "pcm_s16le", "-ar", str(cfg.audio_rate),
        "-f", "wav", "-",
    ]


# -------- Kill helper --------


def kill_proc(p: subprocess.Popen | None) -> None:
    """Immediately tear down a child decoder/player (stateless, so SIGKILL)."""
    if p is None:
        return
    with contextlib.suppress(ProcessLookupError, PermissionError):
        p.kill()
    with contextlib.suppress(Exception):
        p.wait(timeout=0.2)


def signal_proc(p: subprocess.Popen | None, sig: int) -> None:
    """Best-effort delivery of SIGSTOP/SIGCONT (process may already be gone)."""
    if p is None:
        return
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.kill(p.pid, sig)


# -------- Pygame viewer --------


class VideoWindow:
    """Pygame window displaying raw RGB frames from ffmpeg.

    Two independent pipelines:
      - Video: ffmpeg → rawvideo pipe → pygame display
      - Audio: ffmpeg → WAV pipe → ffplay -nodisp (when not --no-audio)

    Audio ffplay is started BEFORE video ffmpeg so pipes are ready.
    The window stays open across tile replacements.
    """

    def __init__(
        self,
        width: int,
        height: int,
        fullscreen: bool = False,
        verbose: bool = False,
    ):
        pygame.display.init()
        pygame.key.set_repeat(300, 80)

        flags = pygame.FULLSCREEN | pygame.SCALED if fullscreen else 0
        self.screen = pygame.display.set_mode((width, height), flags)
        pygame.display.set_caption(TITLE)
        if fullscreen:
            pygame.mouse.set_visible(False)

        self.width = width
        self.height = height
        self.verbose = verbose
        self.running = False
        self.paused = False

        self._pending_seek: float | None = None
        self._clock = pygame.time.Clock()

        # One decoder per tile, composited by pygame. These lists mirror
        # len(self._tiles) and let a replace/seek touch only one tile.
        self._vid_procs: list[subprocess.Popen | None] = []
        self._vid_bufs: list[bytearray] = []
        self._tile_surfaces: list[pygame.Surface | None] = []
        self._aud_ffmpeg: list[subprocess.Popen | None] = []
        self._aud_ffplay: list[subprocess.Popen | None] = []
        self._audio_gated: list[bool] = []  # per-tile: audio held for first frame
        self._audio_started: list[bool] = []  # per-tile: ffplay clock is running
        self._audio_pos: list[float | None] = []  # per-tile ffplay clock (s)
        self._audio_release_wall: list[float | None] = []
        self._vid_allowed: list[bool] = []  # per-tile: may present video
        self._rows = 0
        self._cols = 0
        self._border = 0
        self._cell_w = 0
        self._cell_h = 0
        self._tile_frame_size = 0
        self._wall: pygame.Surface | None = None
        # Wall-clock frame budget: content advances at VIDEO_FPS regardless of
        # how fast the loop actually iterates.
        self._last_tick: float | None = None
        self._frames_due = 0.0

        # External state
        self._tiles: list[TileState] = []
        self._pool: list[Path] = []
        self._banned: set[Path] = set()  # paths that caused crashes
        self._start_pct = 0.5
        self._end_pct = 0.75
        self._loop = False
        self._make_cfg = None

    # ----- Per-tile pipelines -----

    def _cfg(self) -> PipelineConfig:
        return self._make_cfg(self._tiles)

    def _start_tile(self, index: int) -> None:
        """(Re)start video + audio for a single tile only."""
        cfg = self._cfg()
        self._stop_tile_video(index)
        self._stop_tile_audio(index)

        vid_cmd, _, _ = build_tile_video_cmd(cfg, index)
        proc = subprocess.Popen(
            vid_cmd,
            stdout=subprocess.PIPE,
            stderr=None if self.verbose else subprocess.DEVNULL,
        )
        if proc.stdout is not None:
            fd = proc.stdout.fileno()
            # A raw frame (e.g. 480x270x3 = 388 KB) is far bigger than the
            # default 64 KB pipe; grow it so a whole frame can be buffered and
            # read without stalling the loop.
            want = max(self._tile_frame_size * 4, 1 << 20)
            setpipesz = getattr(fcntl, "F_SETPIPE_SZ", None)
            if setpipesz is not None:
                with contextlib.suppress(OSError, ValueError):
                    fcntl.fcntl(fd, setpipesz, min(want, _pipe_max_size()))
            with contextlib.suppress(OSError, ValueError):
                os.set_blocking(fd, False)
        self._vid_procs[index] = proc
        self._vid_bufs[index] = bytearray()
        self._tile_surfaces[index] = None
        self._audio_started[index] = False
        self._audio_pos[index] = None
        self._audio_release_wall[index] = None

        aud_cmd = build_tile_audio_cmd(cfg, index)
        self._vid_allowed[index] = aud_cmd is None
        if aud_cmd:
            self._start_tile_audio(index, aud_cmd)

        if self.paused:
            signal_proc(proc, signal.SIGSTOP)
            signal_proc(self._aud_ffmpeg[index], signal.SIGSTOP)

    def _stop_tile_video(self, index: int) -> None:
        kill_proc(self._vid_procs[index])
        self._vid_procs[index] = None

    def _start_tile_audio(self, index: int, cmd: list[str]) -> None:
        if not shutil.which("ffplay"):
            if self.verbose:
                print("[warn] ffplay not found — audio disabled", file=sys.stderr)
            self._vid_allowed[index] = True
            return
        # ffplay first (reader), held until this tile's first video frame; then
        # we wait for its clock to actually start before showing the video.
        ffplay = subprocess.Popen(
            [
                "ffplay", "-nodisp", "-vn",
                # info level so the status line (audio clock) is emitted.
                "-loglevel", "info",
                "-autoexit", "-i", "pipe:0",
            ],
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self._aud_ffplay[index] = ffplay
        self._audio_started[index] = False
        signal_proc(ffplay, signal.SIGSTOP)
        self._audio_gated[index] = ffplay.poll() is None
        threading.Thread(
            target=self._watch_audio_clock, args=(index, ffplay), daemon=True
        ).start()
        # ffmpeg second (writes into ffplay stdin)
        self._aud_ffmpeg[index] = subprocess.Popen(cmd, stdout=ffplay.stdin)

    def _watch_audio_clock(self, index: int, proc: subprocess.Popen) -> None:
        """Set ``_audio_started[index]`` when ffplay reports it is playing."""
        if proc.stderr is None:
            return
        fd = proc.stderr.fileno()
        pending = b""
        while True:
            try:
                chunk = os.read(fd, 4096)
            except OSError:
                break
            if not chunk:
                break
            pending += chunk
            parts = re.split(rb"[\r\n]", pending)
            pending = parts.pop()
            for line in parts:
                m = _AUDIO_POS_RE.match(line)
                if m and self._aud_ffplay[index] is proc:
                    self._audio_pos[index] = float(m.group(1))
                    self._audio_started[index] = True

    def _release_tile_audio(self, index: int) -> None:
        if index < len(self._audio_gated) and self._audio_gated[index]:
            self._audio_gated[index] = False
            self._audio_release_wall[index] = time.monotonic()
            signal_proc(self._aud_ffplay[index], signal.SIGCONT)

    def _audio_timed_out(self, index: int) -> bool:
        released = self._audio_release_wall[index]
        return released is not None and time.monotonic() - released > AUDIO_START_TIMEOUT

    def _stop_tile_audio(self, index: int) -> None:
        kill_proc(self._aud_ffmpeg[index])
        kill_proc(self._aud_ffplay[index])
        self._aud_ffmpeg[index] = None
        self._aud_ffplay[index] = None
        self._audio_gated[index] = False
        self._audio_started[index] = False
        self._audio_pos[index] = None
        self._audio_release_wall[index] = None

    def _stop_procs(self) -> None:
        for i in range(len(self._vid_procs)):
            self._stop_tile_video(i)
            self._stop_tile_audio(i)

    def _start_all(self) -> None:
        """Build all tile decoders for the current grid."""
        cfg = self._cfg()
        self._rows, self._cols = cfg.rows, cfg.cols
        self._cell_w, self._cell_h = cfg.cell_w, cfg.cell_h
        self._border = cfg.border

        w = cfg.cols * (cfg.cell_w + cfg.border) - cfg.border
        h = cfg.rows * (cfg.cell_h + cfg.border) - cfg.border
        if (w, h) != (self.width, self.height):
            self.width, self.height = w, h
            self.screen = pygame.display.set_mode((w, h), self.screen.get_flags())
        self._tile_frame_size = cfg.cell_w * cfg.cell_h * 3
        self._wall = pygame.Surface((w, h))

        n = len(self._tiles)
        self._vid_procs = [None] * n
        self._vid_bufs = [bytearray() for _ in range(n)]
        self._tile_surfaces = [None] * n
        self._aud_ffmpeg = [None] * n
        self._aud_ffplay = [None] * n
        self._audio_gated = [False] * n
        self._audio_started = [False] * n
        self._audio_pos = [None] * n
        self._audio_release_wall = [None] * n
        self._vid_allowed = [True] * n
        for i in range(n):
            self._start_tile(i)

    def _pump_frames(self, due: int) -> None:
        """Advance every tile by ``due`` frames of content (newest wins).

        ``due`` comes from the wall clock (see ``run``), so content plays at
        ``VIDEO_FPS`` even if the loop iterates faster or slower. When behind,
        older frames are discarded so a tile snaps back to the present.
        """
        if due <= 0:
            return
        fs = self._tile_frame_size
        want = fs * due
        for i, proc in enumerate(self._vid_procs):
            if proc is None or proc.stdout is None or proc.poll() is not None:
                continue
            fd = proc.stdout.fileno()
            buf = self._vid_bufs[i]

            if not self._vid_allowed[i]:
                # Hold this tile's video until its audio pipeline is actually
                # playing (its seek can take a while on 4K/NFS), otherwise the
                # video would start first and stay ahead of the audio.
                while len(buf) < fs:
                    try:
                        data = os.read(fd, fs - len(buf))
                    except (BlockingIOError, InterruptedError, OSError):
                        break
                    if not data:
                        break
                    buf += data
                if len(buf) < fs:
                    continue
                if self._audio_release_wall[i] is None:
                    self._audio_release_wall[i] = time.monotonic()
                if self._audio_gated[i]:
                    self._release_tile_audio(i)
                    continue
                if not self._audio_started[i] and not self._audio_timed_out(i):
                    continue
                self._vid_allowed[i] = True

            while len(buf) < want:
                try:
                    data = os.read(fd, want - len(buf))
                except (BlockingIOError, InterruptedError, OSError):
                    break
                if not data:
                    break
                buf += data

            if len(buf) < fs:
                continue
            avail = len(buf) // fs
            # Present the newest available frame, drop the rest (catch-up).
            frame = bytes(buf[(avail - 1) * fs: avail * fs])
            del buf[: avail * fs]
            self._tile_surfaces[i] = pygame.image.frombuffer(
                frame, (self._cell_w, self._cell_h), "RGB"
            )

    def _composite(self) -> None:
        if self._wall is None:
            return
        self._wall.fill((0, 0, 0))
        stride_w = self._cell_w + self._border
        stride_h = self._cell_h + self._border
        for i, surf in enumerate(self._tile_surfaces):
            if surf is None:
                continue
            col = i % self._cols
            row = i // self._cols
            self._wall.blit(surf, (col * stride_w, row * stride_h))

    # ----- Keyboard -----

    def _handle_key(self, key: int, mod: int):
        if key == pygame.K_q:
            self.running = False
            return
        if key == pygame.K_SPACE:
            self._toggle_pause()
            return
        if key == pygame.K_r:
            self._replace_tile(random.randrange(len(self._tiles)))
            return
        if key == pygame.K_f:
            self._pending_seek = 30.0 if (mod & pygame.KMOD_SHIFT) else 10.0
            return
        if key == pygame.K_b:
            self._pending_seek = -30.0 if (mod & pygame.KMOD_SHIFT) else -10.0
            return
        idx = self._key_to_tile(key)
        if idx is not None and idx < len(self._tiles):
            if self._pending_seek is not None:
                self._seek_tile(idx, self._pending_seek)
                self._pending_seek = None
            else:
                self._replace_tile(idx)

    @staticmethod
    def _key_to_tile(key: int) -> int | None:
        if pygame.K_1 <= key <= pygame.K_9:
            return key - pygame.K_1
        if key == pygame.K_0:
            return MAX_TILES - 1
        if pygame.K_a <= key <= pygame.K_f:
            return key - pygame.K_a + 9
        return None

    def _toggle_pause(self) -> None:
        if not self.paused:
            for i in range(len(self._vid_procs)):
                signal_proc(self._vid_procs[i], signal.SIGSTOP)
                signal_proc(self._aud_ffmpeg[i], signal.SIGSTOP)
                signal_proc(self._aud_ffplay[i], signal.SIGSTOP)
            self.paused = True
        else:
            for i in range(len(self._vid_procs)):
                signal_proc(self._vid_procs[i], signal.SIGCONT)
                signal_proc(self._aud_ffmpeg[i], signal.SIGCONT)
                # Keep audio held if this tile's first frame is still pending.
                if not self._audio_gated[i]:
                    signal_proc(self._aud_ffplay[i], signal.SIGCONT)
            self.paused = False

    # ----- Pipeline lifecycle -----

    def _replace_tile(self, index: int, ban: Path | None = None) -> None:
        if not (0 <= index < len(self._tiles)):
            return

        if ban:
            self._banned.add(ban)

        active_paths = {t.path for j, t in enumerate(self._tiles) if j != index}
        current_path = self._tiles[index].path
        pool = self._pool

        def _pick(candidates: list[Path], label: str) -> Path | None:
            if not candidates:
                return None
            shuffled = list(candidates)
            random.shuffle(shuffled)
            for p in shuffled:
                if p in self._banned:
                    continue
                if is_valid_video(p):
                    if self.verbose:
                        print(f"[info] replace: {label}: {p.name}", file=sys.stderr)
                    return p
            return None

        new_path = _pick(
            [p for p in pool if p not in active_paths and p != current_path],
            "unused+different",
        )
        if new_path is None:
            new_path = _pick([p for p in pool if p not in active_paths], "unused")
        if new_path is None:
            new_path = _pick([p for p in pool if p != current_path], "any different")
        if new_path is None:
            if self.verbose:
                print("[info] replace: fallback to current", file=sys.stderr)
            new_path = current_path

        self._tiles[index] = TileState(
            path=new_path,
            seek=random_seek(new_path, self._loop, self._start_pct, self._end_pct),
        )
        # Only this tile restarts; the others keep playing untouched.
        self._start_tile(index)

    def _seek_tile(self, index: int, delta: float) -> None:
        if not (0 <= index < len(self._tiles)):
            return
        t = self._tiles[index]
        t.seek = normalize_seek(t.path, t.seek + delta, self._loop)
        self._start_tile(index)

    def run(
        self,
        tiles: list[TileState],
        pool: list[Path],
        start_pct: float,
        end_pct: float,
        loop: bool,
        make_cfg,
    ) -> None:
        """Main loop: pump tile decoders, composite, handle pygame events."""
        self._tiles = tiles
        self._pool = pool
        self._start_pct = start_pct
        self._end_pct = end_pct
        self._loop = loop
        self._make_cfg = make_cfg
        self.running = True

        self._start_all()

        print(
            "Controls:\n"
            "  SPACE     = pause/resume\n"
            "  r         = replace a random tile\n"
            "  1-9, a-f  = replace specific tile (0=last, a=10..f=15)\n"
            "  f/b       = seek +10s/-10s, then press a tile number\n"
            "  Shift+f/b = seek +30s/-30s\n"
            "  q         = quit\n",
            file=sys.stderr,
        )

        try:
            while self.running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                        break
                    if event.type == pygame.KEYDOWN:
                        self._handle_key(event.key, event.mod)

                if not self.running:
                    break

                if not self.paused:
                    # Recycle any tile whose decoder died (EOF, corrupt file…).
                    for i, proc in enumerate(self._vid_procs):
                        if proc is not None and proc.poll() is not None:
                            self._replace_tile(i, ban=self._tiles[i].path)
                            break

                    # How many 1/30s frames of content should have advanced?
                    now = time.monotonic()
                    if self._last_tick is None:
                        self._last_tick = now
                        self._frames_due = 1.0
                    else:
                        self._frames_due += (now - self._last_tick) * VIDEO_FPS
                        self._last_tick = now
                    due = int(self._frames_due)
                    self._frames_due -= due
                    if due > MAX_CATCHUP_FRAMES:
                        due = MAX_CATCHUP_FRAMES
                        self._frames_due = 0.0

                    self._pump_frames(due)
                    self._composite()
                    if self._wall is not None:
                        self._blit_frame(self._wall)
                        pygame.display.flip()
                else:
                    self._last_tick = None

                self._clock.tick(VIDEO_FPS)
        finally:
            self._stop_procs()
            pygame.display.quit()
            pygame.quit()

    def _blit_frame(self, surface: pygame.Surface) -> None:
        """Show a frame, preserving aspect ratio with black bars.

        A tiling compositor (e.g. niri) can resize the window to something
        that no longer matches the grid, so scale to *fit* and centre instead
        of stretching the wall to fill.
        """
        win_w, win_h = self.screen.get_size()
        if (win_w, win_h) == (self.width, self.height):
            self.screen.blit(surface, (0, 0))
            return

        scale = min(win_w / self.width, win_h / self.height)
        dst_w = max(1, int(self.width * scale))
        dst_h = max(1, int(self.height * scale))
        self.screen.fill((0, 0, 0))
        self.screen.blit(
            pygame.transform.scale(surface, (dst_w, dst_h)),
            ((win_w - dst_w) // 2, (win_h - dst_h) // 2),
        )


# -------- Main --------


def main() -> None:
    need("ffmpeg")
    need("ffprobe")

    ap = argparse.ArgumentParser(
        description="Single-window video wall with pygame viewer + audio.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Keyboard controls (pygame window):\n"
            "  SPACE       = pause/resume\n"
            "  r           = replace a random tile\n"
            "  1-9, a-f    = replace specific tile (0=last, a=10..f=15)\n"
            "  f/F + tile  = seek forward 10s/30s\n"
            "  b/B + tile  = seek backward 10s/30s\n"
            "  q           = quit"
        ),
    )
    ap.add_argument("folder", type=Path, help="Folder with videos")

    # Grid
    ap.add_argument("--count", "-n", type=int, default=4, help="Number of tiles (1–16)")
    ap.add_argument(
        "--grid", metavar="NxM",
        help="Grid layout (e.g. 3x3, 2x4). Auto-computed from --count if omitted.",
    )
    ap.add_argument("--cell-width", type=int, default=480, help="Tile width in px")
    ap.add_argument("--cell-height", type=int, default=270, help="Tile height in px")
    ap.add_argument("--border", type=int, default=0, help="Black gap (px) between tiles")

    # Playback
    ap.add_argument("--loop", action="store_true", help="Loop each input")
    ap.add_argument(
        "--exts",
        default=",".join(DEFAULT_EXTS),
        help="Comma-separated extensions (default: %(default)s)",
    )
    ap.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan subdirectories for videos",
    )
    ap.add_argument(
        "--start-pct",
        type=float, default=0.5, metavar="PCT",
        help="Start of random seek window (0–1, default: 0.5)",
    )
    ap.add_argument(
        "--end-pct",
        type=float, default=0.75, metavar="PCT",
        help="End of random seek window (0–1, default: 0.75)",
    )
    ap.add_argument("--seed", type=int, help="Random seed for deterministic runs")
    ap.add_argument("--verbose", "-v", action="store_true", help="Show ffmpeg logs")

    # Audio (enabled by default)
    ap.add_argument(
        "--no-audio", action="store_true",
        help="Disable audio playback",
    )
    ap.add_argument(
        "--volume", type=float, default=0.5,
        help="Per-input pre-mix gain (default: 0.5)",
    )
    ap.add_argument(
        "--audio-mode",
        choices=["mix", "one"], default="mix",
        help="Mix all audio or use only one tile's audio",
    )
    ap.add_argument(
        "--audio-tile",
        type=int, default=0, metavar="TILE",
        help="Tile index (0-based) for --audio-mode=one",
    )
    ap.add_argument(
        "--audio-rate",
        type=int, default=48000, metavar="HZ",
        help="Sample rate in Hz (default: 48000)",
    )

    # Display
    ap.add_argument(
        "--fullscreen", "-f", action="store_true",
        help="Fullscreen mode",
    )

    # HW
    ap.add_argument(
        "--hwaccel",
        default="off",
        choices=["auto", "off", "cuda", "vaapi"],
        help="HW decode: off (default), auto, cuda, or vaapi",
    )

    args = ap.parse_args()

    grid = parse_grid(args.grid)

    if args.count < 1 or args.count > MAX_TILES:
        raise SystemExit(f"--count must be between 1 and {MAX_TILES}.")
    if grid and grid[0] * grid[1] < args.count:
        raise SystemExit(
            f"Grid {grid[0]}x{grid[1]} = {grid[0]*grid[1]} slots "
            f"is smaller than --count {args.count}."
        )
    if not args.folder.is_dir():
        raise SystemExit(f"{args.folder} is not a directory.")

    exts = tuple(s.strip().lower() for s in args.exts.split(",") if s.strip())

    if args.recursive:
        print(f"Scanning {args.folder} recursively...", file=sys.stderr, flush=True)
        ext_set = set(exts)
        pool = []
        for dirpath, _, filenames in os.walk(args.folder):
            for name in filenames:
                if Path(name).suffix.lower() in ext_set:
                    pool.append(Path(dirpath) / name)
    else:
        pool = [
            p for p in args.folder.iterdir()
            if p.is_file() and p.suffix.lower() in exts
        ]

    if len(pool) < args.count:
        raise SystemExit(
            f"Need at least {args.count} videos with extensions {exts}, "
            f"found {len(pool)} in {args.folder}."
        )

    initial_paths = pick_videos(pool, args.count, exts, args.seed)
    tiles: list[TileState] = [
        TileState(
            path=p,
            seek=random_seek(p, args.loop, args.start_pct, args.end_pct),
        )
        for p in initial_paths
    ]
    rows, cols = compute_grid(args.count, grid)

    hw_cfg = choose_hwaccel(args.hwaccel)
    if args.verbose:
        print(f"[info] HW accel: {hw_cfg.mode or 'software'}", file=sys.stderr)
        if hw_cfg.mode == "vaapi" and "-vaapi_device" in hw_cfg.global_opts:
            print(
                f"  (device {hw_cfg.global_opts['-vaapi_device']})",
                file=sys.stderr,
            )
        print(f"[info] Grid: {cols}×{rows} = {args.count} tiles", file=sys.stderr)

    total_w = cols * (args.cell_width + args.border) - args.border
    total_h = rows * (args.cell_height + args.border) - args.border

    def make_cfg(cur_tiles: list[TileState]) -> PipelineConfig:
        return PipelineConfig(
            tiles=cur_tiles,
            rows=rows,
            cols=cols,
            cell_w=args.cell_width,
            cell_h=args.cell_height,
            loop=args.loop,
            volume=args.volume,
            hwaccel=hw_cfg,
            verbose=args.verbose,
            no_audio=args.no_audio,
            audio_mode=args.audio_mode,
            audio_tile=args.audio_tile,
            audio_rate=args.audio_rate,
            border=args.border,
            fullscreen=args.fullscreen,
        )

    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        raise SystemExit(
            "No display detected (DISPLAY/WAYLAND_DISPLAY not set).\n"
            "Run from a desktop session with a display server running."
        )

    viewer = VideoWindow(
        width=total_w,
        height=total_h,
        fullscreen=args.fullscreen,
        verbose=args.verbose,
    )

    try:
        viewer.run(
            tiles=tiles,
            pool=pool,
            start_pct=args.start_pct,
            end_pct=args.end_pct,
            loop=args.loop,
            make_cfg=make_cfg,
        )
    except pygame.error as exc:
        raise SystemExit(f"Cannot open display: {exc}")


if __name__ == "__main__":
    main()
