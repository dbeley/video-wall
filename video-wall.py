#!/usr/bin/env python3
"""video-wall — single-window video wall with pygame viewer and audio."""

from __future__ import annotations

import argparse
import contextlib
import ctypes.util
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

# Must be set before pygame is imported: silences its banner and the spurious
# AVX2 warning some builds print.
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
warnings.filterwarnings("ignore", message=".*avx2.*", category=RuntimeWarning)

import pygame  # late import: the environment above has to be set first

MIN_FILE_SIZE = 50 * 1024  # 50 KB — skips torrent stubs / partial downloads
DEFAULT_EXTS = (".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm")
MAX_TILES = 16
TITLE = "Video Wall"
VIDEO_FPS = 30
# A tile that falls this far behind may drop more than one frame per loop tick.
MAX_CATCHUP_FRAMES = 60
# How long to hold a tile's picture while its audio starts, before giving up
# and playing the video anyway (a broken audio pipeline must not stay black).
AUDIO_START_TIMEOUT = 3.0
# How long ffplay may go without reporting its clock before we stop trusting it
# as the master clock (audio ended, ffplay died, stream stalled).
AUDIO_CLOCK_STALE = 1.0
# ffplay status line, e.g. "  12.34 M-A:  0.000 fd=...". The leading number is
# the audio clock in seconds; lines that have no clock yet show "nan" and fail
# to match.
AUDIO_POS_RE = re.compile(rb"^\s*([-\d.]+)\s+M-A:")

CONTROLS = (
    "  SPACE        = pause/resume\n"
    "  r            = replace a random tile\n"
    "  1-9, 0, a-f  = replace specific tile (1-9=1-9, 0=10, a-f=11-16)\n"
    "  Left/Right   = seek -10s/+10s, then press a tile key\n"
    "  Shift+arrows = seek -30s/+30s\n"
    "  right-click  = replace the tile under the cursor\n"
    "  q            = quit"
)


# -------- Models --------


@dataclass(frozen=True)
class VideoMeta:
    duration: float | None
    has_audio: bool
    has_video: bool
    video_codec: str | None


@dataclass(frozen=True)
class HwAccelConfig:
    """Hardware decode setup. ``mode is None`` means software decoding."""

    mode: str | None = None  # "cuda" | "vaapi" | None
    vaapi_device: str | None = None


@dataclass
class PipelineConfig:
    """Everything needed to build a tile's ffmpeg commands.

    ``tiles`` is mutated in place when a tile is replaced or seeked, so one
    config serves the whole run.
    """

    tiles: list[TileState]
    rows: int
    cols: int
    cell_w: int
    cell_h: int
    loop: bool
    volume: float
    hwaccel: HwAccelConfig
    verbose: bool
    no_audio: bool
    audio_mode: str  # "mix" | "one"
    audio_tile: int | None
    audio_rate: int
    border: int
    fullscreen: bool


@dataclass
class TileState:
    """What a tile is playing: a file and a position in it."""

    path: Path
    seek: float

    @property
    def metadata(self) -> VideoMeta:
        return get_metadata(self.path)


@dataclass
class TilePlayback:
    """Runtime state of one tile: its decoders, frame buffer and audio clock."""

    video: subprocess.Popen | None = None
    buffer: bytearray = field(default_factory=bytearray)
    surface: pygame.Surface | None = None
    # Frames consumed since this tile (re)started, i.e. its content position.
    frames_consumed: int = 0
    # Latched once the audio gate has opened; the picture may be shown then.
    video_ready: bool = False
    audio_ffmpeg: subprocess.Popen | None = None
    audio_player: subprocess.Popen | None = None
    audio_held: bool = False  # ffplay is SIGSTOPped until the first frame
    audio_released_at: float | None = None
    audio_clock: float | None = None  # ffplay's position, in seconds
    audio_clock_at: float | None = None  # when that position was reported

    def clock(self) -> float | None:
        """ffplay's audio position, or None when it is not live."""
        if self.audio_clock is None or self.audio_clock_at is None:
            return None
        if time.monotonic() - self.audio_clock_at > AUDIO_CLOCK_STALE:
            return None
        return self.audio_clock


# -------- Small helpers --------


def require_tools(*names: str) -> None:
    for name in names:
        if shutil.which(name) is None:
            raise SystemExit(f"Missing '{name}' in PATH.")


def run_cmd(argv: list[str]) -> str:
    """Run a command and return its output, or "" if it failed."""
    try:
        return subprocess.check_output(argv, stderr=subprocess.STDOUT).decode(
            "utf-8", "ignore"
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"[warn] {' '.join(argv)} failed: {exc}", file=sys.stderr)
        return ""


def pipe_max_size() -> int:
    """Largest pipe size we are allowed to request (Linux)."""
    try:
        with open("/proc/sys/fs/pipe-max-size") as fh:
            return int(fh.read().strip())
    except (OSError, ValueError):
        return 1 << 20


def kill_proc(proc: subprocess.Popen | None) -> None:
    """SIGKILL a child decoder/player; they hold no state worth flushing."""
    if proc is None:
        return
    with contextlib.suppress(ProcessLookupError):
        proc.kill()
    with contextlib.suppress(subprocess.TimeoutExpired):
        proc.wait(timeout=0.2)


def signal_proc(proc: subprocess.Popen | None, sig: int) -> None:
    """Best-effort SIGSTOP/SIGCONT (the process may already be gone)."""
    if proc is None:
        return
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.kill(proc.pid, sig)


# -------- Metadata --------

_metadata_cache: dict[Path, VideoMeta] = {}


def get_metadata(path: Path) -> VideoMeta:
    """Probe a file once and cache the result."""
    cached = _metadata_cache.get(path)
    if cached is not None:
        return cached

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
    video_codec: str | None = None
    try:
        data = json.loads(out) if out else {}
        duration_text = (data.get("format") or {}).get("duration")
        if duration_text:
            duration = float(duration_text)
        for stream in data.get("streams") or []:
            kind = (stream.get("codec_type") or "").lower()
            has_audio = has_audio or kind == "audio"
            if kind == "video":
                has_video = True
                codec = stream.get("codec_name")
                if video_codec is None and isinstance(codec, str):
                    video_codec = codec.lower()
    except (ValueError, TypeError, AttributeError) as exc:
        print(f"[warn] cannot parse ffprobe output for {path}: {exc}", file=sys.stderr)

    meta = VideoMeta(duration, has_audio, has_video, video_codec)
    _metadata_cache[path] = meta
    return meta


def is_valid_video(path: Path) -> bool:
    """Cached, cheap bail-out on stubs: needs a video stream and real bytes."""
    if not get_metadata(path).has_video:
        return False
    try:
        return path.stat().st_size >= MIN_FILE_SIZE
    except OSError:
        return False


# -------- Seeking --------


def normalize_seek(path: Path, position: float, loop: bool) -> float:
    """Clamp a seek into the file, wrapping around when looping."""
    duration = get_metadata(path).duration or 0.0
    if duration > 0:
        if loop:
            position %= duration
        position = min(position, max(0.0, duration - 3.0))
    return max(0.0, position)


def random_seek(path: Path, loop: bool, start_pct: float, end_pct: float) -> float:
    duration = get_metadata(path).duration or 0.0
    if duration <= 0:
        return 0.0
    return normalize_seek(path, random.uniform(start_pct, end_pct) * duration, loop)


# -------- Folder scan, validation, grid --------


def scan_folder(folder: Path, exts: tuple[str, ...], recursive: bool) -> list[Path]:
    wanted = set(exts)
    if not recursive:
        return [
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in wanted
        ]
    print(f"Scanning {folder} recursively...", file=sys.stderr, flush=True)
    return [
        Path(dirpath) / name
        for dirpath, _, names in os.walk(folder)
        for name in names
        if Path(name).suffix.lower() in wanted
    ]


def pick_videos(
    pool: list[Path], count: int, exts: tuple[str, ...], seed: int | None
) -> list[Path]:
    """Pick ``count`` distinct, playable videos, probing them in parallel."""
    if seed is not None:
        random.seed(seed)
    candidates = list(pool)
    random.shuffle(candidates)

    selected: list[Path] = []
    checked = 0
    workers = min(8, len(candidates))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(is_valid_video, path): path
            for path in candidates[: count * 20]
        }
        for future in as_completed(futures):
            checked += 1
            if checked % 5 == 0:
                print(
                    f"\rValidating: {checked}/{len(futures)} checked, "
                    f"{len(selected)}/{count} found",
                    end="",
                    file=sys.stderr,
                    flush=True,
                )
            if future.result():
                selected.append(futures[future])
                if len(selected) >= count:
                    # Leave the remaining probes queued but unfinished.
                    executor.shutdown(wait=False, cancel_futures=True)
                    break
    print(file=sys.stderr)  # end the progress line

    if len(selected) < count:
        raise SystemExit(
            f"Need at least {count} valid videos with extensions {exts}, "
            f"only found {len(selected)}/{count} after checking {checked} candidates."
        )
    return selected


def parse_grid(text: str | None) -> tuple[int, int] | None:
    """Parse an ``NxM`` layout into (rows, cols), or None when not given."""
    if not text:
        return None
    try:
        rows_text, cols_text = text.lower().split("x")
        rows, cols = int(rows_text), int(cols_text)
    except ValueError:
        raise SystemExit(f"Invalid grid '{text}' — use NxM (e.g. 3x3, 2x4).") from None
    if rows < 1 or cols < 1:
        raise SystemExit("Grid rows and columns must be at least 1.")
    if rows * cols > MAX_TILES:
        raise SystemExit(f"Grid {rows}x{cols} exceeds the {MAX_TILES}-tile maximum.")
    return rows, cols


def compute_grid(count: int, grid: tuple[int, int] | None) -> tuple[int, int]:
    """(rows, cols) for ``count`` tiles: the given grid, else as square as possible."""
    if grid:
        return grid
    cols = math.ceil(math.sqrt(count))
    return math.ceil(count / cols), cols


# -------- Hardware acceleration --------

# Codecs whose frames must stay in software: the hardware scale filters
# cannot consume them.
SOFTWARE_ONLY_CODECS = {"mpeg4", "msmpeg4v3", "mpeg1video"}


def list_hwaccels() -> set[str]:
    out = run_cmd(["ffmpeg", "-hide_banner", "-hwaccels"])
    return {
        line.strip().lower()
        for line in out.splitlines()
        if line.strip() and not line.lower().startswith("hardware acceleration")
    }


def find_vaapi_device() -> str | None:
    for path in sorted(glob.glob("/dev/dri/renderD*")):
        if os.access(path, os.R_OK | os.W_OK):
            return path
    return None


def cuda_runtime_available() -> bool:
    return ctypes.util.find_library("cuda") is not None


def choose_hwaccel(preference: str | None) -> HwAccelConfig:
    """Resolve ``--hwaccel`` against what this machine actually supports."""
    preference = (preference or "auto").lower()
    if preference in ("off", "none"):
        return HwAccelConfig()

    cuda = HwAccelConfig(mode="cuda")
    vaapi = HwAccelConfig(mode="vaapi", vaapi_device=find_vaapi_device())

    if preference == "cuda":
        if cuda_runtime_available():
            return cuda
        print("[warn] --hwaccel cuda: no CUDA runtime, using software", file=sys.stderr)
        return HwAccelConfig()
    if preference == "vaapi":
        if vaapi.vaapi_device:
            return vaapi
        print(
            "[warn] --hwaccel vaapi: no VAAPI render node, using software",
            file=sys.stderr,
        )
        return HwAccelConfig()

    available = list_hwaccels()
    if "cuda" in available and cuda_runtime_available():
        return cuda
    if "vaapi" in available and vaapi.vaapi_device:
        return vaapi
    return HwAccelConfig()


def hw_mode(cfg: PipelineConfig, tile: TileState) -> str | None:
    """Hardware decode mode usable for this tile (None means software)."""
    if cfg.hwaccel.mode is None:
        return None
    if (tile.metadata.video_codec or "").lower() in SOFTWARE_ONLY_CODECS:
        return None
    return cfg.hwaccel.mode


# -------- ffmpeg command builders --------


def tile_filter_chain(cfg: PipelineConfig, tile: TileState) -> list[str]:
    """Fill the cell, keep the aspect ratio (black bars), force ``VIDEO_FPS``."""
    mode = hw_mode(cfg, tile)
    if mode == "cuda":
        scale = (
            f"scale_cuda={cfg.cell_w}:{cfg.cell_h}:"
            f"force_original_aspect_ratio=decrease"
        )
        ops = [scale, "hwdownload", "format=nv12"]
    elif mode == "vaapi":
        scale = (
            f"scale_vaapi=w={cfg.cell_w}:h={cfg.cell_h}:"
            f"force_original_aspect_ratio=decrease"
        )
        ops = [scale, "hwdownload", "format=nv12"]
    else:
        ops = [
            f"scale={cfg.cell_w}:{cfg.cell_h}:force_original_aspect_ratio=decrease"
        ]
    ops += [
        f"pad={cfg.cell_w}:{cfg.cell_h}:(ow-iw)/2:(oh-ih)/2:black",
        "format=yuv420p",
        f"fps={VIDEO_FPS}",
    ]
    return ops


def build_tile_video_cmd(cfg: PipelineConfig, index: int) -> list[str]:
    """ffmpeg argv for one tile: cell-sized raw RGB24 frames on stdout.

    One process per tile, so replacing or seeking a tile never touches the
    other tiles' decoders.
    """
    tile = cfg.tiles[index]
    args = ["ffmpeg", "-hide_banner", "-loglevel", "info" if cfg.verbose else "error"]
    if cfg.hwaccel.vaapi_device:
        args += ["-vaapi_device", cfg.hwaccel.vaapi_device]
    if cfg.loop:
        args += ["-stream_loop", "-1"]
    mode = hw_mode(cfg, tile)
    if mode:
        args += ["-hwaccel", mode, "-hwaccel_output_format", mode]
    args += ["-ss", f"{tile.seek:.3f}", "-i", str(tile.path.absolute())]
    args += ["-an", "-vf", ",".join(tile_filter_chain(cfg, tile))]
    args += ["-pix_fmt", "rgb24", "-f", "rawvideo", "-"]
    return args


def build_tile_audio_cmd(cfg: PipelineConfig, index: int) -> list[str] | None:
    """ffmpeg argv for one tile's audio → WAV on stdout, or None when muted.

    Each tile owns its ffmpeg→ffplay chain, so replacing a tile restarts only
    that tile's audio. The sound server mixes the per-tile streams, which is
    where ``--volume`` is applied.
    """
    if cfg.no_audio:
        return None
    tile = cfg.tiles[index]
    if not tile.metadata.has_audio:
        return None
    if cfg.audio_mode == "one":
        with_audio = [i for i, t in enumerate(cfg.tiles) if t.metadata.has_audio]
        # Fall back to the first audible tile if --audio-tile is out of range.
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


# -------- Keyboard mapping --------


def tile_index_for_key(key: int) -> int | None:
    """Map 1-9, 0, a-f to tile indexes 0-15."""
    if pygame.K_1 <= key <= pygame.K_9:
        return key - pygame.K_1
    if key == pygame.K_0:
        return 9
    if pygame.K_a <= key <= pygame.K_f:
        return key - pygame.K_a + 10
    return None


# -------- Pygame viewer --------


class VideoWindow:
    """Single pygame window compositing one decoder per tile.

    Per tile:
      video: ffmpeg → raw RGB frames → pygame surface
      audio: ffmpeg → WAV → ffplay -nodisp (the sound server mixes the tiles)

    The window and the other tiles survive a tile replacement.
    """

    def __init__(
        self, cfg: PipelineConfig, pool: list[Path], start_pct: float, end_pct: float
    ):
        self.cfg = cfg
        self.pool = pool
        self.start_pct = start_pct
        self.end_pct = end_pct
        self.banned: set[Path] = set()  # files that crashed a decoder
        self.running = False
        self.paused = False

        self.wall_w = cfg.cols * (cfg.cell_w + cfg.border) - cfg.border
        self.wall_h = cfg.rows * (cfg.cell_h + cfg.border) - cfg.border
        self.frame_size = cfg.cell_w * cfg.cell_h * 3

        pygame.display.init()
        pygame.key.set_repeat(300, 80)
        flags = pygame.FULLSCREEN | pygame.SCALED if cfg.fullscreen else 0
        self.screen = pygame.display.set_mode((self.wall_w, self.wall_h), flags)
        pygame.display.set_caption(TITLE)
        if cfg.fullscreen:
            pygame.mouse.set_visible(False)

        self.wall = pygame.Surface((self.wall_w, self.wall_h))
        # Where the wall was last blitted: (off_x, off_y, dst_w, dst_h). Used to
        # map a window-space mouse click back to a tile.
        self.dst_rect: tuple[int, int, int, int] | None = None

        self.playback = [TilePlayback() for _ in cfg.tiles]
        self._pending_seek: float | None = None
        self._clock = pygame.time.Clock()
        # Wall-clock frame budget, for tiles that have no audio to follow.
        self._last_tick: float | None = None
        self._frames_due = 0.0

    # ----- Tile lifecycle -----

    def _start_tile(self, index: int) -> None:
        """(Re)start one tile's decoders; every other tile keeps playing."""
        self._stop_playback(self.playback[index])
        tile = TilePlayback()
        self.playback[index] = tile
        self._spawn_video(tile, index)
        self._spawn_audio(tile, index)
        if self.paused:
            signal_proc(tile.video, signal.SIGSTOP)
            signal_proc(tile.audio_ffmpeg, signal.SIGSTOP)

    def _stop_playback(self, tile: TilePlayback) -> None:
        kill_proc(tile.video)
        kill_proc(tile.audio_ffmpeg)
        kill_proc(tile.audio_player)
        tile.video = tile.audio_ffmpeg = tile.audio_player = None

    def _spawn_video(self, tile: TilePlayback, index: int) -> None:
        tile.video = subprocess.Popen(
            build_tile_video_cmd(self.cfg, index),
            stdout=subprocess.PIPE,
            stderr=None if self.cfg.verbose else subprocess.DEVNULL,
        )
        self._widen_pipe(tile.video)

    def _widen_pipe(self, video: subprocess.Popen) -> None:
        """Grow the pipe so it can hold a few whole frames (default is 64 KB)."""
        if video.stdout is None:
            return
        fd = video.stdout.fileno()
        size = min(max(self.frame_size * 4, 1 << 20), pipe_max_size())
        set_pipe_size = getattr(fcntl, "F_SETPIPE_SZ", None)
        if set_pipe_size is not None:
            with contextlib.suppress(OSError, ValueError):
                fcntl.fcntl(fd, set_pipe_size, size)
        with contextlib.suppress(OSError, ValueError):
            os.set_blocking(fd, False)

    def _spawn_audio(self, tile: TilePlayback, index: int) -> None:
        command = build_tile_audio_cmd(self.cfg, index)
        if command is None:
            tile.video_ready = True
            return
        if shutil.which("ffplay") is None:
            if self.cfg.verbose:
                print("[warn] ffplay not found — audio disabled", file=sys.stderr)
            tile.video_ready = True
            return
        # ffplay first (it reads the pipe), held until this tile's first frame
        # arrives; the video then waits for its clock, see _video_gate_open.
        player = subprocess.Popen(
            [
                "ffplay", "-nodisp", "-vn",
                # info level is what emits the status line (the audio clock)
                "-loglevel", "info",
                "-autoexit", "-i", "pipe:0",
            ],
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        tile.audio_player = player
        # Hold ffplay until this tile has a frame to show (unless it died
        # instantly, in which case there is nothing to hold).
        tile.audio_held = player.poll() is None
        signal_proc(player, signal.SIGSTOP)
        threading.Thread(
            target=self._watch_audio_clock, args=(tile, player), daemon=True
        ).start()
        # ffmpeg second: it writes into ffplay's stdin.
        tile.audio_ffmpeg = subprocess.Popen(command, stdout=player.stdin)

    def _watch_audio_clock(self, tile: TilePlayback, player: subprocess.Popen) -> None:
        r"""Track ffplay's status line, which carries the audio clock.

        The line is ``\r``-terminated, so lines are split by hand.
        """
        if player.stderr is None:
            return
        fd = player.stderr.fileno()
        pending = b""
        while True:
            try:
                chunk = os.read(fd, 4096)
            except OSError:
                return
            if not chunk:
                return
            pending += chunk
            *lines, pending = re.split(rb"[\r\n]", pending)
            for line in lines:
                match = AUDIO_POS_RE.match(line)
                if match:
                    tile.audio_clock = float(match.group(1))
                    tile.audio_clock_at = time.monotonic()

    # ----- A/V sync -----

    def _video_gate_open(self, tile: TilePlayback) -> bool:
        """Hold a new tile's picture until its audio is actually playing.

        An audio seek can be slow (4K on NFS). Starting the video first would
        leave it permanently ahead of the soundtrack.
        """
        if len(tile.buffer) < self.frame_size:
            self._read_frames(tile, self.frame_size)
            if len(tile.buffer) < self.frame_size:
                return False
        if tile.audio_released_at is None:
            tile.audio_released_at = time.monotonic()
        if tile.audio_held:
            tile.audio_held = False
            signal_proc(tile.audio_player, signal.SIGCONT)
            return False
        if tile.clock() is None and not self._audio_timed_out(tile):
            return False
        tile.video_ready = True
        return True

    def _audio_timed_out(self, tile: TilePlayback) -> bool:
        released = tile.audio_released_at
        return released is not None and time.monotonic() - released > AUDIO_START_TIMEOUT

    def _pump_frames(self, due: int) -> None:
        """Advance every tile to the frame it should be showing *now*.

        A tile with live audio follows ffplay's audio clock and drops the
        frames in between, so a stall (tile restart, slow seek) cannot leave
        the picture permanently behind the soundtrack. A tile without audio is
        paced by the wall clock, ``due`` frames per call.
        """
        for tile in self.playback:
            if tile.video is None or tile.video.poll() is not None:
                continue
            if not tile.video_ready and not self._video_gate_open(tile):
                continue
            self._present_frame(tile, due)

    def _present_frame(self, tile: TilePlayback, due: int) -> None:
        clock = tile.clock()
        if clock is None:
            if due <= 0:
                return
            want = due * self.frame_size
        else:
            target = int(clock * VIDEO_FPS)
            if target < tile.frames_consumed:
                # The clock is behind the frame on screen (just after an audio
                # timeout, or clock jitter): wait for the audio to catch up.
                return
            want = (target - tile.frames_consumed + 1) * self.frame_size

        self._read_frames(tile, want)
        available = len(tile.buffer) // self.frame_size
        if available == 0:
            return
        # Show the newest frame available and drop the older ones (catch-up).
        tile.frames_consumed += available
        start = (available - 1) * self.frame_size
        frame = bytes(tile.buffer[start : start + self.frame_size])
        del tile.buffer[: available * self.frame_size]
        tile.surface = pygame.image.frombuffer(
            frame, (self.cfg.cell_w, self.cfg.cell_h), "RGB"
        )

    def _read_frames(self, tile: TilePlayback, want: int) -> None:
        """Read until ``want`` bytes are buffered or the pipe runs dry."""
        if tile.video is None or tile.video.stdout is None:
            return
        fd = tile.video.stdout.fileno()
        while len(tile.buffer) < want:
            try:
                # Bounded so a big catch-up cannot allocate a huge chunk at once.
                chunk = os.read(
                    fd, min(want - len(tile.buffer), self.frame_size * 8)
                )
            except (BlockingIOError, InterruptedError, OSError):
                return
            if not chunk:
                return
            tile.buffer += chunk

    # ----- Drawing -----

    def _composite(self) -> None:
        self.wall.fill((0, 0, 0))
        stride_w = self.cfg.cell_w + self.cfg.border
        stride_h = self.cfg.cell_h + self.cfg.border
        for i, tile in enumerate(self.playback):
            if tile.surface is None:
                continue
            col = i % self.cfg.cols
            row = i // self.cfg.cols
            self.wall.blit(tile.surface, (col * stride_w, row * stride_h))

    def _draw(self) -> None:
        """Show the wall, letterboxed so a resized window cannot stretch it.

        A tiling compositor (e.g. niri) may resize the window to an aspect
        ratio that no longer matches the grid, so scale to *fit* and centre.
        """
        win_w, win_h = self.screen.get_size()
        if (win_w, win_h) == (self.wall_w, self.wall_h):
            self.dst_rect = (0, 0, win_w, win_h)
            self.screen.blit(self.wall, (0, 0))
        else:
            scale = min(win_w / self.wall_w, win_h / self.wall_h)
            dst_w = max(1, int(self.wall_w * scale))
            dst_h = max(1, int(self.wall_h * scale))
            off_x = (win_w - dst_w) // 2
            off_y = (win_h - dst_h) // 2
            self.dst_rect = (off_x, off_y, dst_w, dst_h)
            self.screen.fill((0, 0, 0))
            self.screen.blit(
                pygame.transform.scale(self.wall, (dst_w, dst_h)), (off_x, off_y)
            )
        pygame.display.flip()

    # ----- Input -----

    def _handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return
            if event.type == pygame.KEYDOWN:
                self._handle_key(event.key, event.mod)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
                self._replace_tile_at(event.pos)

    def _handle_key(self, key: int, mod: int) -> None:
        if key == pygame.K_q:
            self.running = False
        elif key == pygame.K_SPACE:
            self._set_paused(not self.paused)
        elif key == pygame.K_r:
            self._replace_tile(random.randrange(len(self.cfg.tiles)))
        elif key in (pygame.K_LEFT, pygame.K_RIGHT):
            # Seek is applied to the next tile key pressed.
            step = 30.0 if mod & pygame.KMOD_SHIFT else 10.0
            self._pending_seek = step if key == pygame.K_RIGHT else -step
        else:
            index = tile_index_for_key(key)
            if index is None or index >= len(self.cfg.tiles):
                return
            if self._pending_seek is None:
                self._replace_tile(index)
            else:
                self._seek_tile(index, self._pending_seek)
                self._pending_seek = None

    def _replace_tile_at(self, position: tuple[int, int]) -> None:
        """Replace the tile under a window-space click, if it hit the wall."""
        if self.dst_rect is None:
            return
        off_x, off_y, dst_w, dst_h = self.dst_rect
        win_x, win_y = position
        if not (off_x <= win_x < off_x + dst_w and off_y <= win_y < off_y + dst_h):
            return  # the click landed in the letterbox bars
        # Invert the fit-and-centre transform from _draw.
        wall_x = (win_x - off_x) * self.wall_w / dst_w
        wall_y = (win_y - off_y) * self.wall_h / dst_h
        stride_w = self.cfg.cell_w + self.cfg.border
        stride_h = self.cfg.cell_h + self.cfg.border
        col = int(wall_x // stride_w)
        row = int(wall_y // stride_h)
        if not (0 <= col < self.cfg.cols and 0 <= row < self.cfg.rows):
            return
        index = row * self.cfg.cols + col
        if index < len(self.cfg.tiles):
            self._replace_tile(index)

    def _set_paused(self, paused: bool) -> None:
        self.paused = paused
        sig = signal.SIGSTOP if paused else signal.SIGCONT
        for tile in self.playback:
            signal_proc(tile.video, sig)
            signal_proc(tile.audio_ffmpeg, sig)
            if not tile.audio_held:  # held audio must stay stopped
                signal_proc(tile.audio_player, sig)

    # ----- Tile selection -----

    def _replace_tile(self, index: int, ban: Path | None = None) -> None:
        if ban is not None:
            self.banned.add(ban)
        path = self._pick_replacement(index)
        self.cfg.tiles[index] = TileState(
            path, random_seek(path, self.cfg.loop, self.start_pct, self.end_pct)
        )
        self._start_tile(index)

    def _pick_replacement(self, index: int) -> Path:
        """Prefer an unused file, then any different one, then the same one."""
        current = self.cfg.tiles[index].path
        in_use = {t.path for i, t in enumerate(self.cfg.tiles) if i != index}
        groups = (
            [p for p in self.pool if p not in in_use and p != current],
            [p for p in self.pool if p not in in_use],
            [p for p in self.pool if p != current],
        )
        for group in groups:
            random.shuffle(group)
            for path in group:
                if path not in self.banned and is_valid_video(path):
                    if self.cfg.verbose:
                        print(f"[info] replace: {path.name}", file=sys.stderr)
                    return path
        return current

    def _seek_tile(self, index: int, delta: float) -> None:
        tile = self.cfg.tiles[index]
        tile.seek = normalize_seek(tile.path, tile.seek + delta, self.cfg.loop)
        self._start_tile(index)

    # ----- Main loop -----

    def _due_frames(self) -> int:
        """How many 1/VIDEO_FPS frames of content the wall clock owes."""
        now = time.monotonic()
        if self._last_tick is None:
            self._last_tick = now
            self._frames_due = 1.0
        else:
            self._frames_due += (now - self._last_tick) * VIDEO_FPS
            self._last_tick = now
        due = int(self._frames_due)
        self._frames_due -= due
        return min(due, MAX_CATCHUP_FRAMES)

    def _recycle_dead_tiles(self) -> None:
        """Replace a tile whose decoder exited (EOF, corrupt file, …)."""
        for i, tile in enumerate(self.playback):
            if tile.video is not None and tile.video.poll() is not None:
                self._replace_tile(i, ban=self.cfg.tiles[i].path)
                return

    def run(self) -> None:
        """Start every tile and loop until the user quits."""
        print(f"Controls:\n{CONTROLS}\n", file=sys.stderr)
        self.running = True
        try:
            for i in range(len(self.playback)):
                self._start_tile(i)
            while self.running:
                self._handle_events()
                if not self.running:
                    break
                if self.paused:
                    self._last_tick = None
                else:
                    self._recycle_dead_tiles()
                    self._pump_frames(self._due_frames())
                    self._composite()
                    self._draw()
                self._clock.tick(VIDEO_FPS)
        finally:
            for tile in self.playback:
                self._stop_playback(tile)
            pygame.display.quit()
            pygame.quit()


# -------- CLI --------


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Single-window video wall with pygame viewer + audio.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Keyboard controls (pygame window):\n{CONTROLS}",
    )
    ap.add_argument("folder", type=Path, help="Folder with videos")

    grid = ap.add_argument_group("grid")
    grid.add_argument("--count", "-n", type=int, default=4, help="Number of tiles (1–16)")
    grid.add_argument(
        "--grid", metavar="NxM",
        help="Grid layout (e.g. 3x3, 2x4). Auto-computed from --count if omitted.",
    )
    grid.add_argument("--cell-width", type=int, default=480, help="Tile width in px")
    grid.add_argument("--cell-height", type=int, default=270, help="Tile height in px")
    grid.add_argument("--border", type=int, default=0, help="Black gap (px) between tiles")

    playback = ap.add_argument_group("playback")
    playback.add_argument("--loop", action="store_true", help="Loop each input")
    playback.add_argument(
        "--exts", default=",".join(DEFAULT_EXTS),
        help="Comma-separated extensions (default: %(default)s)",
    )
    playback.add_argument(
        "--recursive", action="store_true",
        help="Recursively scan subdirectories for videos",
    )
    playback.add_argument(
        "--start-pct", type=float, default=0.5, metavar="PCT",
        help="Start of random seek window (0–1, default: 0.5)",
    )
    playback.add_argument(
        "--end-pct", type=float, default=0.75, metavar="PCT",
        help="End of random seek window (0–1, default: 0.75)",
    )
    playback.add_argument("--seed", type=int, help="Random seed for deterministic runs")
    playback.add_argument("--verbose", "-v", action="store_true", help="Show ffmpeg logs")

    audio = ap.add_argument_group("audio")
    audio.add_argument("--no-audio", action="store_true", help="Disable audio playback")
    audio.add_argument(
        "--volume", type=float, default=0.5, help="Per-input pre-mix gain (default: 0.5)"
    )
    audio.add_argument(
        "--audio-mode", choices=["mix", "one"], default="mix",
        help="Mix all audio or use only one tile's audio",
    )
    audio.add_argument(
        "--audio-tile", type=int, default=0, metavar="TILE",
        help="Tile index (0-based) for --audio-mode=one",
    )
    audio.add_argument(
        "--audio-rate", type=int, default=48000, metavar="HZ",
        help="Sample rate in Hz (default: 48000)",
    )

    display = ap.add_argument_group("display")
    display.add_argument("--fullscreen", "-f", action="store_true", help="Fullscreen mode")
    display.add_argument(
        "--hwaccel", default="off", choices=["auto", "off", "cuda", "vaapi"],
        help="HW decode: off (default), auto, cuda, or vaapi",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    require_tools("ffmpeg", "ffprobe")

    if args.count < 1 or args.count > MAX_TILES:
        raise SystemExit(f"--count must be between 1 and {MAX_TILES}.")
    grid = parse_grid(args.grid)
    if grid and grid[0] * grid[1] < args.count:
        raise SystemExit(
            f"Grid {grid[0]}x{grid[1]} = {grid[0] * grid[1]} slots "
            f"is smaller than --count {args.count}."
        )
    if not args.folder.is_dir():
        raise SystemExit(f"{args.folder} is not a directory.")
    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        raise SystemExit(
            "No display detected (DISPLAY/WAYLAND_DISPLAY not set).\n"
            "Run from a desktop session with a display server running."
        )

    exts = tuple(s.strip().lower() for s in args.exts.split(",") if s.strip())
    pool = scan_folder(args.folder, exts, args.recursive)
    if len(pool) < args.count:
        raise SystemExit(
            f"Need at least {args.count} videos with extensions {exts}, "
            f"found {len(pool)} in {args.folder}."
        )
    tiles = [
        TileState(path, random_seek(path, args.loop, args.start_pct, args.end_pct))
        for path in pick_videos(pool, args.count, exts, args.seed)
    ]

    rows, cols = compute_grid(args.count, grid)
    hwaccel = choose_hwaccel(args.hwaccel)
    if args.verbose:
        print(f"[info] HW accel: {hwaccel.mode or 'software'}", file=sys.stderr)
        if hwaccel.vaapi_device:
            print(f"  (device {hwaccel.vaapi_device})", file=sys.stderr)
        print(f"[info] Grid: {cols}x{rows} = {args.count} tiles", file=sys.stderr)

    cfg = PipelineConfig(
        tiles=tiles,
        rows=rows,
        cols=cols,
        cell_w=args.cell_width,
        cell_h=args.cell_height,
        loop=args.loop,
        volume=args.volume,
        hwaccel=hwaccel,
        verbose=args.verbose,
        no_audio=args.no_audio,
        audio_mode=args.audio_mode,
        audio_tile=args.audio_tile,
        audio_rate=args.audio_rate,
        border=args.border,
        fullscreen=args.fullscreen,
    )
    try:
        VideoWindow(cfg, pool, args.start_pct, args.end_pct).run()
    except pygame.error as exc:
        raise SystemExit(f"Cannot open display: {exc}")


if __name__ == "__main__":
    main()
