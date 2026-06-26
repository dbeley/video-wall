#!/usr/bin/env python3
"""video-wall — Single-window video wall with pygame viewer + audio."""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import select
import shutil
import signal
import subprocess
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

# Suppress pygame compile-time warnings
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
warnings.filterwarnings("ignore", message=".*avx2.*", category=RuntimeWarning)

import pygame  # noqa: E402 — late import after env setup

MIN_FILE_SIZE = 50 * 1024  # 50 KB — skip torrent stubs / incomplete dls
DEFAULT_EXTS = (".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm")
GRID_PRESETS = {
    "2x2": (2, 2),
    "3x2": (2, 3),
    "2x3": (3, 2),
    "3x3": (3, 3),
    "4x3": (3, 4),
    "3x4": (4, 3),
    "4x4": (4, 4),
}
MAX_TILES = 16
TITLE = "Video Wall"
VIDEO_FPS = 30


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
    selected: list[Path] = []
    tried = 0
    for p in candidates:
        tried += 1
        if is_valid_video(p):
            selected.append(p)
            if len(selected) >= count:
                break
    if len(selected) < count:
        raise SystemExit(
            f"Need at least {count} valid videos with extensions {exts}, "
            f"only found {len(selected)}/{count} after checking {tried} candidates."
        )
    return selected


def compute_grid(n: int, rows: int | None, cols: int | None) -> tuple[int, int]:
    if rows and cols:
        return rows, cols
    if not cols:
        cols = math.ceil(math.sqrt(n))
    if not rows:
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


def _build_filter_graph(cfg: PipelineConfig) -> tuple[list[str], list[str]]:
    """Build video filter graph. Audio filters are NOT included (separate pipeline)."""
    flt: list[str] = []
    vouts: list[str] = []

    stride_w = cfg.cell_w + cfg.border
    stride_h = cfg.cell_h + cfg.border

    for i, tile in enumerate(cfg.tiles):
        ops: list[str] = []

        hw = cfg.hwaccel.mode
        if hw == "cuda":
            ops.append(
                f"scale_cuda={cfg.cell_w}:{cfg.cell_h}:"
                f"force_original_aspect_ratio=decrease"
            )
            ops.append("hwdownload")
            ops.append("format=nv12")
        elif hw == "vaapi":
            ops.append(
                f"scale_vaapi=w={cfg.cell_w}:h={cfg.cell_h}:"
                f"force_original_aspect_ratio=decrease"
            )
            ops.append("hwdownload")
            ops.append("format=nv12")
        else:
            ops.append(
                f"scale={cfg.cell_w}:{cfg.cell_h}:"
                f"force_original_aspect_ratio=decrease"
            )

        ops.append(f"pad={cfg.cell_w}:{cfg.cell_h}:(ow-iw)/2:(oh-ih)/2:black")
        ops.append("format=yuv420p")

        vouts.append(f"[v{i}]")
        flt.append(f"[{i}:v]{','.join(ops)}[v{i}]")

    layout_parts: list[str] = []
    for i in range(len(cfg.tiles)):
        col = i % cfg.cols
        row = i // cfg.cols
        layout_parts.append(f"{col * stride_w}_{row * stride_h}")

    layout = "|".join(layout_parts)
    flt.append(
        f"{''.join(vouts)}xstack=inputs={len(cfg.tiles)}:layout={layout},"
        f"fps={VIDEO_FPS}[V]"
    )

    return flt, vouts


def build_video_cmd(cfg: PipelineConfig) -> tuple[list[str], int, int]:
    """ffmpeg command: raw RGB frames to stdout (video only, no audio in filter)."""
    args = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "info" if cfg.verbose else "error",
    ]

    for k, v in cfg.hwaccel.global_opts.items():
        args += [k, v]

    for tile in cfg.tiles:
        meta = tile.metadata
        codec = (meta.video_codec or "").lower()
        wants_hw = cfg.hwaccel.enabled and codec not in {
            "mpeg4",
            "msmpeg4v3",
            "mpeg1video",
        }

        if cfg.loop:
            args += ["-stream_loop", "-1"]
        if wants_hw and cfg.hwaccel.mode == "cuda":
            args += ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
        elif wants_hw and cfg.hwaccel.mode == "vaapi":
            args += ["-hwaccel", "vaapi", "-hwaccel_output_format", "vaapi"]
        path_str = str(tile.path.absolute())
        args += ["-ss", f"{tile.seek:.3f}", "-i", path_str]

    flt, vouts = _build_filter_graph(cfg)
    maps = ["-map", "[V]"]

    total_w = cfg.cols * (cfg.cell_w + cfg.border) - cfg.border
    total_h = cfg.rows * (cfg.cell_h + cfg.border) - cfg.border

    args += ["-filter_complex", ";".join(flt), *maps, "-s", f"{total_w}x{total_h}"]
    args += [
        "-c:v", "rawvideo", "-pix_fmt", "rgb24",
        "-r", str(VIDEO_FPS),
        "-f", "image2pipe", "-",
    ]
    return args, total_w, total_h


def build_audio_cmd(cfg: PipelineConfig) -> list[str] | None:
    """Separate ffmpeg command: mix all audio streams → WAV to stdout.

    No per-stream volume filters (avoids naming conflicts).
    Amix weights parameter handles per-input gain directly.
    """
    with_audio = [i for i, t in enumerate(cfg.tiles) if t.metadata.has_audio]
    if not with_audio:
        return None

    args = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "info" if cfg.verbose else "error",
    ]
    for t in cfg.tiles:
        args += ["-ss", f"{t.seek:.3f}", "-i", str(t.path.absolute())]

    # Build audio filter: connect streams directly to amix (no volume filters)
    flt_parts: list[str] = []
    if cfg.audio_mode == "one":
        chosen = with_audio[0] if cfg.audio_tile is None else cfg.audio_tile
        if chosen not in with_audio:
            chosen = with_audio[0]
        flt_parts.append(
            f"[{chosen}:a]aresample=async=1:min_hard_comp=0.100[A]"
        )
    else:
        flt_parts.append(
            f"{''.join(f'[{i}:a]' for i in with_audio)}"
            f"amix=inputs={len(with_audio)}:"
            f"dropout_transition=200:weights={cfg.volume}"
            f"[Apre]"
        )
        flt_parts.append("[Apre]aresample=async=1:min_hard_comp=0.100[A]")

    args += ["-filter_complex", ";".join(flt_parts)]
    args += ["-map", "[A]", "-c:a", "pcm_s16le", "-ar", str(cfg.audio_rate)]
    args += ["-f", "wav", "-"]
    return args


# -------- Kill helper --------


def kill_proc(p: subprocess.Popen | None) -> None:
    if p is None:
        return
    try:
        p.terminate()
    except Exception:
        pass
    try:
        p.wait(timeout=0.3)
    except Exception:
        try:
            p.kill()
        except Exception:
            pass


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
        self.frame_size = width * height * 3
        self.verbose = verbose
        self.running = False
        self.paused = False

        self._ffmpeg_vid: subprocess.Popen | None = None
        self._ffmpeg_aud: subprocess.Popen | None = None
        self._ffplay_aud: subprocess.Popen | None = None
        self._pending_seek: float | None = None
        self._clock = pygame.time.Clock()

        # External state
        self._tiles: list[TileState] = []
        self._pool: list[Path] = []
        self._banned: set[Path] = set()  # paths that caused crashes
        self._start_pct = 0.5
        self._end_pct = 0.75
        self._loop = False
        self._make_cfg = None

    # ----- Audio -----

    def _start_audio(self, cmd: list[str] | None) -> None:
        """Start audio pipeline: ffmpeg → WAV pipe → ffplay.

        ffplay is started FIRST so the pipe has a reader before ffmpeg
        opens it. This avoids blocking on write.
        """
        kill_proc(self._ffmpeg_aud)
        kill_proc(self._ffplay_aud)
        self._ffmpeg_aud = None
        self._ffplay_aud = None

        if not cmd:
            return
        if not shutil.which("ffplay"):
            if self.verbose:
                print("[warn] ffplay not found — audio disabled", file=sys.stderr)
            return

        # ffplay first (blocks on pipe read until ffmpeg writes)
        self._ffplay_aud = subprocess.Popen(
            [
                "ffplay",
                "-nodisp", "-vn",
                "-loglevel", "info" if self.verbose else "error",
                "-autoexit",
                "-i", "pipe:0",
            ],
            stdin=subprocess.PIPE,
        )
        # ffmpeg second (writes to pipe → ffplay reads it)
        self._ffmpeg_aud = subprocess.Popen(cmd, stdout=self._ffplay_aud.stdin)

    def _stop_audio(self) -> None:
        kill_proc(self._ffmpeg_aud)
        kill_proc(self._ffplay_aud)
        self._ffmpeg_aud = None
        self._ffplay_aud = None

    # ----- Video -----

    def _start_video(self, cmd: list[str]) -> None:
        kill_proc(self._ffmpeg_vid)
        self._ffmpeg_vid = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=None if self.verbose else subprocess.DEVNULL,
        )

    def _stop_video(self) -> None:
        kill_proc(self._ffmpeg_vid)
        self._ffmpeg_vid = None

    def _stop_procs(self) -> None:
        self._stop_video()
        self._stop_audio()

    def _read_frame(self) -> bytes | None:
        """Read one raw RGB frame from the video pipe.

        Uses select with a 10ms timeout so we don't block event processing.
        Returns None if the pipe is dead, b"" if no data ready yet.
        If a partial frame is read (e.g. after SIGCONT resume), we block
        until the full frame arrives rather than returning None (which
        would trigger a spurious pipeline restart).
        """
        if self._ffmpeg_vid is None or self._ffmpeg_vid.stdout is None:
            return None
        if self._ffmpeg_vid.poll() is not None:
            return None

        fd = self._ffmpeg_vid.stdout.fileno()
        r, _, _ = select.select([fd], [], [], 0.01)
        if not r:
            return b""

        # Read full frame — handles partial pipe reads gracefully
        raw = self._ffmpeg_vid.stdout.read(self.frame_size)
        if len(raw) != self.frame_size:
            # Partial frame (possible after SIGCONT) — don't restart,
            # just wait for the next iteration
            return None if len(raw) == 0 else b""
        return raw

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
            for p in (self._ffmpeg_vid, self._ffmpeg_aud, self._ffplay_aud):
                if p is not None:
                    try:
                        os.kill(p.pid, signal.SIGSTOP)
                    except Exception:
                        pass
            self.paused = True
        else:
            for p in (self._ffmpeg_vid, self._ffmpeg_aud, self._ffplay_aud):
                if p is not None:
                    try:
                        os.kill(p.pid, signal.SIGCONT)
                    except Exception:
                        pass
            self.paused = False

    # ----- Pipeline lifecycle -----

    def _replace_tile(self, index: int, restart: bool = True,
                      ban: Path | None = None) -> None:
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
            new_path = _pick(
                [p for p in pool if p not in active_paths],
                "unused",
            )
        if new_path is None:
            new_path = _pick(
                [p for p in pool if p != current_path],
                "any different",
            )
        if new_path is None:
            if self.verbose:
                print("[info] replace: fallback to current", file=sys.stderr)
            new_path = current_path

        self._tiles[index] = TileState(
            path=new_path,
            seek=random_seek(new_path, self._loop, self._start_pct, self._end_pct),
        )
        if restart:
            self._restart_pipeline()

    def _seek_tile(self, index: int, delta: float) -> None:
        if not (0 <= index < len(self._tiles)):
            return
        t = self._tiles[index]
        t.seek = normalize_seek(t.path, t.seek + delta, self._loop)
        self._restart_pipeline()

    def _restart_pipeline(self) -> None:
        """Kill all and restart video + audio pipelines. Window stays open."""
        if self._make_cfg is None:
            return

        cfg = self._make_cfg(self._tiles)
        vid_cmd, w, h = build_video_cmd(cfg)
        aud_cmd = None if cfg.no_audio else build_audio_cmd(cfg)

        if (w, h) != (self.width, self.height):
            self.width = w
            self.height = h
            self.frame_size = w * h * 3
            flags = self.screen.get_flags()
            self.screen = pygame.display.set_mode((w, h), flags)

        # ffplay first → then ffmpeg video → then ffmpeg audio
        # Audio ffplay started BEFORE video+audio ffmpeg so pipes are ready
        if aud_cmd:
            self._start_audio(aud_cmd)

        self._start_video(vid_cmd)

    def run(
        self,
        tiles: list[TileState],
        pool: list[Path],
        start_pct: float,
        end_pct: float,
        loop: bool,
        make_cfg,
    ) -> None:
        """Main loop: read frames, display, handle pygame events."""
        self._tiles = tiles
        self._pool = pool
        self._start_pct = start_pct
        self._end_pct = end_pct
        self._loop = loop
        self._make_cfg = make_cfg
        self.running = True

        self._restart_pipeline()

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

                if self._ffmpeg_vid is not None and self._ffmpeg_vid.poll() is not None:
                    if not self.paused:
                        try:
                            idx = random.randrange(len(self._tiles))
                            self._replace_tile(idx, ban=self._tiles[idx].path)
                            continue
                        except Exception:
                            pass
                    self._restart_pipeline()

                frame_data = self._read_frame()

                if frame_data is None:
                    if not self.paused and self._ffmpeg_vid is not None:
                        idx = random.randrange(len(self._tiles))
                        self._replace_tile(idx, ban=self._tiles[idx].path)
                    continue

                if frame_data == b"":
                    self._clock.tick(VIDEO_FPS)
                    continue

                surface = pygame.image.frombuffer(
                    frame_data, (self.width, self.height), "RGB"
                )

                win_w, win_h = self.screen.get_size()
                if win_w != self.width or win_h != self.height:
                    surface = pygame.transform.scale(surface, (win_w, win_h))

                self.screen.blit(surface, (0, 0))
                pygame.display.flip()

                self._clock.tick(VIDEO_FPS)
        finally:
            self._stop_procs()
            pygame.display.quit()
            pygame.quit()


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
    ap.add_argument("--rows", type=int, help="Override number of rows")
    ap.add_argument("--cols", type=int, help="Override number of columns")
    ap.add_argument(
        "--preset",
        choices=list(GRID_PRESETS),
        help="Grid preset (e.g. 3x3). Overrides --rows/--cols.",
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
        "--no-recursive",
        action="store_true",
        help="Only include top-level files (default: recursive)",
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

    if args.preset:
        args.rows, args.cols = GRID_PRESETS[args.preset]

    if args.count < 1 or args.count > MAX_TILES:
        raise SystemExit(f"--count must be between 1 and {MAX_TILES}.")
    if not args.folder.is_dir():
        raise SystemExit(f"{args.folder} is not a directory.")

    exts = tuple(s.strip().lower() for s in args.exts.split(",") if s.strip())
    recursive = not args.no_recursive

    def _collect_videos(
        root: Path, extensions: tuple[str, ...], rec: bool
    ) -> list[Path]:
        if rec:
            return [
                p for p in root.rglob("*")
                if p.is_file() and p.suffix.lower() in extensions
            ]
        return [
            p for p in root.iterdir()
            if p.is_file() and p.suffix.lower() in extensions
        ]

    pool = _collect_videos(args.folder, exts, recursive)
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
    rows, cols = compute_grid(args.count, args.rows, args.cols)

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
