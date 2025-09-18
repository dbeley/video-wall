#!/usr/bin/env python3
import argparse
import glob
import json
import math
import os
import random
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

DEFAULT_EXTS = (".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm")


@dataclass(frozen=True)
class VideoMeta:
    duration: float | None
    has_audio: bool
    has_video: bool
    video_codec: str | None


_metadata_cache: dict[Path, VideoMeta] = {}


# -------- Utils --------
def need(binname: str):
    if not shutil.which(binname):
        raise SystemExit(f"Missing '{binname}' in PATH.")


def run_cmd(argv: list[str]) -> str:
    try:
        return subprocess.check_output(argv, stderr=subprocess.STDOUT).decode(
            "utf-8", "ignore"
        )
    except subprocess.CalledProcessError as exc:
        output = exc.output.decode("utf-8", "ignore") if exc.output else ""
        print(
            f"[warn] command {' '.join(argv)} failed with code {exc.returncode}: {output}",
            file=sys.stderr,
        )
        return ""
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"[warn] command {' '.join(argv)} failed: {exc}", file=sys.stderr)
        return ""


def get_metadata(path: Path) -> VideoMeta:
    meta = _metadata_cache.get(path)
    if meta is not None:
        return meta

    # Query all streams + format once; parse audio/video presence and duration
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
    if out:
        try:
            data = json.loads(out)
            fmt = data.get("format") or {}
            dur_str = fmt.get("duration")
            if dur_str:
                duration = float(dur_str)
            streams = data.get("streams") or []
            vcodec: str | None = None
            for s in streams:
                ctype = (s.get("codec_type") or "").lower()
                if ctype == "audio":
                    has_audio = True
                if ctype == "video":
                    has_video = True
                    if vcodec is None:
                        name = s.get("codec_name")
                        if isinstance(name, str) and name:
                            vcodec = name.lower()
        except Exception as exc:  # pragma: no cover - fallback for malformed output
            print(
                f"[warn] failed to parse ffprobe output for {path}: {exc}",
                file=sys.stderr,
            )

    # best effort: if we didn't see a video stream, codec stays None
    meta = VideoMeta(
        duration=duration, has_audio=has_audio, has_video=has_video, video_codec=locals().get("vcodec")
    )
    _metadata_cache[path] = meta
    return meta


def is_valid_video(path: Path) -> bool:
    """Cheap, cached validity check: returns True if ffprobe sees a video stream."""
    try:
        return get_metadata(path).has_video
    except Exception:
        return False


def normalize_seek(path: Path, offset: float, loop: bool) -> float:
    meta = get_metadata(path)
    dur = meta.duration or 0.0
    if dur > 0:
        if loop:
            offset = offset % dur
        max_seek = max(0.0, dur - 3.0)
        offset = min(offset, max_seek)
    return max(0.0, offset)


def random_seek(
    path: Path, loop: bool, start_percentage: float, end_percentage: float
) -> float:
    dur = get_metadata(path).duration or 0.0
    if dur <= 0:
        return 0.0
    start = random.uniform(start_percentage, end_percentage) * dur
    return normalize_seek(path, start, loop)


@dataclass
class TileState:
    path: Path
    seek: float

    @property
    def metadata(self) -> VideoMeta:
        return get_metadata(self.path)


def pick(pool: list[Path], n: int) -> list[Path]:
    if len(pool) < n:
        raise SystemExit(f"Need at least {n} videos.")
    # `random.sample` keeps selection unbiased and avoids mutating the source list
    return random.sample(pool, n)


def grid(n: int, rows: int | None, cols: int | None):
    if rows and cols:
        return rows, cols
    if not cols:
        cols = math.ceil(math.sqrt(n))
    if not rows:
        rows = math.ceil(n / cols)
    return rows, cols


def layout_str(n: int, rows: int, cols: int, cw: int, ch: int) -> str:
    return "|".join(f"{(i % cols) * cw}_{(i // cols) * ch}" for i in range(n))


# -------- HW Accel auto-detect --------
def list_hwaccels() -> set[str]:
    out = run_cmd(["ffmpeg", "-hide_banner", "-hwaccels"])
    # Output looks like:
    # Hardware acceleration methods:
    # vdpau
    # vaapi
    # cuda
    # ...
    accels = set()
    for line in out.splitlines():
        s = line.strip().lower()
        if s and not s.startswith("hardware acceleration"):
            accels.add(s)
    return accels


def find_vaapi_device() -> str | None:
    # Try common render nodes
    for path in sorted(glob.glob("/dev/dri/renderD*")):
        if os.access(path, os.R_OK | os.W_OK):
            return path
    return None


def choose_hwaccel(pref: str | None = None) -> tuple[str | None, dict]:
    """
    Returns (accel, extra_global_opts_dict)
    accel in {'cuda','vaapi',None}
    """
    # user override
    if pref:
        p = pref.lower()
        if p in {"off", "none"}:
            return None, {}
        if p in {"cuda", "vaapi"}:
            if p == "vaapi":
                dev = find_vaapi_device()
                if dev:
                    return "vaapi", {"-vaapi_device": dev}
                # fallback off if no device
                return None, {}
            return "cuda", {}
        if p == "auto":
            pass  # fall through to auto
        else:
            # unknown → treat as off
            return None, {}

    accels = list_hwaccels()
    # Prefer CUDA if present
    if "cuda" in accels:
        return "cuda", {}
    # Then VAAPI if render node exists
    if "vaapi" in accels:
        dev = find_vaapi_device()
        if dev:
            return "vaapi", {"-vaapi_device": dev}
    # Could add qsv/videotoolbox here if you need
    return None, {}


# -------- Command builder --------
def build_ffmpeg_cmd(
    tiles: list[TileState],
    rows: int,
    cols: int,
    cw: int,
    ch: int,
    loop: bool,
    vol: float,
    fast: bool,
    hwaccel_mode: str | None,
    hw_global_opts: dict,
    verbose: bool,
    no_audio: bool,
    start_percentage: float,
    end_percentage: float,
    audio_mode: str,
    audio_tile_index: int | None,
    audio_rate: int,
    filter_threads: int | None,
) -> list[str]:
    n = len(tiles)
    args = ["ffmpeg", "-hide_banner", "-loglevel", "info" if verbose else "error"]

    # Global HW opts (e.g., -vaapi_device /dev/dri/renderD128)
    for k, v in hw_global_opts.items():
        args += [k, v]
    if filter_threads and filter_threads > 0:
        args += ["-filter_threads", str(filter_threads)]

    # Per-input: hwaccel + seek + input (decide per tile based on codec)
    per_input_hw: list[str | None] = []
    for tile in tiles:
        path = tile.path
        meta = tile.metadata
        # Avoid forcing hwaccel for codecs known to be problematic (e.g., mpeg4)
        codec = (meta.video_codec or "").lower()
        wants_hw = hwaccel_mode in {"cuda", "vaapi"} and codec not in {"mpeg4", "msmpeg4v3", "mpeg1video"}

        if loop:
            args += ["-stream_loop", "-1"]
        if wants_hw and hwaccel_mode == "cuda":
            per_input_hw.append("cuda")
            args += ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
        elif wants_hw and hwaccel_mode == "vaapi":
            per_input_hw.append("vaapi")
            args += ["-hwaccel", "vaapi", "-hwaccel_output_format", "vaapi"]
        else:
            per_input_hw.append(None)
        args += ["-ss", f"{tile.seek:.3f}", "-i", str(path.resolve())]

    # Build filter graph
    flt, vouts, with_audio = [], [], []
    for i, tile in enumerate(tiles):
        ops: list[str] = []
        if per_input_hw[i] == "cuda":
            ops.append(f"scale_cuda={cw}:{ch}:force_original_aspect_ratio=decrease")
            ops.append("hwdownload")
            ops.append("format=nv12")
        elif per_input_hw[i] == "vaapi":
            ops.append(
                f"scale_vaapi=w={cw}:h={ch}:force_original_aspect_ratio=decrease"
            )
            ops.append("hwdownload")
            ops.append("format=nv12")
        else:
            ops.append(f"scale={cw}:{ch}:force_original_aspect_ratio=decrease")

        ops.append(f"pad={cw}:{ch}:(ow-iw)/2:(oh-ih)/2:black")
        ops.append("format=yuv420p")
        vouts.append(f"[v{i}]")
        flt.append(f"[{i}:v]{','.join(ops)}[v{i}]")

        meta = tile.metadata
        if not no_audio and meta.has_audio:
            with_audio.append(i)
            # Lighten CPU: apply volume per stream, resample once after mix
            flt.append(f"[{i}:a]volume={vol}[a{i}]")

    layout = layout_str(n, rows, cols, cw, ch)
    flt.append(f"{''.join(vouts)}xstack=inputs={n}:layout={layout}[V]")

    maps = ["-map", "[V]"]
    want_audio = not no_audio and bool(with_audio)
    if want_audio:
        if audio_mode == "one":
            # Choose a single tile's audio (1-indexed external, 0-indexed internal)
            chosen = None
            if audio_tile_index is not None and 0 <= audio_tile_index < len(tiles):
                if audio_tile_index in with_audio:
                    chosen = audio_tile_index
            if chosen is None:
                chosen = with_audio[0]
            flt.append(f"[a{chosen}]aresample=async=1:min_hard_comp=0.100[A]")
            maps += ["-map", "[A]"]
        else:
            # Mix all available audio streams, then resample once
            flt.append(
                f"{''.join(f'[a{i}]' for i in with_audio)}amix=inputs={len(with_audio)}:dropout_transition=200[Apre]"
            )
            flt.append("[Apre]aresample=async=1:min_hard_comp=0.100[A]")
            maps += ["-map", "[A]"]

    total_w, total_h = cols * cw, rows * ch

    args += ["-filter_complex", ";".join(flt), *maps, "-s", f"{total_w}x{total_h}"]

    # FAST path: rawvideo + PCM in a lightweight container (nut) over the pipe
    if fast:
        args += ["-c:v", "rawvideo", "-pix_fmt", "yuv420p"]
        if want_audio:
            args += ["-c:a", "pcm_s16le", "-ar", str(audio_rate)]
        else:
            args += ["-an"]
        args += ["-f", "nut", "-"]
    else:
        # Heavier (encode H.264 + AAC) — not recommended if CPU is tight
        args += [
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-tune",
            "zerolatency",
            "-pix_fmt",
            "yuv420p",
        ]
        if want_audio:
            args += ["-c:a", "aac", "-b:a", "192k", "-ar", str(audio_rate)]
        else:
            args += ["-an"]
        args += ["-f", "matroska", "-"]

    return args


def launch_pipeline(ffmpeg_cmd: list[str], title: str, verbose: bool):
    ffplay_cmd = [
        "ffplay",
        "-loglevel",
        "info" if verbose else "error",
        "-fflags",
        "nobuffer",
        "-flags",
        "low_delay",
        "-autoexit",
        "-window_title",
        title,
        "-i",
        "pipe:0",
    ]
    ffmpeg = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE)
    ffplay = subprocess.Popen(ffplay_cmd, stdin=ffmpeg.stdout)
    return ffmpeg, ffplay


def kill_proc(p):
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


# -------- Main controller --------
def main():
    need("ffmpeg")
    need("ffprobe")
    need("ffplay")

    ap = argparse.ArgumentParser(
        description="Single-window video wall (fast mode + auto HW accel)"
    )
    ap.add_argument("folder", type=Path, help="Folder with videos")
    ap.add_argument("-n", "--count", type=int, default=4, help="Number of tiles (4–16)")
    ap.add_argument("--rows", type=int)
    ap.add_argument("--cols", type=int)
    ap.add_argument("--cell-width", type=int, default=480)
    ap.add_argument("--cell-height", type=int, default=270)
    ap.add_argument("--volume", type=float, default=0.5, help="Per-input pre-mix gain")
    ap.add_argument("--loop", action="store_true", help="Loop inputs")
    ap.add_argument("--exts", default=",".join(DEFAULT_EXTS))
    ap.add_argument(
        "-R",
        "--recursive",
        dest="recursive",
        action="store_true",
        default=True,
        help="Include videos from subdirectories (default)",
    )
    ap.add_argument(
        "--no-recursive",
        dest="recursive",
        action="store_false",
        help="Only include videos in the top-level folder",
    )
    ap.add_argument("--seed", type=int)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument(
        "--start_percentage",
        type=float,
        default=0.5,
        help="Start of the duration randomization",
    )
    ap.add_argument(
        "--end_percentage",
        type=float,
        default=0.75,
        help="End of the duration randomization",
    )

    # Fast toggles
    ap.add_argument("--fast", action="store_true", help="Use rawvideo+pcm (low CPU)")
    ap.add_argument("--no-audio", action="store_true", help="Disable audio mix")
    ap.add_argument(
        "--audio-mode",
        choices=["mix", "one"],
        default="mix",
        help="Mix all audio or use one tile's audio",
    )
    ap.add_argument(
        "--audio-tile",
        type=int,
        help="When --audio-mode one, use this 1-indexed tile for audio",
    )
    ap.add_argument(
        "--audio-rate",
        type=int,
        default=48000,
        help="Audio sample rate (Hz), e.g. 44100 or 32000",
    )
    ap.add_argument(
        "--filter-threads",
        type=int,
        help="Threads for filter graph (advanced)",
    )

    # HW accel: auto/off/cuda/vaapi (default auto)
    ap.add_argument(
        "--hwaccel",
        default="off",
        choices=["auto", "off", "cuda", "vaapi"],
        help="Hardware decode mode (auto=off)",
    )

    args = ap.parse_args()

    if args.count < 4 or args.count > 16:
        raise SystemExit("Choose --count between 4 and 16.")
    if not args.folder.is_dir():
        raise SystemExit(f"{args.folder} is not a directory.")
    if args.seed is not None:
        random.seed(args.seed)

    exts = tuple(s.strip().lower() for s in args.exts.split(",") if s.strip())

    def collect_videos(root: Path, extensions: tuple[str, ...], recursive: bool) -> list[Path]:
        if recursive:
            # Recursively collect files with allowed extensions
            return [
                p
                for p in root.rglob("*")
                if p.is_file() and p.suffix.lower() in extensions
            ]
        else:
            # Only immediate children
            return [
                p
                for p in root.iterdir()
                if p.is_file() and p.suffix.lower() in extensions
            ]

    pool = collect_videos(args.folder, exts, args.recursive)
    if len(pool) < args.count:
        raise SystemExit(f"Need at least {args.count} videos with extensions {exts}")

    # Only probe validity for the videos we actually attempt to use on the grid.
    # Keep picking candidates until we have enough valid ones or we exhaust the pool.
    def select_valid_paths(pool_paths: list[Path], needed: int) -> list[Path]:
        if needed <= 0:
            return []
        candidates = list(pool_paths)
        random.shuffle(candidates)
        selected: list[Path] = []
        tried = 0
        for p in candidates:
            tried += 1
            if is_valid_video(p):
                selected.append(p)
                if len(selected) >= needed:
                    break
        if len(selected) < needed:
            raise SystemExit(
                f"Need at least {needed} valid videos with extensions {exts}"
            )
        if args.verbose:
            print(
                f"[info] initial select: {len(selected)}/{needed} valid "
                f"after checking {tried}/{len(candidates)} candidates",
                file=sys.stderr,
            )
        return selected

    initial_paths = select_valid_paths(pool, args.count)
    tiles = [
        TileState(
            path=path,
            seek=random_seek(
                path, args.loop, args.start_percentage, args.end_percentage
            ),
        )
        for path in initial_paths
    ]
    rows, cols = grid(args.count, args.rows, args.cols)

    # Choose HW accel
    hwaccel_mode, hw_global_opts = choose_hwaccel(args.hwaccel)
    if args.verbose:
        print(f"[info] HW accel: {hwaccel_mode or 'software'}", end="")
        if hwaccel_mode == "vaapi" and "-vaapi_device" in hw_global_opts:
            print(f" (device {hw_global_opts['-vaapi_device']})")
        else:
            print()

    paused = False
    pending_seek: float | None = None
    pipeline_start = time.monotonic()
    ffmpeg = ffplay = None

    def build_cmd():
        return build_ffmpeg_cmd(
            tiles=tiles,
            rows=rows,
            cols=cols,
            cw=args.cell_width,
            ch=args.cell_height,
            loop=args.loop,
            vol=args.volume,
            fast=args.fast,
            hwaccel_mode=hwaccel_mode,
            hw_global_opts=hw_global_opts,
            verbose=args.verbose,
            no_audio=args.no_audio,
            start_percentage=args.start_percentage,
            end_percentage=args.end_percentage,
            audio_mode=args.audio_mode,
            audio_tile_index=(args.audio_tile - 1) if args.audio_tile else None,
            audio_rate=args.audio_rate,
            filter_threads=args.filter_threads,
        )

    def update_tile_offsets(finalizing: bool = False):
        nonlocal pipeline_start
        if paused:
            return
        now = time.monotonic()
        elapsed = now - pipeline_start
        if elapsed <= 0:
            return
        for tile in tiles:
            raw_seek = tile.seek + elapsed
            if finalizing and not args.loop:
                meta = tile.metadata
                dur = meta.duration or 0.0
                if dur > 0 and raw_seek >= dur - 0.5:
                    tile.seek = random_seek(
                        tile.path, args.loop, args.start_percentage, args.end_percentage
                    )
                    continue
            tile.seek = normalize_seek(tile.path, raw_seek, args.loop)
        pipeline_start = now

    def restart_pipeline():
        nonlocal ffmpeg, ffplay, pipeline_start
        kill_proc(ffplay)
        kill_proc(ffmpeg)
        ffmpeg, ffplay = launch_pipeline(
            build_cmd(), "Video Wall (fast+HW)", args.verbose
        )
        pipeline_start = time.monotonic()
        if paused:
            for proc in (ffmpeg, ffplay):
                if proc is None:
                    continue
                try:
                    os.kill(proc.pid, signal.SIGSTOP)
                except Exception:
                    pass

    def replace_tile(index: int):
        if not (0 <= index < len(tiles)):
            return
        update_tile_offsets()
        # Prefer videos that are not currently in use and differ from the tile being
        # replaced so we don't appear to "reroll" to the same clip repeatedly.
        active_paths = {t.path for j, t in enumerate(tiles) if j != index}
        current_path = tiles[index].path

        def choose(candidates: list[Path], reason: str) -> Path | None:
            # Prefer a random valid candidate; skip files that have no video stream.
            if not candidates:
                if args.verbose:
                    print(f"[info] replace: {reason}: no candidates", file=sys.stderr)
                return None
            shuffled = list(candidates)
            random.shuffle(shuffled)
            tried = 0
            for p in shuffled:
                tried += 1
                if is_valid_video(p):
                    if args.verbose:
                        print(
                            f"[info] replace: {reason}: picked after "
                            f"{tried}/{len(shuffled)} checked",
                            file=sys.stderr,
                        )
                    return p
            if args.verbose:
                print(
                    f"[info] replace: {reason}: no valid among {len(shuffled)} candidates",
                    file=sys.stderr,
                )
            return None

        new_path = None
        # First try videos not shown anywhere and different from the current tile.
        new_path = choose(
            [
                p
                for p in pool
                if p not in active_paths and p != current_path
            ],
            "unused+different",
        )
        if new_path is None:
            # Next allow reusing the current tile's path if it's the only unused option.
            new_path = choose([p for p in pool if p not in active_paths], "unused")
        if new_path is None:
            # Finally fall back to any different video, even if it's active elsewhere.
            new_path = choose([p for p in pool if p != current_path], "any different")
        if new_path is None:
            # Worst case: only option is to keep the same video.
            if args.verbose:
                print("[info] replace: fallback to current clip", file=sys.stderr)
            new_path = current_path
        tiles[index] = TileState(
            path=new_path,
            seek=random_seek(
                new_path, args.loop, args.start_percentage, args.end_percentage
            ),
        )
        restart_pipeline()

    def seek_tile(index: int, delta: float):
        if not (0 <= index < len(tiles)):
            return
        update_tile_offsets()
        tile = tiles[index]
        tile.seek = normalize_seek(tile.path, tile.seek + delta, args.loop)
        restart_pipeline()

    restart_pipeline()

    print(
        "\nControls (focus terminal):\n"
        "  SPACE  = pause/resume\n"
        "  r      = replace a random tile\n"
        "  1..9   = replace a specific tile (1-indexed)\n"
        "  f/F + #= seek forward 10s/30s for tile #\n"
        "  b/B + #= seek backward 10s/30s for tile #\n"
        "  q      = quit\n"
    )

    # POSIX raw key reading
    try:
        import select
        import termios
        import tty

        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        tty.setcbreak(fd)

        while True:
            if ffplay.poll() is not None or ffmpeg.poll() is not None:
                # If the pipeline died very quickly, assume a bad input and swap one
                ran_for = max(0.0, time.monotonic() - pipeline_start)
                update_tile_offsets(finalizing=True)
                if ran_for < 1.0:
                    try:
                        replace_tile(random.randrange(len(tiles)))
                        # replace_tile() restarts the pipeline already
                        continue
                    except Exception:
                        # Fallback to plain restart if replacement fails for any reason
                        pass
                restart_pipeline()

            dr, _, _ = select.select([sys.stdin], [], [], 0.1)
            if dr:
                ch = sys.stdin.read(1)
                if pending_seek is not None:
                    if ch.isdigit():
                        idx = int(ch) - 1
                        if 0 <= idx < len(tiles):
                            seek_tile(idx, pending_seek)
                        pending_seek = None
                        continue
                    else:
                        pending_seek = None

                if ch == "q":
                    break
                elif ch == " ":
                    if not paused:
                        update_tile_offsets()
                        for p in (ffmpeg, ffplay):
                            try:
                                os.kill(p.pid, signal.SIGSTOP)
                            except Exception:
                                pass
                        paused = True
                    else:
                        for p in (ffmpeg, ffplay):
                            try:
                                os.kill(p.pid, signal.SIGCONT)
                            except Exception:
                                pass
                        paused = False
                        pipeline_start = time.monotonic()
                elif ch == "r":
                    replace_tile(random.randrange(len(tiles)))
                elif ch in {"f", "F", "b", "B"}:
                    step = 10.0 if ch in {"f", "b"} else 30.0
                    if ch in {"b", "B"}:
                        step = -step
                    pending_seek = step
                elif ch.isdigit():
                    i = int(ch) - 1
                    if 0 <= i < len(tiles):
                        replace_tile(i)
            time.sleep(0.02)
    finally:
        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        except Exception:
            pass
        kill_proc(ffplay)
        kill_proc(ffmpeg)


if __name__ == "__main__":
    main()
