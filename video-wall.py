#!/usr/bin/env python3
import argparse, json, math, os, random, shutil, signal, subprocess, sys, time, glob
from dataclasses import dataclass
from pathlib import Path

DEFAULT_EXTS = (".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm")


@dataclass(frozen=True)
class VideoMeta:
    duration: float | None
    has_audio: bool


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

    out = run_cmd(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "format=duration,stream=index",
            "-of",
            "json",
            str(path),
        ]
    )

    duration: float | None = None
    has_audio = False
    if out:
        try:
            data = json.loads(out)
            fmt = data.get("format") or {}
            dur_str = fmt.get("duration")
            if dur_str:
                duration = float(dur_str)
            streams = data.get("streams") or []
            has_audio = len(streams) > 0
        except Exception as exc:  # pragma: no cover - fallback for malformed output
            print(
                f"[warn] failed to parse ffprobe output for {path}: {exc}",
                file=sys.stderr,
            )

    meta = VideoMeta(duration=duration, has_audio=has_audio)
    _metadata_cache[path] = meta
    return meta


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
    pool = pool[:]
    random.shuffle(pool)
    return pool[:n]


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
) -> list[str]:
    n = len(tiles)
    args = ["ffmpeg", "-hide_banner", "-loglevel", "info" if verbose else "error"]

    # Global HW opts (e.g., -vaapi_device /dev/dri/renderD128)
    for k, v in hw_global_opts.items():
        args += [k, v]

    # Per-input: hwaccel + seek + input
    for tile in tiles:
        path = tile.path
        if loop:
            args += ["-stream_loop", "-1"]
        if hwaccel_mode == "cuda":
            args += ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
        elif hwaccel_mode == "vaapi":
            args += ["-hwaccel", "vaapi", "-hwaccel_output_format", "vaapi"]
        # else: software decode
        args += ["-ss", f"{tile.seek:.3f}", "-i", str(path.resolve())]

    # Build filter graph
    flt, vouts, with_audio = [], [], []
    for i, tile in enumerate(tiles):
        ops: list[str] = []
        if hwaccel_mode == "cuda":
            ops.append(f"scale_cuda={cw}:{ch}:force_original_aspect_ratio=decrease")
            ops.append("hwdownload")
            ops.append("format=nv12")
        elif hwaccel_mode == "vaapi":
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
            flt.append(
                f"[{i}:a]volume={vol},aresample=async=1:min_hard_comp=0.100[a{i}]"
            )

    layout = layout_str(n, rows, cols, cw, ch)
    flt.append(f"{''.join(vouts)}xstack=inputs={n}:layout={layout}[V]")

    maps = ["-map", "[V]"]
    want_audio = not no_audio and bool(with_audio)
    if want_audio:
        flt.append(
            f"{''.join(f'[a{i}]' for i in with_audio)}amix=inputs={len(with_audio)}:dropout_transition=200[A]"
        )
        maps += ["-map", "[A]"]

    total_w, total_h = cols * cw, rows * ch

    args += ["-filter_complex", ";".join(flt), *maps, "-s", f"{total_w}x{total_h}"]

    # FAST path: rawvideo + PCM in a lightweight container (nut) over the pipe
    if fast:
        args += ["-c:v", "rawvideo", "-pix_fmt", "yuv420p"]
        if want_audio:
            args += ["-c:a", "pcm_s16le", "-ar", "48000"]
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
            args += ["-c:a", "aac", "-b:a", "192k", "-ar", "48000"]
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
    pool = [
        p for p in args.folder.iterdir() if p.is_file() and p.suffix.lower() in exts
    ]
    if len(pool) < args.count:
        raise SystemExit(f"Need at least {args.count} videos with extensions {exts}")
    initial_paths = pick(pool, args.count)
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
        active_paths = {t.path for j, t in enumerate(tiles) if j != index}
        candidates = [p for p in pool if p not in active_paths]
        new_path = random.choice(candidates) if candidates else random.choice(pool)
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
        import termios, tty, select

        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        tty.setcbreak(fd)

        while True:
            if ffplay.poll() is not None or ffmpeg.poll() is not None:
                update_tile_offsets(finalizing=True)
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
