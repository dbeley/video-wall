#!/usr/bin/env python3
import argparse, math, os, random, shutil, signal, subprocess, sys, time, glob
from pathlib import Path

DEFAULT_EXTS = (".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm")


# -------- Utils --------
def need(binname: str):
    if not shutil.which(binname):
        raise SystemExit(f"Missing '{binname}' in PATH.")


def run_cmd(argv: list[str]) -> str:
    try:
        return subprocess.check_output(argv, stderr=subprocess.STDOUT).decode(
            "utf-8", "ignore"
        )
    except Exception:
        return ""


def ffprobe_duration(path: Path) -> float | None:
    out = run_cmd(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nw=1:nk=1",
            str(path),
        ]
    )
    try:
        return float(out.strip()) if out.strip() else None
    except Exception:
        return None


def ffprobe_has_audio(path: Path) -> bool:
    out = run_cmd(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=index",
            "-of",
            "csv=p=0",
            str(path),
        ]
    )
    return bool(out.strip())


def random_seek_50_75(path: Path) -> float:
    dur = ffprobe_duration(path) or 0.0
    if dur <= 0:
        return 0.0
    start = random.uniform(0.50, 0.75) * dur
    return min(start, max(0.0, dur - 3.0))


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
    files: list[Path],
    rows: int,
    cols: int,
    cw: int,
    ch: int,
    loop: bool,
    vol: float,
    fast: bool,
    fps: int | None,
    hwaccel_mode: str | None,
    hw_global_opts: dict,
    verbose: bool,
    no_audio: bool,
) -> list[str]:
    n = len(files)
    args = ["ffmpeg", "-hide_banner", "-loglevel", "info" if verbose else "error"]

    # Global HW opts (e.g., -vaapi_device /dev/dri/renderD128)
    for k, v in hw_global_opts.items():
        args += [k, v]

    # Per-input: hwaccel + seek + input
    for f in files:
        if loop:
            args += ["-stream_loop", "-1"]
        if hwaccel_mode == "cuda":
            args += ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
        elif hwaccel_mode == "vaapi":
            args += ["-hwaccel", "vaapi", "-hwaccel_output_format", "vaapi"]
        # else: software decode
        args += ["-ss", f"{random_seek_50_75(f):.3f}", "-i", str(f.resolve())]

    # Build filter graph
    flt, vouts, with_audio = [], [], []
    for i, f in enumerate(files):
        pre = ""
        # If we decoded to hardware frames, download to system memory for CPU filters
        if hwaccel_mode in {"cuda", "vaapi"}:
            # nv12 is a safe intermediate for both
            pre = "hwdownload,format=nv12,"

        vouts.append(f"[v{i}]")
        flt.append(
            f"[{i}:v]{pre}"
            f"scale={cw}:{ch}:force_original_aspect_ratio=decrease,"
            f"pad={cw}:{ch}:(ow-iw)/2:(oh-ih)/2:black,format=yuv420p[v{i}]"
        )
        if not no_audio and ffprobe_has_audio(f):
            with_audio.append(i)
            flt.append(
                f"[{i}:a]volume={vol},aresample=async=1:min_hard_comp=0.100[a{i}]"
            )

    layout = layout_str(n, rows, cols, cw, ch)
    flt.append(f"{''.join(vouts)}xstack=inputs={n}:layout={layout}[V]")

    maps = ["-map", "[V]"]
    if not no_audio and with_audio:
        flt.append(
            f"{''.join(f'[a{i}]' for i in with_audio)}amix=inputs={len(with_audio)}:dropout_transition=200[A]"
        )
        maps += ["-map", "[A]"]
    else:
        maps += ["-an"]

    total_w, total_h = cols * cw, rows * ch

    args += ["-filter_complex", ";".join(flt), *maps, "-s", f"{total_w}x{total_h}"]

    # FPS cap → big CPU saver
    if fps:
        args += ["-r", str(fps)]

    # FAST path: rawvideo + PCM in a lightweight container (nut) over the pipe
    if fast:
        args += ["-c:v", "rawvideo", "-pix_fmt", "yuv420p"]
        if "-an" not in maps:
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
        if "-an" not in maps:
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
    ap.add_argument("-n", "--count", type=int, default=4, help="Number of tiles (4–8)")
    ap.add_argument("--rows", type=int)
    ap.add_argument("--cols", type=int)
    ap.add_argument("--cell-width", type=int, default=480)
    ap.add_argument("--cell-height", type=int, default=270)
    ap.add_argument("--volume", type=float, default=0.5, help="Per-input pre-mix gain")
    ap.add_argument("--loop", action="store_true", help="Loop inputs")
    ap.add_argument("--exts", default=",".join(DEFAULT_EXTS))
    ap.add_argument("--seed", type=int)
    ap.add_argument("--verbose", action="store_true")

    # Fast toggles
    ap.add_argument(
        "--fast", action="store_true", help="Use rawvideo+pcm and FPS cap (low CPU)"
    )
    ap.add_argument(
        "--fps", type=int, default=15, help="Output FPS cap (use with --fast)"
    )
    ap.add_argument("--no-audio", action="store_true", help="Disable audio mix")

    # HW accel: auto/off/cuda/vaapi (default auto)
    ap.add_argument(
        "--hwaccel",
        default="auto",
        choices=["auto", "off", "cuda", "vaapi"],
        help="Hardware decode mode (auto=detect)",
    )

    args = ap.parse_args()

    if args.count < 4 or args.count > 8:
        raise SystemExit("Choose --count between 4 and 8.")
    if not args.folder.is_dir():
        raise SystemExit(f"{args.folder} is not a directory.")
    if args.seed is not None:
        random.seed(args.seed)

    exts = tuple(s.strip().lower() for s in args.exts.split(",") if s.strip())
    pool = [p for p in args.folder.iterdir() if p.suffix.lower() in exts]
    if len(pool) < args.count:
        raise SystemExit(f"Need at least {args.count} videos with extensions {exts}")
    files = pick(pool, args.count)
    rows, cols = grid(args.count, args.rows, args.cols)

    # Choose HW accel
    hwaccel_mode, hw_global_opts = choose_hwaccel(args.hwaccel)
    if args.verbose:
        print(f"[info] HW accel: {hwaccel_mode or 'software'}", end="")
        if hwaccel_mode == "vaapi" and "-vaapi_device" in hw_global_opts:
            print(f" (device {hw_global_opts['-vaapi_device']})")
        else:
            print()

    def build_cmd():
        return build_ffmpeg_cmd(
            files=files,
            rows=rows,
            cols=cols,
            cw=args.cell_width,
            ch=args.cell_height,
            loop=args.loop,
            vol=args.volume,
            fast=args.fast,
            fps=args.fps if args.fast else None,
            hwaccel_mode=hwaccel_mode,
            hw_global_opts=hw_global_opts,
            verbose=args.verbose,
            no_audio=args.no_audio,
        )

    ffmpeg, ffplay = launch_pipeline(build_cmd(), "Video Wall (fast+HW)", args.verbose)
    paused = False

    print(
        "\nControls (focus terminal):\n"
        "  SPACE  = pause/resume\n"
        "  r      = replace a random tile\n"
        "  1..8   = replace a specific tile (1-indexed)\n"
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
                kill_proc(ffplay)
                kill_proc(ffmpeg)
                ffmpeg, ffplay = launch_pipeline(
                    build_cmd(), "Video Wall (fast+HW)", args.verbose
                )
                if paused:
                    for p in (ffmpeg, ffplay):
                        try:
                            os.kill(p.pid, signal.SIGSTOP)
                        except Exception:
                            pass

            dr, _, _ = select.select([sys.stdin], [], [], 0.1)
            if dr:
                ch = sys.stdin.read(1)
                if ch == "q":
                    break
                elif ch == " ":
                    if not paused:
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
                elif ch == "r":
                    i = random.randrange(len(files))
                    candidates = [p for p in pool if p not in files]
                    files[i] = (
                        random.choice(candidates) if candidates else random.choice(pool)
                    )
                    kill_proc(ffplay)
                    kill_proc(ffmpeg)
                    ffmpeg, ffplay = launch_pipeline(
                        build_cmd(), "Video Wall (fast+HW)", args.verbose
                    )
                    if paused:
                        for p in (ffmpeg, ffplay):
                            try:
                                os.kill(p.pid, signal.SIGSTOP)
                            except Exception:
                                pass
                elif ch.isdigit():
                    i = int(ch) - 1
                    if 0 <= i < len(files):
                        candidates = [p for p in pool if p not in files]
                        files[i] = (
                            random.choice(candidates)
                            if candidates
                            else random.choice(pool)
                        )
                        kill_proc(ffplay)
                        kill_proc(ffmpeg)
                        ffmpeg, ffplay = launch_pipeline(
                            build_cmd(), "Video Wall (fast+HW)", args.verbose
                        )
                        if paused:
                            for p in (ffmpeg, ffplay):
                                try:
                                    os.kill(p.pid, signal.SIGSTOP)
                                except Exception:
                                    pass
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
