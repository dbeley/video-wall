#!/usr/bin/env python3
import argparse, math, os, random, shutil, signal, subprocess, sys, time
from pathlib import Path

DEFAULT_EXTS = (".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm")


def need(binname: str):
    if not shutil.which(binname):
        raise SystemExit(f"Missing '{binname}' in PATH.")


def ffprobe_duration(path: Path) -> float | None:
    try:
        out = (
            subprocess.check_output(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=nw=1:nk=1",
                    str(path),
                ],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
        return float(out) if out else None
    except Exception:
        return None


def ffprobe_has_audio(path: Path) -> bool:
    try:
        out = (
            subprocess.check_output(
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
                ],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
        return bool(out)
    except Exception:
        return False


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
    parts = []
    for i in range(n):
        r, c = divmod(i, cols)
        parts.append(f"{c * cw}_{r * ch}")
    return "|".join(parts)


def build_ffmpeg_cmd(
    files: list[Path],
    rows: int,
    cols: int,
    cw: int,
    ch: int,
    loop: bool,
    vol: float,
    vcodec: str,
    acodec: str,
    muxer: str,
    verbose: bool,
) -> list[str]:
    n = len(files)
    args = ["ffmpeg"]
    args += ["-hide_banner"]
    args += ["-loglevel", "info" if verbose else "error"]

    # inputs with random -ss per file; use absolute paths
    for f in files:
        if loop:
            args += ["-stream_loop", "-1"]
        args += ["-ss", f"{random_seek_50_75(f):.3f}", "-i", str(f.resolve())]

    # per-input chains
    flt = []
    vouts = []
    have_a = []
    for i, f in enumerate(files):
        vouts.append(f"[v{i}]")
        flt.append(
            f"[{i}:v]scale={cw}:{ch}:force_original_aspect_ratio=decrease,"
            f"pad={cw}:{ch}:(ow-iw)/2:(oh-ih)/2:black,format=yuv420p[v{i}]"
        )
        if ffprobe_has_audio(f):
            have_a.append(i)
            flt.append(
                f"[{i}:a]volume={vol},aresample=async=1:min_hard_comp=0.100[a{i}]"
            )

    # xstack video
    layout = layout_str(n, rows, cols, cw, ch)
    flt.append(f"{''.join(vouts)}xstack=inputs={n}:layout={layout}[V]")

    maps = ["-map", "[V]"]
    if have_a:
        aconcat = "".join(f"[a{i}]" for i in have_a)
        flt.append(f"{aconcat}amix=inputs={len(have_a)}:dropout_transition=200[A]")
        maps += ["-map", "[A]"]

    total_w, total_h = cols * cw, rows * ch

    args += [
        "-filter_complex",
        ";".join(flt),
        *maps,
        "-s",
        f"{total_w}x{total_h}",
        "-c:v",
        vcodec,
        "-pix_fmt",
        "yuv420p",
    ]
    if have_a:
        args += ["-c:a", acodec, "-ar", "48000"]
    else:
        args += ["-an"]
    # low-latency encodes
    if vcodec.startswith("libx264"):
        args += ["-preset", "veryfast", "-tune", "zerolatency"]
    if acodec == "aac":
        args += ["-b:a", "192k"]

    # write to stdout as a single multiplexed stream
    args += ["-f", muxer, "-"]
    return args


def launch_pipeline(ffmpeg_cmd: list[str], ffplay_title: str, verbose: bool):
    # ffplay reads single stream from stdin
    ffplay_cmd = [
        "ffplay",
        "-loglevel",
        "info" if verbose else "error",
        "-autoexit",  # exit if producer stops
        "-window_title",
        ffplay_title,
        "-i",
        "pipe:0",
    ]
    ffmpeg = subprocess.Popen(
        ffmpeg_cmd,
        stdout=subprocess.PIPE,
        stderr=None if verbose else subprocess.DEVNULL,
    )
    ffplay = subprocess.Popen(
        ffplay_cmd,
        stdin=ffmpeg.stdout,
        stdout=None,
        stderr=None if verbose else subprocess.DEVNULL,
    )
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


def main():
    need("ffmpeg")
    need("ffprobe")
    need("ffplay")
    ap = argparse.ArgumentParser(
        description="Single-window video wall (ffmpeg xstack+amix -> ffplay)"
    )
    ap.add_argument("folder", type=Path, help="Folder with videos")
    ap.add_argument("-n", "--count", type=int, default=4, help="Number of tiles (4–8)")
    ap.add_argument("--rows", type=int)
    ap.add_argument("--cols", type=int)
    ap.add_argument("--cell-width", type=int, default=640)
    ap.add_argument("--cell-height", type=int, default=360)
    ap.add_argument(
        "--volume", type=float, default=0.6, help="Per-input pre-mix volume"
    )
    ap.add_argument("--loop", action="store_true", help="Loop inputs")
    ap.add_argument("--exts", default=",".join(DEFAULT_EXTS))
    ap.add_argument(
        "--vcodec",
        default="libx264",
        help="Output video codec (libx264, h264_videotoolbox, h264_nvenc, etc.)",
    )
    ap.add_argument(
        "--acodec", default="aac", help="Output audio codec (aac, libopus, etc.)"
    )
    ap.add_argument(
        "--muxer", default="matroska", help="Container for the pipe (matroska, nut)"
    )
    ap.add_argument("--seed", type=int, help="Random seed")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.count < 4 or args.count > 8:
        raise SystemExit("Choose --count between 4 and 8.")
    if not args.folder.is_dir():
        raise SystemExit(f"{args.folder} is not a directory.")

    if args.seed is not None:
        random.seed(args.seed)

    exts = tuple(s.strip().lower() for s in args.exts.split(",") if s.strip())
    pool = [p for p in args.folder.rglob("*") if p.suffix.lower() in exts]
    files = pick(pool, args.count)
    rows, cols = grid(args.count, args.rows, args.cols)

    def build():
        return build_ffmpeg_cmd(
            files=files,
            rows=rows,
            cols=cols,
            cw=args.cell_width,
            ch=args.cell_height,
            loop=args.loop,
            vol=args.volume,
            vcodec=args.vcodec,
            acodec=args.acodec,
            muxer=args.muxer,
            verbose=args.verbose,
        )

    ffmpeg, ffplay = launch_pipeline(build(), "Video Wall", args.verbose)
    paused = False

    print(
        "\nControls (focus terminal):\n"
        "  SPACE  = pause/resume\n"
        "  r      = replace a random tile\n"
        "  1..8   = replace a specific tile (1-indexed)\n"
        "  q      = quit\n"
    )

    # raw POSIX key reading
    try:
        import termios, tty, select

        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        tty.setcbreak(fd)
        while True:
            # if either process died, respawn both
            if ffplay.poll() is not None or ffmpeg.poll() is not None:
                kill_proc(ffplay)
                kill_proc(ffmpeg)
                ffmpeg, ffplay = launch_pipeline(build(), "Video Wall", args.verbose)
                if paused:
                    # pause both via SIGSTOP
                    try:
                        os.kill(ffmpeg.pid, signal.SIGSTOP)
                    except Exception:
                        pass
                    try:
                        os.kill(ffplay.pid, signal.SIGSTOP)
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
                    # replace a random slot and rebuild pipeline
                    i = random.randrange(len(files))
                    candidates = [p for p in pool if p not in files]
                    files[i] = (
                        random.choice(candidates) if candidates else random.choice(pool)
                    )
                    kill_proc(ffplay)
                    kill_proc(ffmpeg)
                    ffmpeg, ffplay = launch_pipeline(
                        build(), "Video Wall", args.verbose
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
                            build(), "Video Wall", args.verbose
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
