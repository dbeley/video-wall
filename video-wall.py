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
    return "|".join(f"{(i % cols) * cw}_{(i // cols) * ch}" for i in range(n))


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
    fps: int | None,
    hwaccel: str | None,
    verbose: bool,
    no_audio: bool,
) -> list[str]:
    n = len(files)
    args = ["ffmpeg", "-hide_banner", "-loglevel", "info" if verbose else "error"]

    # Optional hardware accel for decoding (helps when not using rawvideo; may still help)
    if hwaccel:
        args += ["-hwaccel", hwaccel]
        # best-effort; not forcing output_format to keep filters happy

    # inputs with random -ss per file
    for f in files:
        if loop:
            args += ["-stream_loop", "-1"]
        args += ["-ss", f"{random_seek_50_75(f):.3f}", "-i", str(f.resolve())]

    # build filter graph
    flt, vouts, with_audio = [], [], []
    for i, f in enumerate(files):
        vouts.append(f"[v{i}]")
        flt.append(
            f"[{i}:v]scale={cw}:{ch}:force_original_aspect_ratio=decrease,"
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

    # FPS cap (big CPU saver)
    if fps:
        args += ["-r", str(fps)]

    # --- Fast path: raw video + raw PCM audio in a lightweight container (nut) ---
    args += ["-c:v", vcodec, "-pix_fmt", "yuv420p"]
    if "-an" not in maps:
        args += ["-c:a", acodec, "-ar", "48000"]
    else:
        args += ["-an"]

    # Output through a container to carry both rawvideo + pcm over one pipe
    args += ["-f", muxer, "-"]
    return args


def launch_pipeline(ffmpeg_cmd: list[str], viewer_title: str, verbose: bool):
    # Single-window playback via ffplay reading from stdin
    ffplay_cmd = [
        "ffplay",
        "-loglevel",
        "info" if verbose else "error",
        "-autoexit",
        "-window_title",
        viewer_title,
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


def main():
    need("ffmpeg")
    need("ffprobe")
    need("ffplay")

    ap = argparse.ArgumentParser(description="Single-window video wall (fast mode)")
    ap.add_argument("folder", type=Path, help="Folder with videos")
    ap.add_argument("-n", "--count", type=int, default=4, help="Number of tiles (4–8)")
    ap.add_argument("--rows", type=int)
    ap.add_argument("--cols", type=int)
    ap.add_argument("--cell-width", type=int, default=480)  # smaller tiles by default
    ap.add_argument("--cell-height", type=int, default=270)
    ap.add_argument("--volume", type=float, default=0.5, help="Per-input pre-mix gain")
    ap.add_argument("--loop", action="store_true", help="Loop inputs")
    ap.add_argument("--exts", default=",".join(DEFAULT_EXTS))
    ap.add_argument("--seed", type=int)
    ap.add_argument("--verbose", action="store_true")

    # FAST toggles
    ap.add_argument(
        "--fast",
        action="store_true",
        help="Enable rawvideo+pcm, fps cap (very low CPU)",
    )
    ap.add_argument(
        "--fps", type=int, default=30, help="Output FPS cap (works best with --fast)"
    )
    ap.add_argument("--no-audio", action="store_true", help="Disable audio entirely")

    # HW accel hint (may or may not help depending on drivers/pipeline)
    ap.add_argument(
        "--hwaccel", choices=["vaapi", "cuda", "vdpau", "dxva2", "videotoolbox"]
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
        raise SystemExit(f"Need at least {args.count} videos with {exts}")

    files = pick(pool, args.count)
    rows, cols = grid(args.count, args.rows, args.cols)

    # Choose codecs/container for fast vs normal
    if args.fast:
        vcodec, acodec, muxer = "rawvideo", "pcm_s16le", "nut"
    else:
        # (still works, but CPU heavier)
        vcodec, acodec, muxer = "libx264", "aac", "matroska"

    def build_cmd():
        return build_ffmpeg_cmd(
            files=files,
            rows=rows,
            cols=cols,
            cw=args.cell_width,
            ch=args.cell_height,
            loop=args.loop,
            vol=args.volume,
            vcodec=vcodec,
            acodec=acodec,
            muxer=muxer,
            fps=args.fps if args.fast else None,
            hwaccel=args.hwaccel,
            verbose=args.verbose,
            no_audio=args.no_audio,
        )

    ffmpeg, ffplay = launch_pipeline(build_cmd(), "Video Wall (fast)", args.verbose)
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
            # if pipeline died (closed window), restart
            if ffplay.poll() is not None or ffmpeg.poll() is not None:
                kill_proc(ffplay)
                kill_proc(ffmpeg)
                ffmpeg, ffplay = launch_pipeline(
                    build_cmd(), "Video Wall (fast)", args.verbose
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
                        build_cmd(), "Video Wall (fast)", args.verbose
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
                            build_cmd(), "Video Wall (fast)", args.verbose
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
