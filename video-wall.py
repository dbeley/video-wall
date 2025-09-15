#!/usr/bin/env python3
import argparse
import math
import os
import random
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_EXTS = (".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm")


def require_bin(name: str):
    if not shutil.which(name):
        raise SystemExit(
            f"'{name}' not found in PATH. Please install FFmpeg (ffplay/ffprobe)."
        )


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


def random_seek_50_75_pct(path: Path) -> float:
    dur = ffprobe_duration(path) or 0.0
    if dur <= 0:
        return 0.0
    start = random.uniform(0.50, 0.75) * dur
    return min(start, max(0.0, dur - 3.0))


def pick_files(pool: list[Path], n: int) -> list[Path]:
    if len(pool) < n:
        raise SystemExit(f"Need at least {n} video files.")
    files = pool[:]
    random.shuffle(files)
    return files[:n]


def grid_for_count(n: int, rows: int | None, cols: int | None):
    if rows and cols:
        return rows, cols
    if not cols:
        cols = math.ceil(math.sqrt(n))
    if not rows:
        rows = math.ceil(n / cols)
    return rows, cols


def make_xstack_layout(n: int, rows: int, cols: int, cell_w: int, cell_h: int) -> str:
    coords = []
    for i in range(n):
        r, c = divmod(i, cols)
        coords.append(f"{c * cell_w}_{r * cell_h}")
    return "|".join(coords)


def build_ffplay_command(
    files: list[Path],
    rows: int,
    cols: int,
    cell_w: int,
    cell_h: int,
    volume: float,
    loop: bool,
    window_title: str,
    verbose: bool,
) -> list[str]:
    n = len(files)

    args = ["ffplay"]
    # Show logs if verbose, else keep output quiet
    if verbose:
        args += ["-v", "info"]
    else:
        args += ["-v", "error", "-nostats"]

    # Lower latency flags (helpful for mosaics)
    args += ["-fflags", "nobuffer", "-flags", "low_delay"]

    # Per-input options must come before corresponding -i
    for f in files:
        if loop:
            args += ["-stream_loop", "-1"]
        args += [
            "-ss",
            f"{random_seek_50_75_pct(f):.3f}",
            "-i",
            # os.path.abspath(str(f)),
            str(f.resolve()),
        ]

    filter_parts = []
    vlabels = []
    # Ensure consistent pixel format per tile (avoids xstack pixfmt complaints)
    for idx in range(n):
        vin = f"[{idx}:v]"
        vout = f"[v{idx}]"
        vlabels.append(vout)
        filter_parts.append(
            f"{vin}"
            f"scale={cell_w}:{cell_h}:force_original_aspect_ratio=decrease,"
            f"pad={cell_w}:{cell_h}:(ow-iw)/2:(oh-ih)/2:black,"
            f"format=yuv420p{vout}"
        )

    # Only include audio chains for inputs that actually have audio
    audio_idxs = [i for i, f in enumerate(files) if ffprobe_has_audio(f)]
    alabels = []
    for idx in audio_idxs:
        ain = f"[{idx}:a]"
        aout = f"[a{idx}]"
        alabels.append(aout)
        filter_parts.append(
            f"{ain}volume={volume},aresample=async=1:min_hard_comp=0.100{aout}"
        )

    # xstack for video
    layout = make_xstack_layout(n, rows, cols, cell_w, cell_h)
    filter_parts.append(f"{''.join(vlabels)}xstack=inputs={n}:layout={layout}[V]")

    maps = ["-map", "[V]"]
    if alabels:
        # dropout_transition smooths if a stream ends (when not looping)
        filter_parts.append(
            f"{''.join(alabels)}amix=inputs={len(alabels)}:dropout_transition=200[aout]"
        )
        maps += ["-map", "[aout]"]
    else:
        maps += ["-an"]

    total_w = cols * cell_w
    total_h = rows * cell_h

    filter_complex = ";".join(filter_parts)
    args += [
        "-filter_complex",
        filter_complex,
        *maps,
        "-s",
        f"{total_w}x{total_h}",
        "-window_title",
        window_title,
    ]
    return args


def launch_ffplay(cmd: list[str], env: dict, verbose: bool) -> subprocess.Popen:
    stdout = None if verbose else subprocess.DEVNULL
    stderr = None if verbose else subprocess.DEVNULL
    return subprocess.Popen(
        cmd, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, env=env
    )


def pause_proc(p: subprocess.Popen):
    try:
        os.kill(p.pid, signal.SIGSTOP)
    except ProcessLookupError:
        pass


def resume_proc(p: subprocess.Popen):
    try:
        os.kill(p.pid, signal.SIGCONT)
    except ProcessLookupError:
        pass


def controller(
    folder: Path,
    count: int,
    rows: int | None,
    cols: int | None,
    cell_w: int,
    cell_h: int,
    volume: float,
    loop: bool,
    exts: tuple[str, ...],
    seed: int | None,
    sdl_driver: str | None,
    verbose: bool,
):
    if seed is not None:
        random.seed(seed)

    pool = [p for p in folder.rglob("*") if p.suffix.lower() in exts]
    if len(pool) < count:
        raise SystemExit(
            f"Need at least {count} videos with extensions {exts} in {folder}"
        )

    files = pick_files(pool, count)
    rows, cols = grid_for_count(count, rows, cols)

    env = os.environ.copy()
    if sdl_driver:
        env["SDL_VIDEODRIVER"] = sdl_driver  # e.g., 'wayland' or 'x11'

    def start_proc(title="Video Wall"):
        cmd = build_ffplay_command(
            files=files,
            rows=rows,
            cols=cols,
            cell_w=cell_w,
            cell_h=cell_h,
            volume=volume,
            loop=loop,
            window_title=title,
            verbose=verbose,
        )
        if verbose:
            print("FFPLAY CMD:\n", " ".join(cmd), flush=True)
        return launch_ffplay(cmd, env, verbose)

    proc = start_proc()
    paused = False

    print(
        "\nControls (focus the TERMINAL, not the ffplay window):\n"
        "  SPACE  = pause/resume\n"
        "  r      = replace a random tile\n"
        "  1..8   = replace specific tile (1-indexed)\n"
        "  q      = quit\n"
        "Tip: ffplay window itself also supports 'p' to pause, 'm' to mute, arrow keys to seek.\n"
    )

    # Raw key reading (POSIX)
    try:
        import termios, tty, select

        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        tty.setcbreak(fd)
        while True:
            if proc.poll() is not None:
                # Window was closed or ffplay crashed: restart with a fresh graph (new random seeks)
                proc = start_proc()
                if paused:
                    pause_proc(proc)

            dr, _, _ = select.select([sys.stdin], [], [], 0.1)
            if dr:
                ch = sys.stdin.read(1)
                if ch == "q":
                    break
                elif ch == " ":
                    if not paused:
                        pause_proc(proc)
                        paused = True
                    else:
                        resume_proc(proc)
                        paused = False
                elif ch == "r":
                    idx = random.randrange(len(files))
                    candidates = [p for p in pool if p not in files]
                    files[idx] = (
                        random.choice(candidates) if candidates else random.choice(pool)
                    )
                    try:
                        proc.terminate()
                        time.sleep(0.1)
                    except Exception:
                        pass
                    proc = start_proc()
                    if paused:
                        pause_proc(proc)
                elif ch.isdigit():
                    idx = int(ch) - 1
                    if 0 <= idx < len(files):
                        candidates = [p for p in pool if p not in files]
                        files[idx] = (
                            random.choice(candidates)
                            if candidates
                            else random.choice(pool)
                        )
                        try:
                            proc.terminate()
                            time.sleep(0.1)
                        except Exception:
                            pass
                        proc = start_proc()
                        if paused:
                            pause_proc(proc)
            time.sleep(0.02)
    finally:
        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        except Exception:
            pass
        try:
            proc.terminate()
            time.sleep(0.2)
            proc.kill()
        except Exception:
            pass


def main():
    require_bin("ffplay")
    require_bin("ffprobe")

    ap = argparse.ArgumentParser(
        description="Single-window video wall (ffplay xstack + amix)"
    )
    ap.add_argument("folder", type=Path, help="Folder containing videos")
    ap.add_argument("-n", "--count", type=int, default=4, help="Number of tiles (4–8)")
    ap.add_argument("--rows", type=int, help="Rows (auto if omitted)")
    ap.add_argument("--cols", type=int, help="Columns (auto if omitted)")
    ap.add_argument("--cell-width", type=int, default=640, help="Per-tile width")
    ap.add_argument("--cell-height", type=int, default=360, help="Per-tile height")
    ap.add_argument(
        "--volume", type=float, default=0.6, help="Per-input volume pre-mix (0.0–1.0+)"
    )
    ap.add_argument("--loop", action="store_true", help="Loop inputs")
    ap.add_argument(
        "--exts", default=",".join(DEFAULT_EXTS), help="Comma-separated extensions"
    )
    ap.add_argument("--seed", type=int, help="Random seed")
    ap.add_argument(
        "--sdl-driver", choices=["x11", "wayland"], help="Force SDL video driver"
    )
    ap.add_argument(
        "--verbose", action="store_true", help="Show ffplay command and logs"
    )
    args = ap.parse_args()

    if args.count < 4 or args.count > 8:
        raise SystemExit("Please choose --count between 4 and 8.")

    if not args.folder.is_dir():
        raise SystemExit(f"{args.folder} is not a directory.")

    exts = tuple(x.strip().lower() for x in args.exts.split(",") if x.strip())

    controller(
        folder=args.folder,
        count=args.count,
        rows=args.rows,
        cols=args.cols,
        cell_w=args.cell_width,
        cell_h=args.cell_height,
        volume=args.volume,
        loop=args.loop,
        exts=exts,
        seed=args.seed,
        sdl_driver=args.sdl_driver,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
