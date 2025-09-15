#!/usr/bin/env python3
import argparse
import math
import os
import random
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

# ---------------- Utilities ----------------

DEFAULT_EXTS = (".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm")


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


def random_seek_50_75_pct(path: Path) -> float:
    dur = ffprobe_duration(path) or 0.0
    if dur <= 0:
        return 0.0
    start = random.uniform(0.50, 0.75) * dur
    # leave a small buffer before the end so we don’t instantly EOF
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
    # choose a compact grid automatically
    if not cols:
        cols = math.ceil(math.sqrt(n))
    if not rows:
        rows = math.ceil(n / cols)
    return rows, cols


def make_xstack_layout(n: int, rows: int, cols: int, cell_w: int, cell_h: int) -> str:
    # layout is "x_y|x_y|..." pixel positions for each input after per-tile scaling/padding
    coords = []
    for i in range(n):
        r, c = divmod(i, cols)
        x = c * cell_w
        y = r * cell_h
        coords.append(f"{x}_{y}")
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
) -> list[str]:
    """
    Build a single ffplay command with N inputs, random -ss per input,
    filter_complex: per-tile scale+pad -> xstack, and audio amix.
    """

    n = len(files)
    # Inputs (with -ss for fast seek) and loop flags
    args = ["ffplay", "-v", "error", "-nostats"]
    # Better A/V sync in mosaics
    args += ["-fflags", "nobuffer", "-flags", "low_delay"]

    # Prepend per-input options (-ss must be before -i for fast seek)
    per_input_labels_v = []
    per_input_labels_a = []

    filter_parts = []

    for idx, f in enumerate(files):
        ss = random_seek_50_75_pct(f)
        if loop:
            args += ["-stream_loop", "-1"]
        args += ["-ss", f"{ss:.3f}", "-i", str(f)]

    # Build per-input video chains: scale to fit and pad to fill the cell
    for idx in range(n):
        vin = f"[{idx}:v]"
        vout = f"[v{idx}]"
        per_input_labels_v.append(vout)
        chain = (
            f"{vin}scale={cell_w}:{cell_h}:force_original_aspect_ratio=decrease,"
            f"pad={cell_w}:{cell_h}:(ow-iw)/2:(oh-ih)/2:black{vout}"
        )
        filter_parts.append(chain)

    # Per-input audio scaling (volume) before amix
    for idx in range(n):
        ain = f"[{idx}:a]"
        aout = f"[a{idx}]"
        per_input_labels_a.append(aout)
        # Some files may have no audio — volume filter will error if stream absent.
        # We guard by using "anullsrc" fallback via amerge with a silent source only when needed.
        # BUT ffplay doesn’t support conditional graphs; easiest is to ignore missing
        # and let amix handle fewer streams (we’ll map only existing audio streams).
        # We’ll include the label; ffplay will error if input lacks audio.
        # Workaround: use "aresample=async=1:min_hard_comp=0.100" to keep clocks happy.
        filter_parts.append(
            f"{ain}volume={volume},aresample=async=1:min_hard_comp=0.100{aout}"
        )

    # xstack layout
    layout = make_xstack_layout(n, rows, cols, cell_w, cell_h)
    total_w = cols * cell_w
    total_h = rows * cell_h

    # xstack requires exactly inputs=N
    vinputs = "".join(per_input_labels_v)
    filter_parts.append(f"{vinputs}xstack=inputs={n}:layout={layout}[V]")

    # amix on as many audio labels as we actually have
    # If some inputs truly lack audio, referencing them breaks. So detect via ffprobe now:
    have_audio = []
    for idx, f in enumerate(files):
        try:
            has_a = (
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
                        str(f),
                    ],
                    stderr=subprocess.DEVNULL,
                )
                .decode()
                .strip()
            )
            if has_a:
                have_audio.append(idx)
        except Exception:
            pass

    if have_audio:
        ainputs = "".join(f"[a{idx}]" for idx in have_audio)
        # dropout_transition smooths when one stream ends (if not looping)
        filter_parts.append(
            f"{ainputs}amix=inputs={len(have_audio)}:dropout_transition=200[aout]"
        )
        maps = ["-map", "[V]", "-map", "[aout]"]
    else:
        maps = ["-map", "[V]", "-an"]

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
    # Hint: ffplay understands keyboard controls in its window.
    # We’ll pause via signals from Python for reliability across WMs.

    return args


# ---------------- Controller ----------------


def launch_ffplay(cmd: list[str]) -> subprocess.Popen:
    return subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
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


def run_controller(
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
):
    if seed is not None:
        random.seed(seed)

    pool = [p for p in folder.iterdir() if p.suffix.lower() in exts]
    if len(pool) < count:
        raise SystemExit(
            f"Need at least {count} videos with extensions {exts} in {folder}"
        )

    files = pick_files(pool, count)
    rows, cols = grid_for_count(count, rows, cols)

    def build_and_launch(title="Video Wall"):
        cmd = build_ffplay_command(
            files=files,
            rows=rows,
            cols=cols,
            cell_w=cell_w,
            cell_h=cell_h,
            volume=volume,
            loop=loop,
            window_title=title,
        )
        return launch_ffplay(cmd)

    proc = build_and_launch()
    paused = False

    print(
        "\nControls:\n"
        "  SPACE  = pause/resume\n"
        "  r      = replace a random tile\n"
        "  1..8   = replace a specific tile (1-indexed)\n"
        "  q      = quit\n"
    )

    # raw key reading (POSIX)
    try:
        import termios, tty, select

        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        tty.setcbreak(fd)
        while True:
            # restart if died (e.g., user closed window)
            if proc.poll() is not None:
                # respawn with same files (new random seeks)
                proc = build_and_launch()

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
                    # pick a different file when possible
                    candidates = [p for p in pool if p not in files]
                    files[idx] = (
                        random.choice(candidates) if candidates else random.choice(pool)
                    )
                    # restart ffplay with new graph
                    try:
                        proc.terminate()
                        time.sleep(0.1)
                    except Exception:
                        pass
                    proc = build_and_launch()
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
                        proc = build_and_launch()
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


# ---------------- Main ----------------


def main():
    ap = argparse.ArgumentParser(
        description="Single-window video wall using ffplay (xstack + amix)"
    )
    ap.add_argument("folder", type=Path, help="Folder containing videos")
    ap.add_argument("-n", "--count", type=int, default=4, help="Number of tiles (4–8)")
    ap.add_argument("--rows", type=int, help="Rows (auto if omitted)")
    ap.add_argument("--cols", type=int, help="Columns (auto if omitted)")
    ap.add_argument(
        "--cell-width", type=int, default=640, help="Per-tile width (pixels)"
    )
    ap.add_argument(
        "--cell-height", type=int, default=360, help="Per-tile height (pixels)"
    )
    ap.add_argument(
        "--volume",
        type=float,
        default=0.6,
        help="Per-input volume multiplier (0.0–1.0+)",
    )
    ap.add_argument("--loop", action="store_true", help="Loop inputs")
    ap.add_argument(
        "--exts",
        default=",".join(DEFAULT_EXTS),
        help="Comma-separated extensions to include",
    )
    ap.add_argument("--seed", type=int, help="Random seed")
    args = ap.parse_args()

    if args.count < 4 or args.count > 8:
        raise SystemExit("Please choose --count between 4 and 8.")

    exts = tuple(x.strip().lower() for x in args.exts.split(",") if x.strip())
    if not args.folder.is_dir():
        raise SystemExit(f"{args.folder} is not a directory.")

    run_controller(
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
    )


if __name__ == "__main__":
    main()
