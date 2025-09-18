# video-wall

Single-window video wall that plays multiple videos in a grid, with low-latency fast mode and optional hardware-accelerated decode.

## Features
- Grid of N videos (`--count` 4–16) with optional `--rows`/`--cols`.
- Per-cell sizing via `--cell-width`/`--cell-height` (total size is computed from grid).
- Fast path (`--fast`) using rawvideo+pcm piped to `ffplay` for minimal latency and CPU.
- Audio options: `--no-audio`, `--audio-mode {mix,one}`, `--audio-tile`, `--audio-rate`.
- Hardware decode: `--hwaccel {off,auto,cuda,vaapi}` with auto VAAPI device discovery.
- Recursive file discovery by default; disable via `--no-recursive`.
- Randomized starting offset within `--start_percentage/--end_percentage` of duration.
- Deterministic runs with `--seed`.

## Requirements
- Runtime: `ffmpeg`, `ffprobe`, `ffplay` available in `PATH`.
- Python 3.10+.

Dev environment (optional):
- `direnv allow` to load nix shell and env.
- `nix-shell` to enter a dev shell with Python, ruff, and ffmpeg tools.

## Usage
```
python3 video-wall.py <folder> [options]
```

Example:
- Balanced preview: `python3 video-wall.py ~/videos --count 9 --fast --hwaccel auto`
- Audio-light: `python3 video-wall.py ~/videos --count 6 --fast --audio-mode one --audio-rate 44100`
- Deterministic: `python3 video-wall.py ./clips --count 4 --seed 123`

Key options:
- `--count` N (4–16) — number of tiles.
- `--rows/--cols` — override auto grid.
- `--cell-width/--cell-height` — per-tile dimensions.
- `--fast` — rawvideo+pcm for low latency and CPU.
- `--no-audio` — disable audio entirely.
- `--audio-mode mix|one` — mix all audio or use one tile only.
- `--audio-tile` — 1-indexed tile to use when `--audio-mode one`.
- `--audio-rate` — audio sample rate (e.g., 48000, 44100, 32000).
- `--hwaccel off|auto|cuda|vaapi` — hardware decode mode (default off). Use `auto` to detect CUDA/VAAPI when possible.
- `--exts` — comma-separated list of extensions to include (defaults to common video types).
- `--recursive/--no-recursive` — include subdirectories (default recursive).
- `--start_percentage/--end_percentage` — random start window as a fraction of duration.
- `--seed` — reproducible selection and seeking.
- `--verbose` — show ffmpeg/ffplay logs.
- `--filter-threads` — threads for filter graph (advanced tuning).

Supported extensions (by default): `.mp4,.mov,.mkv,.avi,.m4v,.webm`.

## Keyboard Controls
- SPACE: pause/resume.
- r: replace a random tile.
- 1..9: replace a specific tile (1-indexed).
- f/F + number: seek forward 10s/30s for the chosen tile.
- b/B + number: seek backward 10s/30s for the chosen tile.
- q: quit.

## Performance Tips
- Prefer `--fast` for lowest latency and CPU.
- Enable hardware decode with `--hwaccel auto` when available (CUDA preferred, VAAPI supported).
- Reduce audio cost via `--audio-mode one` or `--no-audio` when visuals are the focus.
- Lower `--audio-rate` to `44100` or `32000` to save CPU.
- Set `--filter-threads 2` (or higher) on multi-core systems.
- Reduce `--count` or decrease `--cell-width/--cell-height` on constrained hardware.

## Notes on Hardware Acceleration
- `--hwaccel auto` attempts CUDA first, then VAAPI if a writable `/dev/dri/renderD*` device is present; otherwise falls back to software.
- VAAPI runs with auto `-vaapi_device` injection when the device is detected.

## Linting
Use `ruff check .` to lint the codebase.
