# video-wall

Single-window video wall that plays multiple videos in a grid, using a **pygame** viewer with rawvideo pipe from ffmpeg.

## Features

- Grid of N videos (`--count` 1–16) with an optional explicit `--grid NxM` (e.g. `3x3`).
- Per-cell sizing via `--cell-width`/`--cell-height`; total size computed from grid.
- **Rawvideo pipe only** — no H.264 re-encode overhead.
- **Pygame viewer** replaces ffplay — keyboard shortcuts work when the **video window is focused**, not the terminal.
- **Window stays alive** on tile replacement/seeking — only ffmpeg restarts, no window flicker.
- Audio options: `--no-audio`, `--audio-mode {mix,one}`, `--audio-tile`, `--audio-rate`.
  Audio is played through a separate headless ffplay (`-nodisp -vn`) fed from a second ffmpeg instance.
- **A/V sync**: audio is held (SIGSTOP) until the first video frame is ready, then video
  is paced by a monotonic clock with frame-drop catch-up — so audio no longer starts
  seconds ahead of the slow-to-start grid.
- **Aspect-preserving display**: if the window/display aspect differs from the grid
  (e.g. a tiling compositor resizes the window), the wall is letterboxed with black bars
  instead of being stretched.
- Hardware decode: `--hwaccel {auto,off,cuda,vaapi}` with runtime-CUDA detection.
- Recursive file discovery with `--recursive` (top-level only by default).
- Randomized starting offset within `--start-pct`/`--end-pct` of duration.
- Deterministic runs with `--seed`.
- Explicit grid layout: `--grid 3x3`, `--grid 2x3`, etc.
- Tile border: `--border N` adds N-pixel black gap between tiles.
- Fullscreen: `--fullscreen` / `-f`.

## Requirements

- Runtime: `ffmpeg`, `ffprobe` in `PATH`. `ffplay` is only needed for audio (optional).
- Python 3.10+ with **pygame**.
- A running desktop session (X11 or Wayland).

Dev environment (optional):

- `direnv allow` to load nix shell and env.
- `nix-shell` to enter a dev shell with all dependencies.

## Usage

```
python3 video-wall.py <folder> [options]
```

Examples:

- Basic: `python3 video-wall.py ~/videos --count 9`
- With grid: `python3 video-wall.py ~/videos --grid 3x3`
- Fullscreen: `python3 video-wall.py ~/videos --grid 3x3 --fullscreen --border 2`
- Deterministic: `python3 video-wall.py ./clips --count 4 --seed 123`

### Options reference

| Option | Default | Description |
|--------|---------|-------------|
| `--count / -n` | 4 | Number of tiles (1–16) |
| `--grid` | auto | Grid layout `NxM` (e.g. `3x3`); auto-computed from `--count` |
| `--cell-width` | 480 | Tile width in px |
| `--cell-height` | 270 | Tile height in px |
| `--border` | 0 | Black gap (px) between tiles |
| `--volume` | 0.5 | Pre-mix gain per input |
| `--loop` | off | Loop each input |
| `--exts` | common types | Comma-separated extensions |
| `--recursive` | off | Scan subdirectories (default: top-level only) |
| `--start-pct` | 0.5 | Start of random seek window (0–1) |
| `--end-pct` | 0.75 | End of random seek window (0–1) |
| `--seed` | — | Random seed for deterministic runs |
| `--verbose / -v` | off | Show ffmpeg logs |
| `--no-audio` | off | Disable audio |
| `--audio-mode` | mix | `mix` all audio or `one` tile only |
| `--audio-tile` | 0 | Tile index (0-based) for `--audio-mode=one` |
| `--audio-rate` | 48000 | Audio sample rate in Hz |
| `--fullscreen / -f` | off | Fullscreen mode |
| `--hwaccel` | off | `auto`, `off`, `cuda`, `vaapi` |

## Keyboard Controls

Keys work when the **pygame video window** is focused (not the terminal).

| Key | Action |
|-----|--------|
| SPACE | pause/resume |
| r | replace a random tile |
| 1-9 | replace specific tile (1-indexed) |
| a-f | replace tiles 10–15 (0 = last tile) |
| f/b | seek forward/backward 10s, then press a tile |
| Shift+f/b | seek forward/backward 30s |
| q | quit |

## Architecture

```
ffmpeg (video decode + filter graph) ──rawvideo pipe──▶ pygame viewer (display + keys)
ffmpeg (audio mix only)              ──wav pipe──────▶ ffplay -nodisp -vn (headless audio)
```

- **Video pipeline**: Single ffmpeg decodes all input videos, builds an xstack filter graph, and outputs raw RGB24 frames via `image2pipe -` to stdout. The pygame viewer reads frames from the pipe using `select`-based non-blocking reads and displays them via `pygame.image.frombuffer`.
- **Audio pipeline**: A second ffmpeg runs a simplified audio-only filter graph and outputs PCM/WAV to a second pipe, consumed by a headless ffplay (`-nodisp -vn`). Both pipelines share the same seek positions and are restarted together on tile changes.
- **A/V sync**: ffplay is SIGSTOPped at spawn and resumed only once the first video frame is on screen, so audio cannot run ahead while the grid is still decoding. From then on, frames are presented against a monotonic deadline and late frames are dropped, keeping the wall locked to audio instead of drifting.
- **Window persistence**: On tile replace or seek, only ffmpeg subprocesses are killed and restarted. The pygame window stays open — no flicker.
- **Scaling**: Each frame is scaled to *fit* the current window/surface, centred with black bars, so a tiling compositor resizing the window cannot stretch the grid.

## Notes

- Requires a display server (X11 or Wayland). Checked at startup via `DISPLAY`/`WAYLAND_DISPLAY` env vars.
- The pygame AVX2 compile-time warning can be ignored (performance impact negligible).
- Previous `--fast` flag removed — always uses rawvideo pipe (lowest latency).

## Linting

```bash
ruff check .
```
