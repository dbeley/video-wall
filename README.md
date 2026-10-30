# video-wall

Single-window video wall that plays multiple videos in a grid, using a **pygame** viewer with rawvideo pipe from ffmpeg.

## Features

- Grid of N videos (`--count` 1–16) with an optional explicit `--grid NxM` (e.g. `3x3`).
- Per-cell sizing via `--cell-width`/`--cell-height`; total size computed from grid.
- **Rawvideo pipe only** — no H.264 re-encode overhead.
- **Pygame viewer** replaces ffplay — keyboard shortcuts work when the **video window is focused**, not the terminal.
- **One decoder per tile** — replacing or seeking a tile restarts only that tile; the
  other tiles keep playing from where they were (no whole-wall reset, no long stall).
- Audio options: `--no-audio`, `--audio-mode {mix,one}`, `--audio-tile`, `--audio-rate`.
  Each tile has its own `ffmpeg → ffplay` chain; the sound server mixes the streams
  (`--volume` applies per stream).
- **A/V sync**: audio is held (SIGSTOP) until the first video frame is ready, and from then
  on each tile's video follows ffplay's audio clock, dropping frames to stay on it — so audio
  never starts seconds ahead of the grid, and repeated tile changes can't make the video
  drift further behind the soundtrack.
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

## Keyboard & Mouse Controls

Keyboard and mouse events work when the **pygame video window** is focused (not the terminal).

| Key | Action |
|-----|--------|
| SPACE | pause/resume |
| r | replace a random tile |
| 1-9 | replace tiles 1–9 |
| 0 | replace tile 10 |
| a-f | replace tiles 11–16 |
| Left/Right | seek backward/forward 10s, then press a tile |
| Shift+Left/Right | seek backward/forward 30s |
| right-click | replace the tile under the cursor |
| q | quit |

## Architecture

```
per tile: ffmpeg (decode + scale/pad) ──rawvideo pipe──▶ pygame compositor ──▶ window
per tile: ffmpeg (audio only)         ──wav pipe──────▶ ffplay -nodisp -vn (mixed by sound server)
```

- **Per-tile video**: one ffmpeg per tile decodes that video, scales/pads it to the cell
  size (aspect preserved, black bars) and streams raw RGB24. pygame composites the tiles
  into the wall and scales the wall to the window.
- **Per-tile audio**: each tile runs its own ffmpeg→ffplay chain; the sound server mixes
  the streams. `--volume` is applied per stream and `--audio-mode one` starts a single chain.
- **Isolated tile changes**: replacing or seeking a tile kills and restarts only that tile's
  processes. Other tiles keep decoding and keep their playback position.
- **A/V sync**: each tile's ffplay is SIGSTOPped at spawn; once that tile's first video frame
  is ready, audio is released and the video is held until ffplay's clock actually starts.
  This absorbs the audio pipeline's seek/startup latency (large for 4K/NFS), so the video
  can't start first and stay ahead of the audio.
- **Audio is the master clock**: after that, each tile shows the frame whose position matches
  ffplay's audio clock, dropping frames to stay on it. Wall-clock pacing alone let a stall
  (tile restart, VAAPI hiccup) permanently lose video time — the video drifted further behind
  the audio with every tile change. Tiles without live audio (muted/no track/ffplay ended)
  fall back to the wall clock.
- **Presentation**: the wall clock decides how many 1/30 s frames an audio-less tile should
  advance per iteration; older frames are dropped so a tile stays current (never races,
  never freezes).
- **Scaling**: the composited wall is scaled to *fit* the current window, centred with black
  bars, so a tiling compositor resizing the window cannot stretch the grid.

## Notes

- Requires a display server (X11 or Wayland). Checked at startup via `DISPLAY`/`WAYLAND_DISPLAY` env vars.
- The pygame AVX2 compile-time warning can be ignored (performance impact negligible).
- Previous `--fast` flag removed — always uses rawvideo pipe (lowest latency).

## Linting

```bash
ruff check .
```
