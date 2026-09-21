# Repository Guidelines

## Project Structure & Module Organization
- Root-level Python tool: `video-wall.py` (main entrypoint).
- Docs: `README.md`.
- Dev environment: `shell.nix`, `.envrc` (direnv+nix).
- No packages or tests folder yet; keep single-file CLI unless a refactor is discussed.

## Build, Test, and Development Commands
- `direnv allow` — load nix shell and env (requires direnv).
- `nix-shell` — enter dev shell with Python, ruff, ffmpeg, and pygame.
- `python3 video-wall.py <folder>` — run video wall.
  - Example: `python3 video-wall.py ~/videos --count 9 --grid 3x3`.
- `ruff check .` — lint Python code.

Runtime dependencies: `ffmpeg`, `ffprobe` in `PATH`. `ffplay` needed only for audio.

## Current Feature Set (CLI)
- Grid layout with `--count` 1–16; optional `--grid NxM` (e.g. `--grid 3x3`).
- Cell sizing via `--cell-width/--cell-height`; auto total size from grid.
- Always rawvideo+pcm pipe to pygame viewer (no H.264 encode overhead).
- `--border` adds black gaps between tiles.
- `--fullscreen` / `-f` fullscreen mode.
- Audio control: `--no-audio`, `--audio-mode {mix,one}`, `--audio-tile`, `--audio-rate`.
  Audio played via separate headless ffplay (`-nodisp -vn`).
- Hardware decode `--hwaccel {off,auto,cuda,vaapi}` with runtime CUDA lib check.
- Input selection: top-level only by default (`--recursive` to scan subdirectories).
- Randomized start within `--start-pct/--end-pct` of duration.
- Deterministic runs with `--seed`.
- Verbose logging with `--verbose`.

## Runtime Keyboard Controls
Keys work when the **pygame window** is focused (not the terminal).

- SPACE: pause/resume pipeline.
- r: replace a random tile.
- 1..9: replace a specific tile (1-indexed).
- a..f: replace tiles 10-15.
- 0: replace the last tile (tile 16).
- f/b: seek +10s/-10s, then press a tile number.
- Shift+f/b: seek +30s/-30s.
- q: quit.

## Architecture

```
per tile: ffmpeg (decode + scale/pad) ──rawvideo pipe──▶ pygame compositor ──▶ window
per tile: ffmpeg (audio)              ──wav pipe─────▶ ffplay -nodisp -vn (sound server mixes)
```

### VideoWindow
- `VideoWindow` owns the pygame window and one ffmpeg per tile, plus pygame compositing.
- Per-tile stdout pipes are non-blocking; `_pump_frames(due)` assembles whole frames and
  keeps the newest (drops stale) so a tile stays current.
- `pygame.image.frombuffer()` converts each cell's raw RGB bytes to a surface; `_composite()`
  blits cells into the wall surface; `_blit_frame()` scales the wall to the window.
- Keyboard events processed via `pygame.event.get()` — works when SDL window is focused.
- Replacing/seeking a tile calls `_start_tile(i)` only, so other tiles are untouched.
- Each tile's ffplay is SIGSTOPped at spawn and released on that tile's first frame
  (`_release_tile_audio`).

### Models
- `VideoMeta`, `HwAccelConfig`, `PipelineConfig`, `TileState` dataclasses.
- `build_tile_video_cmd(cfg, i)` builds one tile's video ffmpeg argv, returns `(cmd, w, h)`.
- `build_tile_audio_cmd(cfg, i)` builds one tile's audio ffmpeg argv (None if muted).
- `_tile_filter_ops(cfg, tile)` generates the per-tile scale/pad/fps filter chain.

## Implementation Notes & Tips
- Subprocess: never use `shell=True`; always pass argv lists.
- Metadata: `ffprobe` JSON is cached per-path in `_metadata_cache`.
- Seeking: use `normalize_seek` and `random_seek` helpers; respect `--loop`.
- Selection: `random.sample` is preferred to avoid mutation bias.
- Tile replace: aim to pick unused, different videos first; fall back gracefully.
- HW accel: `--hwaccel auto` probes CUDA with runtime lib check, then VAAPI.
- Display check: verifies `DISPLAY` or `WAYLAND_DISPLAY` is set before init.
- Audio: one ffmpeg→ffplay per tile; the sound server mixes them. ffplay is gated on that
  tile's first frame, and the video is then held until ffplay's clock actually starts
  (`_watch_audio_clock` → `_audio_started`). Without that hold, 4K/NFS audio seek latency
  makes video lead audio by ~0.5 s. `AUDIO_START_TIMEOUT` prevents a permanently black tile
  if audio fails.
- Frame pacing: `run()` accumulates `elapsed * VIDEO_FPS` into `_frames_due` and pumps
  that many frames per tile; content is locked to the wall clock (30 fps) and stalls are
  absorbed by dropping older frames. Cap via `MAX_CATCHUP_FRAMES`.
- Pipe sizing: a raw cell frame is far larger than the 64 KB pipe default, so
  `_start_tile` raises `F_SETPIPE_SZ`. Read a whole frame per pump, not one chunk.
- Teardown: `kill_proc` uses SIGKILL — children are stateless and graceful waits made
  tile replacement feel slow.
- Display scaling: `_blit_frame` scales to *fit* the window and centres with black
  bars, so tiling compositors (niri/sway) can't stretch the grid. Never
  `pygame.transform.scale` the frame straight to the window size.

## Testing Guidelines
- Primary: manual runs against a local folder with supported extensions.
- Requires a desktop session (X11/Wayland) — headless servers will error out cleanly.
- Determinism: use `--seed <int>` for reproducible selection/seek behavior.
- Smoke checks: `--hwaccel off --no-audio` on 4–9 files.
- If adding logic, isolate pure helpers for future unit tests; keep side effects at the edges.

## Commit & Pull Request Guidelines
- Commits: short, imperative mood ("Add …", "Fix …"), scoped changes.
- PRs: include concise description, rationale, usage examples, and before/after behavior.
- Screenshots or short screen recordings helpful when UI changes.

## Performance Tuning
- All modes use rawvideo pipe (no encode overhead).
- Enable hardware decode with `--hwaccel auto`.
- Reduce audio cost with `--audio-mode one` or `--no-audio`.
- Lower audio rate (e.g., `--audio-rate 44100` or `32000`).
- Reduce grid size or cell dimensions on constrained systems.
- The pygame viewer uses `select.select` with 50ms timeout for non-blocking I/O.
