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
  - Example: `python3 video-wall.py ~/videos --count 9 --preset 3x3`.
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
Replaces the old ffplay-based pipeline with a **pygame viewer**:

```
ffmpeg (video) ──rawvideo pipe──▶ pygame viewer (SDL window, keyboard events)
ffmpeg (audio)  ──wav pipe──────▶ ffplay -nodisp -vn (headless audio)
```

### VideoWindow
- `VideoWindow` class owns the pygame window and ffmpeg subprocess lifecycle.
- `select.select`-based non-blocking frame reads from ffmpeg stdout pipe.
- `pygame.image.frombuffer()` converts raw RGB bytes to a display surface.
- Keyboard events processed via `pygame.event.get()` — works when SDL window is focused.
- On tile replace/seek: `_restart_ffmpeg()` kills the old ffmpeg, starts a new one.
  **Pygame window stays open** — no flicker.
- `AudioPipeline` helper manages a separate ffmpeg→ffplay chain for audio.

### Models
- `VideoMeta`, `HwAccelConfig`, `PipelineConfig`, `TileState` dataclasses.
- `build_ffmpeg_cmd(cfg)` builds video ffmpeg argv, returns `(cmd, width, height)`.
- `_build_audio_cmd(cfg)` builds simplified audio-only ffmpeg argv.
- `_build_filter_graph()` generates the filter_complex string.

## Implementation Notes & Tips
- Subprocess: never use `shell=True`; always pass argv lists.
- Metadata: `ffprobe` JSON is cached per-path in `_metadata_cache`.
- Seeking: use `normalize_seek` and `random_seek` helpers; respect `--loop`.
- Selection: `random.sample` is preferred to avoid mutation bias.
- Tile replace: aim to pick unused, different videos first; fall back gracefully.
- HW accel: `--hwaccel auto` probes CUDA with runtime lib check, then VAAPI.
- Display check: verifies `DISPLAY` or `WAYLAND_DISPLAY` is set before init.
- Audio pipeline: started only when `--no-audio` is not set and at least one tile has audio.

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
