# Repository Guidelines

## Project Structure & Module Organization
- Root-level Python tool: `video-wall.py` (main entrypoint).
- Docs: `README.md`.
- Dev environment: `shell.nix`, `.envrc` (direnv+nix; optional venv).
- No packages or tests folder yet; keep single-file CLI unless a refactor is discussed.

## Build, Test, and Development Commands
- `direnv allow` — load nix shell and env (requires direnv).
- `nix-shell` — enter dev shell with Python, ruff, and ffmpeg tools.
- `python3 video-wall.py <folder>` — run video wall.
  - Example: `python3 video-wall.py ~/videos --count 9 --fast --hwaccel auto`.
- `ruff check .` — lint Python code.

Runtime dependencies: ensure `ffmpeg`, `ffprobe`, and `ffplay` are in `PATH`.

## Current Feature Set (CLI)
- Grid layout with `--count` 4–16; optional `--rows/--cols`.
- Cell sizing via `--cell-width/--cell-height`; auto total size from grid.
- Fast path `--fast` streaming rawvideo+pcm to `ffplay` (low latency/CPU).
- Audio control: `--no-audio`, `--audio-mode {mix,one}`, `--audio-tile`, `--audio-rate`.
- Hardware decode `--hwaccel {off,auto,cuda,vaapi}` with auto device discovery for VAAPI.
- Input selection: recursive by default (`--no-recursive` to limit to top-level).
- Supported extensions configurable via `--exts` (defaults to common video types).
- Randomized start within `--start_percentage/--end_percentage` of duration.
- Deterministic runs with `--seed`.
- Verbose logging with `--verbose`.

## Runtime Keyboard Controls
- SPACE: pause/resume pipeline.
- r: replace a random tile.
- 1..9: replace a specific tile (1-indexed).
- f/F + number: seek forward 10s/30s for tile.
- b/B + number: seek backward 10s/30s for tile.
- q: quit.

## Coding Style & Naming Conventions
- Language: Python 3.10+ (uses `|` unions, dataclasses).
- Indentation: 4 spaces; target line width ~100 chars.
- Naming: `snake_case` for functions/vars, `PascalCase` for classes, `UPPER_CASE` for constants.
- CLI: prefer explicit flags; keep current semantics backward compatible.
- Linting: fix `ruff` findings or `# noqa` with a brief rationale.

## Implementation Notes & Tips
- Subprocess: never use `shell=True`; always pass argv lists (already implemented).
- Metadata: `ffprobe` JSON is cached per-path in `_metadata_cache`.
- Seeking: use `normalize_seek` and `random_seek` helpers; respect `--loop`.
- Selection: `random.sample` is preferred to avoid mutation bias.
- Tile replace: aim to pick unused, different videos first; fall back gracefully.
- HW accel: `--hwaccel auto` prefers CUDA, then VAAPI if a writable render node is found.
- VAAPI: auto-sets `-vaapi_device` when a suitable `/dev/dri/renderD*` is accessible.
- Filters: performance can benefit from `--filter-threads` on multi-core systems.

## Testing Guidelines
- Primary: manual runs against a local folder with supported extensions: `.mp4,.mov,.mkv,.avi,.m4v,.webm`.
- Determinism: use `--seed <int>` for reproducible selection/seek behavior.
- Smoke checks: `--fast --hwaccel auto --no-audio` on 4–9 files; verify tile replace/seek keys per on-screen help.
- If adding logic, isolate pure helpers for future unit tests; keep side effects at the edges.

## Commit & Pull Request Guidelines
- Commits: short, imperative mood ("Add …", "Fix …"), scoped changes.
- PRs: include concise description, rationale, usage examples, and before/after behavior. Link issues as applicable and note HW/OS tested.
- Screenshots or short screen recordings are helpful when UI/behavior changes.

## Security & Configuration Tips
- Do not use `shell=True`; keep subprocess calls as argv lists.
- Avoid changing default flags or hardware-accel behavior without discussion.
- Validate file extensions and existence (already enforced); preserve these checks when modifying I/O.

## Performance Tuning Cheat Sheet
- Prefer `--fast` for low latency and CPU usage.
- Set `--hwaccel auto` to enable CUDA/VAAPI when available.
- Limit audio cost with `--audio-mode one` or `--no-audio`.
- Lower audio rate (e.g., `--audio-rate 44100` or `32000`).
- Adjust `--filter-threads 2` (or higher) on multi-core systems.
- Reduce grid size or cell dimensions for constrained systems.
