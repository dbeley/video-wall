# Repository Guidelines

## Project Structure & Module Organization
- Root-level Python tool: `video-wall.py` (main entrypoint).
- Docs: `README.md`.
- Dev environment: `shell.nix`, `.envrc` (direnv+nix; optional venv).
- No packages or tests folder yet; keep single-file CLI unless a refactor is discussed.

## Build, Test, and Development Commands
- `direnv allow` — load nix shell and env (requires direnv).
- `nix-shell` — enter dev shell with Python, ruff, ffmpeg libs available.
- `python3 video-wall.py <folder>` — run video wall. Example: `python3 video-wall.py ~/videos --count 9 --fast --hwaccel auto`.
- `ruff check .` — lint Python code.

Runtime dependencies: `ffmpeg`, `ffprobe`, `ffplay` must be in `PATH`.

## Coding Style & Naming Conventions
- Language: Python 3.10+ (uses `|` type unions, dataclasses).
- Indentation: 4 spaces; max line width ~100 chars.
- Naming: `snake_case` for functions/vars, `PascalCase` for classes, `UPPER_CASE` for constants.
- CLI: prefer explicit flags; keep current semantics backward compatible.
- Linting: fix `ruff` findings or `# noqa` with rationale.

## Testing Guidelines
- Primary: manual run against a local folder with supported extensions: `.mp4,.mov,.mkv,.avi,.m4v,.webm`.
- Determinism: use `--seed <int>` for reproducible selection/seek.
- Smoke checks: `--fast --hwaccel auto --no-audio` on 4–9 files; verify tile replace/seek keys per on-screen help.
- If adding logic, isolate pure helpers for easy unit tests later; keep side effects at the edge.

## Commit & Pull Request Guidelines
- Commits: short, imperative mood (“Add …”, “Fix …”), scoped changes.
- PRs: include concise description, rationale, usage example, and before/after behavior. Link issues when applicable and note HW/OS tested.
- Screenshots or short screen recordings are helpful when UI/behavior changes.

## Security & Configuration Tips
- Do not shell=True; keep subprocess calls as arg lists (as implemented).
- Avoid changing default flags or hardware-accel behavior without discussion.
- Validate file extensions and existence (already enforced); preserve these checks when modifying I/O.

