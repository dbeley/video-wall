# video-wall

Simple script to display several looped videos in a grid. 

## Dependencies

- ffmpeg

## Tips

- Use `--fast` for lowest latency and CPU.
- For audio-heavy walls, try `--audio-mode one` to use a single tile's audio instead of mixing all tiles.
- Reduce audio cost with `--audio-rate 44100` (or `32000`) and optionally `--filter-threads 2` on multi-core systems.
