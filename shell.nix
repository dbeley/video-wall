{ pkgs ? import <nixpkgs> { } }:

pkgs.mkShell {
  nativeBuildInputs = with pkgs; [
    ffmpeg
    python3
    python3Packages.pygame
    python3Packages.pillow
    ruff
  ];
}
