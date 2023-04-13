{ pkgs ? import <nixpkgs> {} }:

let
   beegenn = import ./default.nix {pkgs = pkgs;};
in
  pkgs.mkShell {
    packages = [pkgs.python310 beegenn];
  }
