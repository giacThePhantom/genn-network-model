{ pkgs ? import <nixpkgs> {} }:

let
   beegenn = import ./default.nix;
in
  pkgs.mkShell {
    packages = [beegenn pkgs.python310];
  }
