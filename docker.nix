{ pkgs ? import <nixpkgs> { }
, pkgsLinux ? import <nixpkgs> { system = "x86_64-linux"; }
}:

let
  beegenn = import ./default.nix;
in
  pkgs.dockerTools.buildImage {
  name = "beegenn";
  config = {
    # TODO
    Cmd = [ "${python310} -m beegenn simulation data" ];
  };
}

