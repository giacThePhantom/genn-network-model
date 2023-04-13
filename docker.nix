{ pkgs ? import (fetchTarball "https://github.com/NixOS/nixpkgs/archive/d1c3fea7ecbed758168787fe4e4a3157e52bc808.tar.gz") {}
, pkgsLinux ? import (fetchTarball "https://github.com/NixOS/nixpkgs/archive/d1c3fea7ecbed758168787fe4e4a3157e52bc808.tar.gz") {}
}:

let
  beegenn = import ./default.nix {inherit pkgs; };
in
  pkgs.dockerTools.buildImage {
    name = "erolmatei/beegenn";
    tag = "latest";

    contents = pkgs.buildEnv {
      name = "image-root";
      paths = with pkgs; [
        python310
        busybox
        beegenn
        bashInteractive
      ]; 
      #pathsToLink = ["/bin"]; # We need EVERYTHING
    };

    runAsRoot = ''
    #!${pkgs.runtimeShell}
      mkdir -p /out
    '';

    config = {
       Cmd = [ "/bin/python" "-m" "beegenn" "simulation" "/data" "sim3" ];
      # Cmd = ["/bin/python" "--version" ];
      WorkingDir = "/out";
      Volumes = { "/out" = { }; };
    };
}

