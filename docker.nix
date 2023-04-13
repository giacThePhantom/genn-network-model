{
  pkgs ? import (fetchTarball "https://github.com/NixOS/nixpkgs/archive/3fb8eedc450286d5092e4953118212fa21091b3b.tar.gz") {}
, pkgsLinux ?  import (fetchTarball "https://github.com/NixOS/nixpkgs/archive/3fb8eedc450286d5092e4953118212fa21091b3b.tar.gz") { system = "x86_64-linux"; }
}:

let
  beegenn = import ./default.nix {inherit pkgs; };
in
  # run the following:
  # docker load $(nix-build docker.nix)
  # This will avoid making another intermediate file...
  pkgs.dockerTools.buildImage {
    name = "beegenn";
    tag = "latest";

    copyToRoot = pkgs.buildEnv {
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
       Cmd = [ "/bin/python" "-m" "beegenn" "simulation" "/out/data" "sim3" ];
      # Cmd = ["/bin/python" "--version" ];
      WorkingDir = "/out";
      Volumes = { "/out" = { }; };
    };
}

