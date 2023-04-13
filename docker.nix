{ pkgs ? import <nixpkgs> { }
, pkgsLinux ? import <nixpkgs> { system = "x86_64-linux"; }
}:

let
  beegenn = import ./default.nix;
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
        # TODO: this does not work. propagatedBuildInputs does not work as I expected. 
        linuxPackages.nvidia_x11 
        cudatoolkit
      ] ++ beegenn.env.propagatedBuildInputs
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

