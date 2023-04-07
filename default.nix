with import <nixpkgs> {};

let
    cuda = cudaPackages.cudatoolkit;
    cudaPkgs = [cuda linuxPackages.nvidia_x11 ];

    src = fetchFromGithub {
          url = "https://github.com/genn-team/genn";
          sha256 = "5aa20a0f9cdba07cd899f6ff38fcdb9d2d61957e";
        };


    genn = stdenv.mkDerivation {
        name = "genn";
        version = "4.8.0";
        src = src;
        propagatedBuildInputs = [gnumake cuda swig];

        makeFlags = [ "PREFIX=$(out)"
                      "DYNAMIC=1"
                      "LIBRARY_DIRECTORY=$(out)/lib"];

        CUDA_PATH="${cuda}";

        # Not needed, as we only need the static packages.
        dontInstall = true; 
    };

    pythonDeps = with python310Packages; [deprecated numpy genn six psutil];
    doCheck = false;

    pygenn = python310Packages.buildPythonPackage rec {
        name = "pygenn";
        version = "4.8.0";
        src = src;
        propagatedBuildInputs = pythonDeps ++ cudaPkgs;
        nativeBuildInputs = [swig];

        CUDA_PATH="${cuda}";
    };

    beegennPythonDeps = with python310Packages; [
      matplotlib
      numpy
      tables
      networkx
      jsonschema
      pandas
      tqdm
      pygenn
    ];

    beegenn = python310Packages.buildPythonPackage {
      name = "beegenn";
      version = "0.0.1"
      src = ./.; # TODO

      propagatedBuildInputs = beegennPythonDeps ++ cudaPkgs;
    };
in
    beegenn
