with import <nixpkgs> {};

let
  cuda = cudaPackages.cudatoolkit;
  cudaPkgs = [cuda linuxPackages.nvidia_x11 ];

  gennSrc2 = fetchFromGitHub {
    owner = "genn-team";
    repo = "genn";
    rev = "5aa20a0f9cdba07cd899f6ff38fcdb9d2d61957e";
    sha256 = "JX0pysp4GgpybeoCuCUS5uCKaiIhDl8elxMK5YBCkdc=";
  };

  gennSrc = ../genn; 

  genn = stdenv.mkDerivation {
    name = "genn";
    version = "4.8.0";
    src = gennSrc;
    propagatedBuildInputs = [gnumake cuda swig];

    makeFlags = [ "PREFIX=$(out)"
    "DYNAMIC=1"
    "LIBRARY_DIRECTORY=$(out)/lib"];

    # Not needed, as we only need the static packages.
    dontInstall = true; 

    CUDA_PATH="${cuda}";
  };

  pythonDeps = with python310Packages; [deprecated numpy genn six psutil setuptools];
  doCheck = false;

  pygenn = python310Packages.buildPythonPackage rec {
    pname = "pygenn";
    version = "4.8.0";
    src = gennSrc;
    propagatedBuildInputs = pythonDeps ++ cudaPkgs;
    nativeBuildInputs = [swig genn];

    CUDA_PATH="${cuda}";
    doCheck = false;
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
    ipykernel
    gnumake
  ];

  beegenn = python310Packages.buildPythonPackage {
    pname = "beegenn";
    version = "0.0.1";
    src = ./.; # TODO

    propagatedBuildInputs = beegennPythonDeps ++ cudaPkgs;

    doCheck = false;
  };


in
  #beegenn
  pygenn
