{ pkgs ? import <nixpkgs> {}
, stdenv ? pkgs.stdenv
, lib ? pkgs.lib} :

with pkgs; 

let
   # To get CUDA 11.1 we need to fix some nixpkgs.
   cudaNixpkgs = import (builtins.fetchGit {
       # Descriptive name to make the store path easier to identify                
       name = "cuda-11.1-nixpkgs";
       url = "https://github.com/NixOS/nixpkgs/";                       
       ref = "refs/heads/nixpkgs-unstable";                     
       rev = "d1c3fea7ecbed758168787fe4e4a3157e52bc808";
   }) {};

  cuda = cudaNixpkgs.cudaPackages.cudatoolkit_11_1;
  nvidia_x11 = cudaPkgs.linuxPackages.nvidia_x11;

  cudaPkgs = [cuda nvidia_x11];

  gennSrc = fetchFromGitHub {
    owner = "genn-team";
    repo = "genn";
    rev = "5aa20a0f9cdba07cd899f6ff38fcdb9d2d61957e";
    sha256 = "JX0pysp4GgpybeoCuCUS5uCKaiIhDl8elxMK5YBCkdc=";
  };

  #gennSrc = lib.cleanSource ../genn;

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
    nativeBuildInputs = [swig];
    makeFlags = [ "PREFIX=$(out)" ];
    patches = [ ./genn-setup.patch ];

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
    src = lib.cleanSource ./.; # TODO

    propagatedBuildInputs = beegennPythonDeps ++ cudaPkgs;

    doCheck = false;
  };


in
  beegenn
