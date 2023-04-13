{ pkgs ? import (fetchTarball "https://github.com/NixOS/nixpkgs/archive/d1c3fea7ecbed758168787fe4e4a3157e52bc808.tar.gz") {}
}:

import ./default.nix {pkgs = pkgs;}

