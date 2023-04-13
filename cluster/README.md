# Internal Unitn HPC guide

This small guide assumes you already have access to the UniTn cluster and have some familiarity with the cluster infrastructure. If not, refer to [this](https://cbp-unitn.gitlab.io/QCB/tutorial4_HPC_git) first.

## Set up the working directory

1. Download and install squashfs-tools from a tarball. Rationale: it is required by docker and singularity, but is not available as a module.

```bash
git clone https://github.com/plougher/squashfs-tools
cd squasfs-tools/squashfs-tools
INSTALL_PREFIX=~/.local make install
```

2. Clone this repo. Make sure that `genn-network-model` is in your home folder.

3. Setup an output folder, then run the following from your home directory
```bash
mkdir output
qsub genn-network-model/cluster/simulate.qbs
```

