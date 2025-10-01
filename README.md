PyTorch 2.7.1 on Fedora 41 with CUDA 12.9 and Python 3.13
============================

This is my fork of Pytorch 2.7.1 that has been verified to build correctly and work with CUDA 12.9 and Python 3.13, on Fedora 41. It will likely work just as well on more recent Fedora versions. I cannot speak for earlier Fedora releases since I haven't tried.

The `main` branch in this repo contains the unmodified upstream PyTorch version 2.7.1. The fork described in this README is in the `v2.7.1-cuda-12.9-fc41` branch.

The `v2.7.1-macos-15.5-sequoia-arm64` branch in this repo is my fork that has been verified to work on MacOS Sequoia 15.5 with Apple Metal 3.1 and Python 3.13.

The original PyTorch [README.md](https://github.com/steleman/pytorch-cuda-2.7.1/blob/v2.7.1-cuda-12.9-fc41/PYTORCH.README.md) file has been renamed to  [PYTORCH.README.md]((https://github.com/steleman/pytorch-cuda-2.7.1/blob/v2.7.1-cuda-12.9-fc41/PYTORCH.README.md).

Build scripts and build instructions are in the toplevel `build_scripts` directory. Please read the file named `HOWTO.BUILD.README.txt` containing instructions on how to use the build scripts.

You can find and download pre-built  Fedora 41 `rpms` and/or MacOS `pkgs` and their corresponding PyTorch Wheels at my Google Drive:

- Fedora 41:
	- https://drive.google.com/drive/folders/1ZjR9F-lzs5ZKhvv79tDjh511oYUZ3_2q?usp=drive_link

- MacOS 15.5:
	- https://drive.google.com/drive/folders/1rvqBV1-7pTU53CgQqI_1nFjoI7zewkMX?usp=drive_link

The rpm `python3-pytorch-cuda-sm89-2.7.1-700.fc41.x86_64.rpm` contains the Fedora 41 PyTorch 2.7.1 Python Wheel packaged as an rpm.

The Linux `rpms` and Wheels have been built for CUDA Compute Capability 89. That means the GeForce RTX 4080/4090 family.

The Python Wheels install normally with `pip3`.

On MacOS, please use the script `install-pytorch-macos-wheels.sh` to install. There are some dependencies to be installed. 

The MacOS `pkgs` assume that you installed Python 3.13 from `Homebrew`.

Installing CUDA 12.9 on Fedora 41+ is left as an exercise for the reader. :-)

