#!/bin/bash

if [ `id -u` -ne 0 ] ; then
  echo "You must be root to do this."
  exit 1
fi

dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/fedora41/x86_64/cuda-fedora41.repo
dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo
dnf config-manager --add-repo https://developer.download.nvidia.com/hpc-sdk/rhel/nvhpc.repo

echo "dnf -y install libcutensor2 libcutensor-devel libcutensor-doc"
dnf -y install libcutensor2 libcutensor-devel libcutensor-doc

echo "dnf -y install cudnn-cuda-12"
dnf -y install cudnn-cuda-12

echo "dnf install -y nvhpc-25.5"
dnf install -y nvhpc-25.5

echo "dnf install -y tensorrt"
dnf install -y tensorrt tensorrt-devel tensorrt-libs

echo "dnf install -y cupti cuda-cupti-12-9"
dnf install -y cupti cuda-cupti-12-9

echo "dnf install -y libnccl libnccl-devel libnccl-static"
dnf install libnccl libnccl-devel libnccl-static

echo "dnf -y install libcusparselt0 libcusparselt-devel"
dnf -y install libcusparselt0 libcusparselt-devel

echo "dnf -y install libcutensor2 libcutensor-devel libcutensor-doc"
dnf -y install libcutensor2 libcutensor-devel libcutensor-doc

