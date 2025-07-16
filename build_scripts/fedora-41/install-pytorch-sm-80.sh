#!/bin/bash

if [ -e /etc/profile.d/modules.sh ] ; then
  echo ""source /etc/profile.d/modules.sh""
  source /etc/profile.d/modules.sh
fi

/usr/bin/gcc --version
/usr/bin/g++ --version

pytorch_version="2.7.1"
here="`pwd`"
topdir="`dirname ${here}`"
destdir="${topdir}/install-pytorch-sm-80"
srcdir="${topdir}/pytorch-${pytorch_version}"
install_output="${here}/pytorch-sm-80-install.out"

export CUDATOP="/usr/local/cuda-12.9"
export CUDA_MAJOR=12
export CUDA_MINOR=9
export PATH="${CUDATOP}/bin:/usr/lib64/openmpi/bin:/usr/bin:/usr/sbin:/usr/local/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/local/cuda-12.9/lib64:/usr/local/cuda-12.9/targets/x86_64-linux/lib"
export DESTDIR="${destdir}"
unset CUDA_FORCE_PTX_JIT

if [ -f /usr/bin/ccache ] ; then
  ccache --clear
fi

cat /dev/null > ${install_output}
echo "Running: gmake DESTDIR=${DESTDIR} install >> ${install_output} 2>&1"
gmake DESTDIR=${DESTDIR} install >> ${install_output} 2>&1

