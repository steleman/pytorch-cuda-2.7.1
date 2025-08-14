#!/bin/bash

export CC="/usr/bin/clang"
export CXX="/usr/bin/clang++"

distro="`uname -s`"

if [ -e /opt/homebrew/bin/brew ] ; then
  /opt/homebrew/bin/brew shellenv >& /tmp/brewshellenv.$$
  source /tmp/brewshellenv.$$
  rm -f /tmp/brewshellenv.$$
fi

export PATH="/opt/homebrew/bin:/usr/bin:/bin:/usr/local/bin:/usr/sbin:${here}/bin"
export PATH="/opt/homebrew/opt/coreutils/libexec/gnubin:${PATH}"
export PATH="/opt/homebrew/opt/grep/libexec/gnubin:${PATH}"
export PATH="/opt/homebrew/opt/findutils/libexec/gnubin:${PATH}"
export PATH="/opt/homebrew/opt/diffutils/libexec/gnubin:${PATH}"
export GMAKE="/opt/homebrew/bin/gmake"
export MAKE="${GMAKE}"
export CMAKE="/opt/homebrew/opt/cmake/bin/cmake"
build_type="Release"

openblas="/usr/local"
openblas_libdir="${openblas}/lib"
openblas_incdir="${openblas}/include"
openmpi="/opt/homebrew/opt/openmpi"
openmpi_incdir="${openmpi}/include"
openmpi_libdir="${openmpi}/lib"
openmp="/usr/local"
openmp_incdir="${openmp}/include"
openmp_libdir="${openmp}/lib"
sleef="/opt/homebrew/opt/sleef"
sleef_incdir="${sleef}/include"
sleef_libdir="${sleef}/lib"
hb_python="/opt/homebrew/opt/python@3.13"
hb_python_libdir="${hb_python}/lib"
eigen3="/opt/homebrew/opt/eigen"
eigen3_pkgconfig="${eigen3}/share/pkgconfig"
abseil="/opt/homebrew/opt/abseil"
abseil_libdir="${abseil}/lib"
protobuf="/opt/homebrew/opt/protobuf"
protobuf_libdir="${protobuf}/lib"

export PKG_CONFIG_PATH="${openblas_libdir}/pkgconfig"
export PKG_CONFIG_PATH="${PKG_CONFIG_PATH}:${openmpi_libdir}/pkgconfig"
export PKG_CONFIG_PATH="${PKG_CONFIG_PATH}:${openblas_libdir}/pkgconfig"
export PKG_CONFIG_PATH="${PKG_CONFIG_PATH}:${hb_python_libdir}/pkgconfig"
export PKG_CONFIG_PATH="${PKG_CONFIG_PATH}:${eigen3_pkgconfig}/pkgconfig"
export PKG_CONFIG_PATH="${PKG_CONFIG_PATH}:${protobuf_libdir}/pkgconfig"
export PKG_CONFIG_PATH="${PKG_CONFIG_PATH}:${abseil_libdir}/pkgconfig"

${CC} --version
${CXX} --version

pytorch_version="2.7.1"
here="`pwd`"
topdir="`dirname ${here}`"
srcdir="${topdir}/pytorch-${pytorch_version}"
njobs="4"
build_output="${here}/pytorch-build-j${njobs}.out"

cat /dev/null > ${build_output}

echo "Running: gmake -j${njobs} >> ${build_output} 2>&1"
gmake -j${njobs}  >> ${build_output} 2>&1

