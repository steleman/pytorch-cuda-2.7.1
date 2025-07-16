#!/bin/bash

here="`pwd`"
arch="`uname -m`"
output_file="${here}/pytorch-wheel-python-whl-build.log"

cuda_home="/usr/local/cuda-12.9"
cuda_libdir="${cuda_home}/lib64"
cuda_target_libdir="${cuda_home}/targets/${arch}-linux/lib"
system_incdir="/usr/include"
system_libdir="/usr/lib64"
pytorch_version="2.7.1"
pytorch_string_version="2_7_1"
pytorch_prefix="/opt/pytorch/${pytorch_version}"
pytorch_incdir="${pytorch_prefix}/include"
pytorch_libdir="${pytorch_prefix}/lib64"
fcversion="fc41"
buildversion="700"

if [ -e /etc/profile.d/modules.sh ] ; then
  source /etc/profile.d/modules.sh
fi

module load mpi/openmpi-x86_64

export PATH="${CUDA_HOME}/bin:/usr/lib64/openmpi/bin:${PATH}"
openmpi_libdir="/usr/lib64/openmpi/lib"

export CC="/usr/bin/gcc"
export CXX="/usr/bin/g++"

export MAX_JOBS=8
export USE_CUDA=1
export CFLAGS="-g -O2 -std=gnu11 -Wall -Wextra"
export CXXFLAGS="-g -O2 -std=gnu++17 -Wall -Wextra"
export LDFLAGS="-Wl,--no-relax -Wl,-z -Wl,noreloc-overflow"
export LDFLAGS="${LDFLAGS} -Wl,-rpath,${openmpi_libdir}"
export LDFLAGS="${LDFLAGS} -Wl,-rpath,${cuda_libdir}"
export LDFLAGS="${LDFLAGS} -Wl,-rpath,${cuda_target_libdir}"
export CUDA_HOME="${cuda_home}"
export USE_CUDNN=1
export USE_CUSPARSELT=1
export USE_CUDSS=1
export USE_CUFILE=1
export USE_FBGEMM=1
export USE_KINETO=1
export USE_NUMPY=1
export BUILD_TEST=1
export USE_MKLDNN=1
export MKLDNN_CPU_RUNTIME="OMP"
export USE_STATIC_MKL=1
export USE_ITT=1
export USE_NNPACK=1
export USE_DISTRIBUTED=1
export USE_TENSORPIPE=1
export USE_TENSORRT=1
export USE_GLOO=1
export USE_GLOO_WITH_OPENSSL=1
export USE_MPI=1
export USE_SYSTEM_NCCL=1
export USE_OPENMP=1
export USE_FLASH_ATTENTION=1
export USE_MEM_EFF_ATTENTION=1
export USE_GOLD_LINKER=0
export BUILD_BINARY=1
export ATEN_AVX512_256="TRUE"
export PYTORCH_BUILD_VERSION="${pytorch_version}"
export PYTORCH_BUILD_NUMBER="700"
export TORCH_CUDA_ARCH_LIST="12.0\;12.0a"
export CMAKE_CUDA_ARCHITECTURES="120;120a"
export BLAS="OpenBLAS"
export MKL_THREADING="OMP"
export CUDAHOSTCXX="${CXX}"
export CUDA_NVCC_EXECUTABLE="${cuda_home}/bin/nvcc"
export CMAKE_CUDA_COMPILER="${cuda_home}/bin/nvcc"
export CUDACXX="${cuda_home}/bin/nvcc"

export CUDNN_LIB_DIR="${system_libdir}"
export CUDNN_INCLUDE_DIR="${system_incdir}"
export CUDNN_LIBRARY="${system_libdir}"

export NCCL_ROOT="/usr"
export NCCL_LIB_DIR="${system_libdir}"
export NCCL_INCLUDE_DIR="${system_incdir}"

export ATEN_THREADING="OMP"
export USE_SYSTEM_LIBS=1

export USE_MIMALLOC=1
export USE_PRIORITIZED_TEXT_FOR_LD=0
export BUILD_LIBTORCH_WHL=1
export BUILD_PYTHON_ONLY=0

export TORCH_PACKAGE_NAME="torch-python-${pytorch_version}-${buildversion}.${fcversion}.${arch}"
export LIBTORCH_PACKAGE_NAME="libtorch-python-${pytorch_version}-${buildversion}.${fcversion}.${arch}"
export TORCH_PACKAGE_VERSION="${pytorch_version}.${buildversion}"

echo "rm -rf ./build ./dist"
rm -rf ./build ./dist

echo "cat /dev/null > ${output_file}"
cat /dev/null > ${output_file}

echo "python3 ./setup.py bdist_wheel --verbose"
python3 ./setup.py bdist_wheel --verbose >> ${output_file} 2>&1

