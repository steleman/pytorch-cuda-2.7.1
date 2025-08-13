#!/bin/bash

if [ -e /etc/profile.d/modules.sh ] ; then
  source /etc/profile.d/modules.sh
fi

module load mpi/openmpi-x86_64

cuda_archs_enabled="89"
cuda_arch_list="8.9"
here="`pwd`"
arch="`uname -m`"
output_file="${here}/pytorch-wheel-libtorch-whl-build-sm-${cuda_archs_enabled}.log"
export CUDA_VERSION="12.9"
export CUDA_HOME="/usr/local/cuda-${CUDA_VERSION}"
cuda_target_lib="${CUDA_HOME}/targets/${arch}-linux/lib"
libnvrtc_filepath="${cuda_target_lib}/libnvrtc.so"
system_incdir="/usr/include"
system_libdir="/usr/lib64"
prefix="/usr/lib64/python3.13/site-packages/torch"
pytorch_version="2.7.1"
pytorch_string_version="2_7_1"
pytorch_prefix="/opt/pytorch/${pytorch_version}"
pytorch_incdir="${pytorch_prefix}/include"
pytorch_libdir="${pytorch_prefix}/lib64"
fcversion="fc41"
buildversion="700"

linker_type="BFD"
bfd_linker_flags="-Wl,--no-relax -Wl,-z -Wl,noreloc-overflow"
bfd_linker_flags="${bfd_linker_flags} -Wl,--export-dynamic"
bfd_linker_flags="${bfd_linker_flags} -Wl,-rpath,${prefix}/lib"
bfd_linker_flags="${bfd_linker_flags} -Wl,-rpath,${CUDA_HOME}/lib64"
bfd_linker_flags="${bfd_linker_flags} -Wl,-rpath,${cuda_target_lib}"
exe_linker_flags="-Wl,--no-relax"
exe_linker_flags="${exe_linker_flags} -Wl,-rpath,${prefix}/lib"
exe_linker_flags="${exe_linker_flags} -Wl,-rpath,${CUDA_HOME}/lib64"
exe_linker_flags="${exe_linker_flags} -Wl,-rpath,${cuda_target_lib}"

export PATH="${CUDA_HOME}/bin:/usr/lib64/openmpi/bin:${PATH}"
openmpi_libdir="/usr/lib64/openmpi/lib"
export PATH="${CUDA_HOME}/bin:/usr/lib64/openmpi/bin:/usr/bin:/usr/sbin:/usr/local/bin:${PATH}"
export MAKE="/usr/bin/gmake"
export GMAKE="/usr/bin/gmake"

export CC="/usr/bin/gcc"
export CXX="/usr/bin/g++"
export CFLAGS="-Wall -Wextra"
export CXXFLAGS="-Wall -Wextra"
export CPPFLAGS="-D_GNU_SOURCE -D_XOPEN_SOURCE=700"
export CMAKE="/usr/bin/cmake"
export CMAKE_BUILD_TYPE="Release"
export CMAKE_MAKE_PROGRAM=${GMAKE}
export CMAKE_C_COMPILER="${CC}"
export CMAKE_CXX_COMPILER="${CXX}"
export CMAKE_CUDA_COMPILER="${CUDA_HOME}/bin/nvcc"
export CUDACXX="${CUDA_HOME}/bin/nvcc"
export CMAKE_C_FLAGS="${CPPFLAGS} ${CFLAGS}"
export CMAKE_CXX_FLAGS="${CPPFLAGS} ${CXXFLAGS}"
export CMAKE_C_FLAGS_RELEASE="${CPPFLAGS} ${CFLAGS}"
export CMAKE_CXX_FLAGS_RELEASE="${CPPFLAGS} ${CXXFLAGS}"
export CMAKE_C_STANDARD="11"
export CMAKE_CXX_STANDARD="17"
export CMAKE_C_EXTENSIONS=1
export CMAKE_CXX_EXTENSIONS=1
export CMAKE_LINKER_TYPE="${linker_type}"
export CMAKE_EXE_LINKER_FLAGS="${bfd_linker_flags}"
export CMAKE_SHARED_LINKER_FLAGS="${bfd_linker_flags}"
export CMAKE_MODULE_LINKER_FLAGS="${bfd_linker_flags}"
export CMAKE_POSITION_INDEPENDENT_CODE=1
export CMAKE_VERBOSE_MAKEFILE=1
export CMAKE_SUPPRESS_REGENERATION=1
export USE_COLORIZE_OUTPUT=0
export BUILD_BINARY=1
export BUILD_CUSTOM_PROTOBUF=0
export BUILD_PYTHON=1
export BUILD_SHARED_LIBS=1
export BUILD_TEST=1
export BUILD_AOT_INDUCTOR_TEST=0
export BUILD_STATIC_RUNTIME_BENCHMARK=0
export USE_CUDA=1
export CUDA_MAJOR="12"
export CUDA_MINOR="9"
export CUDA_TARGET_INCDIR="${CUDA_HOME}/targets/${arch}-linux/include"
export CUDA_TARGET_LIBDIR="${CUDA_HOME}/targets/${arch}-linux/lib"
export CUDA_TOOLKIT_TARGET_INCLUDE="${CUDA_TARGET_INCDIR}"
export CUDA_ARCH_LIST="${cuda_arch_list}"
export TORCH_CUDA_ARCH_LIST="${cuda_arch_list}"
export CUDA_nvrtc_LIBRARY="${libnvrtc_filepath}"
export CUDA_NVRTC_LIB="${libnvrtc_filepath}"
export CMAKE_CUDA_ARCHITECTURES="${cuda_archs_enabled}"
export BLAS="OpenBLAS"
export BUILD_LAZY_CUDA_LINALG=1
export USE_ROCM=0
export USE_CUDNN=1
export CUSPARSELT_LIBRARY_PATH="/usr/lib64/libcusparseLt.so"
export CUSPARSELT_INCLUDE_PATH="/usr/include"
export USE_CUSPARSELT=1
export USE_CUDSS=1
export USE_CUFILE=1
export USE_CUPTI_SO=1
export USE_TENSORRT=1
export TENSORRT_LIBRARY="/usr/lib64/libnvinfer.so"
export TENSORRT_INCLUDE_DIR="/usr/include"
export NCCL_ROOT_DIR="/usr"
export USE_TCP_OPENSSL_LINK=1
export USE_TCP_OPENSSL_LINK_DEFAULT=1
export USE_TCP_OPENSSL_LOAD=0
export USE_TCP_OPENSSL_LOAD_DEFAULT=0
export GLOO_USE_CUDA_TOOLKIT=1
export GLOO_USE_TORCH_DTYPES=1
export BUILD_BENCHMARK=0
export GLOO_INSTALL=1
export CUTLASS_NVCC_EMBED_CUBIN=1
export CUTLASS_NVCC_EMBED_PTX=1
export CUTLASS_NVCC_VERBOSE=1
export CUTLASS_ENABLE_F16C=1
export CUTLASS_NVCC_ARCHS_ENABLED="${cuda_archs_enabled}"
export CUTLASS_LIBRARY_KERNELS="all"
export CUTLASS_LIBRARY_INSTANTIATION_LEVEL="max"
export CUTLASS_ENABLE_TENSOR_CORE_MMA_DEFAULT=1
export CUTLASS_ENABLE_TENSOR_CORE_MMA=1
export CUTLASS_ENABLE_CUBLAS=1
export CUTLASS_ENABLE_CUDNN=1
export CUTLASS_ENABLE_GDC_FOR_SM90=1
export CUTLASS_ENABLE_GDC_FOR_SM100=1
export CUTLASS_ENABLE_GDC_FOR_SM100_DEFAULT=1
export CUTLASS_ENABLE_SM90_EXTENDED_MMA_SHAPES=1
export SLEEF_SHOW_CONFIG=1
export SLEEF_ENABLE_LTO=1
export SLEEF_BUILD_SCALAR_LIB=1
export SLEEF_ENABLE_CUDA=1
export SLEEF_ENABLE_CXX=1
export BUILD_ONNX_PYTHON=1
export USE_LITE_PROTO=0
export USE_NCCL=1
export USE_SYSTEM_NCCL=1
export USE_XPU=0
export USE_NUMA=1
export USE_NVRTC=1
export USE_OPENCL=1
export USE_OPENMP=1
export USE_SYSTEM_EIGEN_INSTALL=1
export USE_ITT=1
export USE_DISTRIBUTED=1
export USE_UCC=0
export USE_GLOO=1
export USE_GLOO_WITH_OPENSSL=1
export USE_C10D_GLOO=1
export USE_C10D_NCCL=1
export USE_MPI=1
export USE_C10D_MPI=1
export USE_C10D_GLOO=1
export USE_C10D_NCCL=1
export USE_NUMPY=1
export USE_TENSORPIPE=1
export HAVE_SOVERSION=1
export USE_CCACHE=0
export BUILD_FUNCTORCH=1
export USE_SYSTEM_LIBS=1
export USE_SYSTEM_SLEEF=1
export USE_SYSTEM_EIGEN_INSTALL=1
export USE_SYSTEM_FP16=1
export USE_SYSTEM_PSIMD=1
export USE_SYSTEM_FXDIV=1
export USE_SYSTEM_PTHREADPOOL=1
export USE_SYSTEM_CPUINFO=1
export USE_SYSTEM_PYBIND11=1
export USE_SYSTEM_NVTX=1
export USE_GOLD_LINKER=0
export USE_MKLDNN=1
export MKLDNN_CPU_RUNTIME="OMP"
export MKL_THREADING="OMP"
export ATEN_AVX512_256="TRUE"
export PYTORCH_BUILD_VERSION="${pytorch_version}"
export PYTORCH_BUILD_NUMBER="700"
export MKL_THREADING="OMP"
export CUDAHOSTCXX="${CXX}"
export CUDA_NVCC_EXECUTABLE="${CUDA_HOME}/bin/nvcc"
export CUDACXX="${CUDA_HOME}/bin/nvcc"
export Torch_DIR="${here}/torch/share/cmake/Torch"

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

for file in \
  ./build/lib/libc10_cuda.so \
  ./build/lib/libonnx.so \
  ./build/lib/libtorch.so.2.7 \
  ./build/lib/libjitbackend_test.so \
  ./build/lib/libbackend_with_compiler.so \
  ./build/lib/libtorch.so \
  ./build/lib/libtorch.so.2.7.1 \
  ./build/lib/libtorch_cuda.so \
  ./build/lib/libc10.so \
  ./build/lib/libtorch_global_deps.so.2.7.1 \
  ./build/lib/libc10.so.2.7 \
  ./build/lib/libcaffe2_nvrtc.so \
  ./build/lib/libaoti_custom_ops.so \
  ./build/lib/libtorch_global_deps.so \
  ./build/lib/fbgemm_gpu_py.so \
  ./build/lib/libc10d_cuda_test.so \
  ./build/lib/libtorchbind_test.so \
  ./build/lib/libtorch_cpu.so.2.7 \
  ./build/lib/libonnx_proto.so \
  ./build/lib/libtorch_cpu.so.2.7.1 \
  ./build/lib/libc10.so.2.7.1 \
  ./build/lib/libtorch_global_deps.so.2.7 \
  ./build/lib/libtorch_cpu.so \
  ./torch/fbgemm_gpu/fbgemm_gpu_py.so \
  ./torch/lib/libc10_cuda.so \
  ./torch/lib/libtorch.so.2.7 \
  ./torch/lib/libjitbackend_test.so \
  ./torch/lib/libbackend_with_compiler.so \
  ./torch/lib/libtorch.so \
  ./torch/lib/libtorch.so.2.7.1 \
  ./torch/lib/libtorch_cuda.so \
  ./torch/lib/libc10.so \
  ./torch/lib/libtorch_global_deps.so.2.7.1 \
  ./torch/lib/libc10.so.2.7 \
  ./torch/lib/libcaffe2_nvrtc.so \
  ./torch/lib/libaoti_custom_ops.so \
  ./torch/lib/libtorch_global_deps.so \
  ./torch/lib/libc10d_cuda_test.so \
  ./torch/lib/libtorchbind_test.so \
  ./torch/lib/libtorch_cpu.so.2.7 \
  ./torch/lib/libtorch_cpu.so.2.7.1 \
  ./torch/lib/libc10.so.2.7.1 \
  ./torch/lib/libtorch_global_deps.so.2.7 \
  ./torch/lib/libtorch_cpu.so \
  ./torch/lib/libtorch_python.so.2.7.1 \
  ./torch/lib/libshm.so.2.7.1 \
  ./torch/lib/libtorch_cuda_linalg.so \
  ./functorch/functorch.so \
  ./torch/lib/libtorch_python.dylib \
  ./torch/lib/libbackend_with_compiler.dylib \
  ./torch/lib/libtorch.dylib \
  ./torch/lib/libtorch_global_deps.dylib \
  ./torch/lib/libtorch_cpu.dylib \
  ./torch/lib/libjitbackend_test.dylib \
  ./torch/lib/libc10.dylib \
  ./torch/lib/libtorchbind_test.dylib \
  ./torch/lib/libshm.dylib \
  ./torch/lib/libaoti_custom_ops.dylib
do
  rm -f ${file}
done

echo "find . -type d -name '__pycache__' -exec rm -rf {};"
find . -type d -name '__pycache__' -exec rm -rf {} \; -print > /dev/null 2>&1

echo "find . -type f -name "*.so" -exec rm -f {};"
find . -type f -name "*.so" -exec rm -f {} \; -print > /dev/null 2>&1

echo "cat /dev/null > ${output_file}"
cat /dev/null > ${output_file}

echo "CMAKE_CUDA_COMPILER: ${CMAKE_CUDA_COMPILER}"
echo "CMAKE_CUDA_COMPILER: ${CMAKE_CUDA_COMPILER}" >> ${output_file}
echo "=============================================" >> ${output_file}

echo "python3 ./setup.py bdist_wheel --verbose"
python3 ./setup.py bdist_wheel --verbose >> ${output_file} 2>&1
rc=$?

if [ ${rc} -eq 0 ] ; then
  ts="`date +%Y%m%d%H%M%S`"
  mv build build-libtorch-linux-wheel-${ts}
  mv dist  dist-libtorch-linux-wheel-${ts}
fi

