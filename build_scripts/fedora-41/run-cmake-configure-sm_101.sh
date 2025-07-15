#!/bin/bash

pytorch_version="2.7.1"
prefix="/opt/pytorch/${pytorch_version}"
here="`pwd`"
topdir="`dirname ${here}`"
srcdir="${topdir}/pytorch-${pytorch_version}"
configure_output="${here}/pytorch-configure.out"
libnvrtc_filepath="/usr/local/cuda-12.9/targets/x86_64-linux/lib/libnvrtc.so"
cret=0
distro="`cat /etc/os-release | grep "^ID=" | cut -d= -f2`"

bfd_linker_flags="-Wl,--no-relax -Wl,-z -Wl,noreloc-overflow"
bfd_linker_flags="${bfd_linker_flags} -Wl,--export-dynamic"
bfd_linker_flags="${bfd_linker_flags} -Wl,-rpath,${prefix}/lib64"
bfd_linker_flags="${bfd_linker_flags} -Wl,-rpath,${prefix}/lib"
exe_linker_flags="-Wl,--no-relax"
exe_linker_flags="${exe_linker_flags} -Wl,-rpath,${prefix}/lib64"
exe_linker_flags="${exe_linker_flags} -Wl,-rpath,${prefix}/lib"
lld_linker_flags="-Wl,--no-relax"

linker_type="BFD"

if [ "${distro}" != "fedora" ] ; then
  echo "This cmake configure script only works on Fedora."
  exit 1
fi

if [ -e /etc/profile.d/modules.sh ] ; then
  source /etc/profile.d/modules.sh
fi

module load mpi/openmpi-x86_64
export CUDATOP="/usr/local/cuda-12.9"
export CUDA_MAJOR=12
export CUDA_MINOR=9
export PATH="${CUDATOP}/bin:/usr/lib64/openmpi/bin:/usr/bin:/usr/sbin:/usr/local/bin:${PATH}"

export CC="/usr/bin/gcc"
export CXX="/usr/bin/g++"
export CFLAGS="-Wall -Wextra"
export CXXFLAGS="-Wall -Wextra"
export CPPFLAGS="-D_GNU_SOURCE -D_XOPEN_SOURCE=700"

${CC} --version
${CXX} --version

# export CMAKE_FLAGS="--trace --trace-expand --trace-format=human"
export CMAKE_FLAGS=""

cmake_flags="-DCMAKE_INSTALL_PREFIX=${prefix}"
cmake_flags="${cmake_flags} -DCMAKE_BUILD_TYPE=Release"
cmake_flags="${cmake_flags} -DCMAKE_C_COMPILER=${CC}"
cmake_flags="${cmake_flags} -DCMAKE_CXX_COMPILER=${CXX}"
cmake_flags="${cmake_flags} -DCMAKE_C_FLAGS=${CPPFLAGS} ${CFLAGS}"
cmake_flags="${cmake_flags} -DCMAKE_CXX_FLAGS=${CPPFLAGS} ${CXXFLAGS}"
cmake_flags="${cmake_flags} -DCMAKE_C_FLAGS_RELEASE=${CPPFLAGS} ${CFLAGS}"
cmake_flags="${cmake_flags} -DCMAKE_CXX_FLAGS_RELEASE=${CPPFLAGS} ${CXXFLAGS}"
cmake_flags="${cmake_flags} -DCMAKE_C_STANDARD=11"
cmake_flags="${cmake_flags} -DCMAKE_CXX_STANDARD=17"
cmake_flags="${cmake_flags} -DCMAKE_C_EXTENSIONS:BOOL=ON"
cmake_flags="${cmake_flags} -DCMAKE_CXX_EXTENSIONS:BOOL=ON"
cmake_flags="${cmake_flags} -DCMAKE_LINKER_TYPE:STRING=${linker_type}"
cmake_flags="${cmake_flags} -DCMAKE_EXE_LINKER_FLAGS:STRING=${bfd_linker_flags}"
cmake_flags="${cmake_flags} -DCMAKE_SHARED_LINKER_FLAGS:STRING=${bfd_linker_flags}"
cmake_flags="${cmake_flags} -DCMAKE_MODULE_LINKER_FLAGS:STRING=${bfd_linker_flags}"
cmake_flags="${cmake_flags} -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON"
cmake_flags="${cmake_flags} -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON"
cmake_flags="${cmake_flags} -DCMAKE_SUPPRESS_REGENERATION:BOOL=ON"
cmake_flags="${cmake_flags} -DUSE_COLORIZE_OUTPUT:BOOL=OFF"
cmake_flags="${cmake_flags} -DBUILD_BINARY:BOOL=ON"
cmake_flags="${cmake_flags} -DBUILD_CUSTOM_PROTOBUF:BOOL=OFF"
cmake_flags="${cmake_flags} -DBUILD_PYTHON:BOOL=ON"
cmake_flags="${cmake_flags} -DBUILD_SHARED_LIBS:BOOL=ON"
cmake_flags="${cmake_flags} -DBUILD_TEST:BOOL=ON"
cmake_flags="${cmake_flags} -DBUILD_AOT_INDUCTOR_TEST:BOOL=OFF"
cmake_flags="${cmake_flags} -DBUILD_STATIC_RUNTIME_BENCHMARK:BOOL=OFF"
cmake_flags="${cmake_flags} -DUSE_CUDA:BOOL=ON"
cmake_flags="${cmake_flags} -DCUDA_VERSION=12.9"
cmake_flags="${cmake_flags} -DCUDA_MAJOR=12"
cmake_flags="${cmake_flags} -DCUDA_MINOR=9"
cmake_flags="${cmake_flags} -DCUDA_ARCH_LIST:STRING=10.1\;10.1a"
cmake_flags="${cmake_flags} -DTORCH_CUDA_ARCH_LIST:STRING=10.1\;10.1a"
cmake_flags="${cmake_flags} -DCUDA_nvrtc_LIBRARY:FILEPATH=${libnvrtc_filepath}"
cmake_flags="${cmake_flags} -DCUDA_NVRTC_LIB:FILEPATH=${libnvrtc_filepath}"
cmake_flags="${cmake_flags} -DCMAKE_CUDA_ARCHITECTURES:STRING=101;101a"
cmake_flags="${cmake_flags} -DBLAS=OpenBLAS"
cmake_flags="${cmake_flags} -DBUILD_LAZY_CUDA_LINALG:BOOL=OFF"
cmake_flags="${cmake_flags} -DUSE_ROCM:BOOL=OFF"
cmake_flags="${cmake_flags} -DUSE_CUDNN:BOOL=ON"
cmake_flags="${cmake_flags} -DCUSPARSELT_LIBRARY_PATH:FILEPATH=/usr/lib64/libcusparseLt.so"
cmake_flags="${cmake_flags} -DCUSPARSELT_INCLUDE_PATH:FILEPATH=/usr/include"
cmake_flags="${cmake_flags} -DUSE_CUSPARSELT:BOOL=ON"
cmake_flags="${cmake_flags} -DUSE_CUDSS:BOOL=ON"
cmake_flags="${cmake_flags} -DUSE_CUFILE:BOOL=ON"
cmake_flags="${cmake_flags} -DUSE_CUPTI_SO:BOOL=ON"
cmake_flags="${cmake_flags} -DUSE_TENSORRT:BOOL=ON"
cmake_flags="${cmake_flags} -DTENSORRT_LIBRARY:FILEPATH=/usr/lib64/libnvinfer.so"
cmake_flags="${cmake_flags} -DTENSORRT_INCLUDE_DIR:FILEPATH=/usr/include"

# GLOO
cmake_flags="${cmake_flags} -DUSE_TCP_OPENSSL_LINK_DEFAULT:BOOL=ON"
cmake_flags="${cmake_flags} -DUSE_TCP_OPENSSL_LOAD_DEFAULT:BOOL=OFF"
cmake_flags="${cmake_flags} -DGLOO_USE_CUDA_TOOLKIT:BOOL=ON"
cmake_flags="${cmake_flags} -DGLOO_USE_TORCH_DTYPES=1"
cmake_flags="${cmake_flags} -DBUILD_BENCHMARK:BOOL=OFF"
cmake_flags="${cmake_flags} -DGLOO_INSTALL:BOOL=ON"
# GLOO

# CUTLASS
cmake_flags="${cmake_flags} -DCMAKE_CUDA_COMPILER:FILEPATH=${CUDATOP}/bin/nvcc"
cmake_flags="${cmake_flags} -DCUTLASS_NVCC_EMBED_CUBIN:BOOL=ON"
cmake_flags="${cmake_flags} -DCUTLASS_NVCC_EMBED_PTX:BOOL=ON"
cmake_flags="${cmake_flags} -DCUTLASS_NVCC_VERBOSE:BOOL=ON"
cmake_flags="${cmake_flags} -DCUTLASS_ENABLE_F16C:BOOL=ON"
cmake_flags="${cmake_flags} -DCUTLASS_NVCC_ARCHS_ENABLED:STRING=101;101a"
cmake_flags="${cmake_flags} -DCUTLASS_LIBRARY_KERNELS:STRING=all"
cmake_flags="${cmake_flags} -DCUTLASS_LIBRARY_INSTANTIATION_LEVEL:STRING=max"
cmake_flags="${cmake_flags} -DCUTLASS_ENABLE_TENSOR_CORE_MMA_DEFAULT:BOOL=ON"
cmake_flags="${cmake_flags} -DCUTLASS_ENABLE_TENSOR_CORE_MMA:BOOL=ON"
cmake_flags="${cmake_flags} -DCUTLASS_ENABLE_CUBLAS:BOOL=ON"
cmake_flags="${cmake_flags} -DCUTLASS_ENABLE_CUDNN:BOOL=ON"
cmake_flags="${cmake_flags} -DCUTLASS_ENABLE_GDC_FOR_SM90:BOOL=ON"
cmake_flags="${cmake_flags} -DCUTLASS_ENABLE_GDC_FOR_SM100:BOOL=ON"
cmake_flags="${cmake_flags} -DCUTLASS_ENABLE_GDC_FOR_SM100_DEFAULT:BOOL=ON"
cmake_flags="${cmake_flags} -DCUTLASS_ENABLE_SM90_EXTENDED_MMA_SHAPES:BOOL=ON"
# CUTLASS

# SLEEF
cmake_flags="${cmake_flags} -DSLEEF_SHOW_CONFIG:BOOL=ON"
cmake_flags="${cmake_flags} -DSLEEF_ENABLE_LTO:BOOL=ON"
cmake_flags="${cmake_flags} -DSLEEF_BUILD_SCALAR_LIB:BOOL=ON"
cmake_flags="${cmake_flags} -DSLEEF_ENABLE_CUDA:BOOL=ON"
cmake_flags="${cmake_flags} -DSLEEF_ENABLE_CXX:BOOL=ON"
# SLEEF

cmake_flags="${cmake_flags} -DUSE_LITE_PROTO:BOOL=OFF"
cmake_flags="${cmake_flags} -DUSE_NCCL:BOOL=ON"
cmake_flags="${cmake_flags} -DUSE_SYSTEM_NCCL:BOOL=ON"
cmake_flags="${cmake_flags} -DUSE_XPU:BOOL=OFF"
cmake_flags="${cmake_flags} -DUSE_NUMA:BOOL=ON"
cmake_flags="${cmake_flags} -DUSE_NVRTC:BOOL=ON"
cmake_flags="${cmake_flags} -DUSE_OPENCL:BOOL=ON"
cmake_flags="${cmake_flags} -DUSE_OPENMP:BOOL=ON"
cmake_flags="${cmake_flags} -DUSE_SYSTEM_EIGEN_INSTALL:BOOL=ON"
cmake_flags="${cmake_flags} -DUSE_ITT:BOOL=ON"
cmake_flags="${cmake_flags} -DUSE_DISTRIBUTED:BOOL=ON"
cmake_flags="${cmake_flags} -DUSE_UCC:BOOL=OFF"
cmake_flags="${cmake_flags} -DUSE_GLOO:BOOL=ON"
cmake_flags="${cmake_flags} -DUSE_GLOO_WITH_OPENSSL:BOOL=ON"
cmake_flags="${cmake_flags} -DUSE_C10D_GLOO:BOOL=ON"
cmake_flags="${cmake_flags} -DUSE_C10D_NCCL:BOOL=ON"
cmake_flags="${cmake_flags} -DUSE_MPI:BOOL=ON"
cmake_flags="${cmake_flags} -DUSE_C10D_MPI:BOOL=ON"
cmake_flags="${cmake_flags} -DUSE_C10D_GLOO:BOOL=ON"
cmake_flags="${cmake_flags} -DUSE_C10D_NCCL:BOOL=ON"
cmake_flags="${cmake_flags} -DUSE_NUMPY:BOOL=ON"
cmake_flags="${cmake_flags} -DUSE_TENSORPIPE:BOOL=ON"
cmake_flags="${cmake_flags} -DHAVE_SOVERSION:BOOL=ON"
cmake_flags="${cmake_flags} -DUSE_CCACHE:BOOL=OFF"
cmake_flags="${cmake_flags} -DBUILD_FUNCTORCH:BOOL=ON"
cmake_flags="${cmake_flags} -DBUILD_BUNDLE_PTXAS:BOOL=ON"
cmake_flags="${cmake_flags} -DUSE_SYSTEM_LIBS:BOOL=ON"
cmake_flags="${cmake_flags} -DUSE_SYSTEM_PTHREADPOOL:BOOL=OFF"
cmake_flags="${cmake_flags} -DUSE_SYSTEM_CPUINFO:BOOL=ON"
cmake_flags="${cmake_flags} -DUSE_SYSTEM_PYBIND11:BOOL=ON"
cmake_flags="${cmake_flags} -DUSE_SYSTEM_NVTX:BOOL=ON"
cmake_flags="${cmake_flags} -DUSE_GOLD_LINKER:BOOL=OFF"

cat /dev/null > ${configure_output}
echo "Running: cmake ${CMAKE_FLAGS} ${cmake_flags} ${srcdir} >> ${configure_output} 2>&1"
cmake ${CMAKE_FLAGS} ${cmake_flags} ${srcdir} >> ${configure_output} 2>&1
cret=$?

if [ ${cret} -ne 0 ] ; then
  echo "CMake configuration failed."
  exit 1
fi

if [ -f ${here}/ptxas ] ; then
  echo "Really copying ptxas to ./bin/ ;-)"
  echo "Really copying ptxas to ./bin/ ;-)" >> ${configure_output} 2>&1
  cp -fp ${here}/ptxas ${here}/bin/
else
  echo "Warning: ptxas wasn't found and/or copied to ./bin/"
  echo "Warning: ptxas wasn't found and/or copied to ./bin/" >> ${configure_output} 2>&1
fi

echo "Fixing bad compiler flags from CMake ..."
echo "Fixing bad compiler flags from CMake ..." >> ${configure_output} 2>&1
listfile="/tmp/badcompilerflags.$$"
cat /dev/null > ${listfile}

find . -type f -name "*.make" -print >> ${listfile} 2>&1

while read -r line
do
  cp -fp ${line} "${line}.orig"
  sed -i 's#\"-g -O2\"#-g -O2#g' ${line}
  touch -r "${line}.orig" -acm ${line}
  rm -f "${line}.orig"
done < ${listfile}

while read -r line
do
  cp -fp ${line} "${line}.orig"
  sed -i 's#-O3#-O2#g' ${line}
  sed -i 's#-O1#-O2#g' ${line}
  sed -i 's#-fvisibility-inlines-hidden##g' ${line}
  sed -i 's#-fvisibility=hidden##g' ${line}
  sed -i 's#-msse2##g' ${line}
  sed -i 's#-Wl,--pic-executable##g' ${line}
  touch -r "${line}.orig" -acm ${line}
  rm -f "${line}.orig"
done < ${listfile}

rm -f ${listfile}

echo "Fixing bad link flags from CMake ..."
echo "Fixing bad link flags from CMake ..." >> ${configure_output} 2>&1

listfile="/tmp/link-relocations.$$"
cat /dev/null > ${listfile}

find . -type f -name "link.txt" -print >> ${listfile} 2>&1

while read -r line
do
  cp -fp ${line} "${line}.orig"
  sed -i 's#-Wl,--no-as-needed#-Wl,--as-needed#g' ${line}
  sed -i 's#-Wl,--whole-archive,#-Wl,--as-needed #g' ${line}
  sed -i 's#-Wl,--whole-archive#-Wl,--as-needed #g' ${line}
  sed -i 's#-Wl,--whole-archive#-Wl,--as-needed #g' ${line}
  sed -i 's#,--no-whole-archive# #g' ${line}
  sed -i 's#-Wl # #g' ${line}
  sed -i 's#-Wl,--no-whole-archive,#-Wl,--as-needed #g' ${line}
  sed -i 's#-Wl,--no-whole-archive#-Wl,--as-needed #g' ${line}
  sed -i 's#-Wl,--as-needed -Wl,--as-needed#-Wl,--as-needed#g' ${line}
  sed -i 's#"##g' ${line}
  sed -i 's#-Wl,--pic-executable##g' ${line}
  touch -r "${line}.orig" -acm ${line}
  rm -f "${line}.orig"
done < ${listfile}

rm -f ${listfile}
echo "Done."
echo "Done." >> ${configure_output} 2>&1

