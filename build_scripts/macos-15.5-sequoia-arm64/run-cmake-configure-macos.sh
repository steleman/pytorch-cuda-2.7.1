#!/bin/bash

pytorch_version="2.7.1"
here="`pwd`"
topdir="`dirname ${here}`"
srcdir="${topdir}/pytorch-${pytorch_version}"
output_file="${here}/pytorch-configure.out"
cret=0
distro="`uname -s`"
lld_linker_flags="-Wl,-no_fixup_chains"
lld_linker_flags="${lld_linker_flags} -Wl,-undefined -Wl,dynamic_lookup"
lld_linker_flags="${lld_linker_flags} -Wl,--color-diagnostics=never"
exe_linker_flags="-Wl,-no_fixup_chains -Wl,--color-diagnostics=never"
linker_type="LLD"

if [ "${distro}" != "Darwin" ] ; then
  echo "This cmake configure script only works on MacOS."
  exit 1
fi

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
openssl="/opt/homebrew/opt/openssl"
openssl_libdir="${openssl}/lib"
openssl_incdir="${openssl}/include"

lld_linker_flags="${lld_linker_flags} -Wl,-L,${sleef_libdir}"
lld_linker_flags="${lld_linker_flags} -Wl,-L,${abseil_libdir}"
lld_linker_flags="${lld_linker_flags} -Wl,-L,${protobuf_libdir}"
exe_linker_flags="${exe_linker_flags} -Wl,-L,${sleef_libdir}"
exe_linker_flags="${exe_linker_flags} -Wl,-L,${abseil_libdir}"
exe_linker_flags="${exe_linker_flags} -Wl,-L,${protobuf_libdir}"
exe_linker_flags="${exe_linker_flags} -Wl,-L,${openssl_libdir}"

cmake_install_rpath="${sleef_libdir};${openssl_libdir};${openblas_libdir};${openmpi_libdir};${openmp_libdir};${protobuf_libdir};${sleef_libdir};@loader_path;\$ORIGIN/../lib"

export PKG_CONFIG_PATH="${openblas_libdir}/pkgconfig"
export PKG_CONFIG_PATH="${PKG_CONFIG_PATH}:${sleef_libdir}/pkgconfig"
export PKG_CONFIG_PATH="${PKG_CONFIG_PATH}:${openmpi_libdir}/pkgconfig"
export PKG_CONFIG_PATH="${PKG_CONFIG_PATH}:${openssl_libdir}/pkgconfig"
export PKG_CONFIG_PATH="${PKG_CONFIG_PATH}:${openblas_libdir}/pkgconfig"
export PKG_CONFIG_PATH="${PKG_CONFIG_PATH}:${hb_python_libdir}/pkgconfig"
export PKG_CONFIG_PATH="${PKG_CONFIG_PATH}:${sleef_libdir}/pkgconfig"
export PKG_CONFIG_PATH="${PKG_CONFIG_PATH}:${eigen3_pkgconfig}/pkgconfig"
export PKG_CONFIG_PATH="${PKG_CONFIG_PATH}:${protobuf_libdir}/pkgconfig"
export PKG_CONFIG_PATH="${PKG_CONFIG_PATH}:${abseil_libdir}/pkgconfig"

export CC="/usr/bin/clang"
export CXX="/usr/bin/clang++"
export CFLAGS="-Wall -Wextra"
export CXXFLAGS="-Wall -Wextra"
export CPPFLAGS=""

${CC} --version
${CXX} --version

# export CMAKE_FLAGS="--trace --trace-expand --trace-format=human"
export CMAKE_FLAGS=""

prefix="/opt/pytorch/${pytorch_version}"
cmake_install_libdir="${prefix}/lib"
cmake_install_rpath="${cmake_install_rpath};${cmake_install_libdir}"

cmake_flags="-DCMAKE_INSTALL_PREFIX=${prefix}"
cmake_flags="${cmake_flags} -DCMAKE_INSTALL_LIBDIR=${cmake_install_libdir}"
cmake_flags="${cmake_flags} -DCMAKE_BUILD_TYPE=${build_type}"
cmake_flags="${cmake_flags} -DCMAKE_C_COMPILER=${CC}"
cmake_flags="${cmake_flags} -DCMAKE_CXX_COMPILER=${CXX}"
cmake_flags="${cmake_flags} -DCMAKE_C_FLAGS=${CFLAGS}"
cmake_flags="${cmake_flags} -DONNX_CMAKE_C_FLAGS=${CFLAGS}"
cmake_flags="${cmake_flags} -DCMAKE_CXX_FLAGS=${CXXFLAGS}"
cmake_flags="${cmake_flags} -DONNX_CMAKE_CXX_FLAGS=${CXXFLAGS}"
cmake_flags="${cmake_flags} -DCMAKE_C_FLAGS_RELEASE=${CFLAGS}"
cmake_flags="${cmake_flags} -DCMAKE_CXX_FLAGS_RELEASE=${CXXFLAGS}"
cmake_flags="${cmake_flags} -DCMAKE_LINKER_TYPE:STRING=${linker_type}"
cmake_flags="${cmake_flags} -DCMAKE_EXE_LINKER_FLAGS:STRING=${exe_linker_flags}"
cmake_flags="${cmake_flags} -DCMAKE_SHARED_LINKER_FLAGS:STRING=${lld_linker_flags}"
cmake_flags="${cmake_flags} -DCMAKE_MODULE_LINKER_FLAGS:STRING=${lld_linker_flags}"
cmake_flags="${cmake_flags} -DCMAKE_C_STANDARD=11"
cmake_flags="${cmake_flags} -DCMAKE_CXX_STANDARD=17"
cmake_flags="${cmake_flags} -DCMAKE_C_EXTENSIONS:BOOL=ON"
cmake_flags="${cmake_flags} -DCMAKE_CXX_EXTENSIONS:BOOL=ON"
cmake_flags="${cmake_flags} -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON"
cmake_flags="${cmake_flags} -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON"
cmake_flags="${cmake_flags} -DCMAKE_SUPPRESS_REGENERATION:BOOL=ON"
cmake_flags="${cmake_flags} -DCMAKE_INSTALL_RPATH:STRING=${cmake_install_rpath}"
cmake_flags="${cmake_flags} -DUSE_COLORIZE_OUTPUT:BOOL=OFF"
cmake_flags="${cmake_flags} -DUSE_PYTORCH_METAL:BOOL=ON"
cmake_flags="${cmake_flags} -DUSE_PYTORCH_METAL_EXPORT:BOOL=OFF"
cmake_flags="${cmake_flags} -DBUILD_BINARY:BOOL=ON"
cmake_flags="${cmake_flags} -DBUILD_CUSTOM_PROTOBUF:BOOL=OFF"
cmake_flags="${cmake_flags} -DBUILD_PYTHON:BOOL=ON"
cmake_flags="${cmake_flags} -DBUILD_SHARED_LIBS:BOOL=ON"
cmake_flags="${cmake_flags} -DBUILD_TEST:BOOL=ON"
cmake_flags="${cmake_flags} -DBUILD_AOT_INDUCTOR_TEST:BOOL=OFF"
cmake_flags="${cmake_flags} -DBUILD_STATIC_RUNTIME_BENCHMARK:BOOL=OFF"
cmake_flags="${cmake_flags} -DUSE_CUDA:BOOL=OFF"
cmake_flags="${cmake_flags} -DUSE_CUDNN=0"
cmake_flags="${cmake_flags} -DBLAS=OpenBLAS"
cmake_flags="${cmake_flags} -DBUILD_LAZY_CUDA_LINALG:BOOL=OFF"
cmake_flags="${cmake_flags} -DUSE_ROCM:BOOL=OFF"

# GLOO
cmake_flags="${cmake_flags} -DUSE_TCP_OPENSSL_LINK_DEFAULT:BOOL=ON"
cmake_flags="${cmake_flags} -DUSE_TCP_OPENSSL_LOAD_DEFAULT:BOOL=OFF"
cmake_flags="${cmake_flags} -DGLOO_USE_CUDA_TOOLKIT:BOOL=OFF"
cmake_flags="${cmake_flags} -DGLOO_USE_TORCH_DTYPES=1"
cmake_flags="${cmake_flags} -DBUILD_BENCHMARK:BOOL=OFF"
cmake_flags="${cmake_flags} -DGLOO_INSTALL:BOOL=ON"
# GLOO

# SLEEF
cmake_flags="${cmake_flags} -DSLEEF_SHOW_CONFIG:BOOL=ON"
cmake_flags="${cmake_flags} -DSLEEF_ENABLE_LTO:BOOL=OFF"
cmake_flags="${cmake_flags} -DSLEEF_BUILD_SCALAR_LIB:BOOL=ON"
cmake_flags="${cmake_flags} -DSLEEF_ENABLE_CUDA:BOOL=OFF"
cmake_flags="${cmake_flags} -DSLEEF_ENABLE_CXX:BOOL=ON"
# SLEEF

# OpenBLAS
cmake_flags="${cmake_flags} -DPYTORCH_OPENBLAS:FILEPATH=${openblas}"
# OpenBLAS

cmake_flags="${cmake_flags} -DTorch_DIR:FILEPATH=${srcdir}/torch/share/cmake/Torch"
cmake_flags="${cmake_flags} -DINTERN_BUILD_MOBILE:BOOL=OFF"
cmake_flags="${cmake_flags} -DUSE_LITE_PROTO:BOOL=OFF"
cmake_flags="${cmake_flags} -DUSE_NCCL:BOOL=OFF"
cmake_flags="${cmake_flags} -DUSE_SYSTEM_NCCL:BOOL=OFF"
cmake_flags="${cmake_flags} -DUSE_XPU:BOOL=OFF"
cmake_flags="${cmake_flags} -DUSE_NUMA:BOOL=OFF"
cmake_flags="${cmake_flags} -DUSE_OPENCL:BOOL=ON"
cmake_flags="${cmake_flags} -DUSE_OPENMP:BOOL=ON"
cmake_flags="${cmake_flags} -DUSE_SYSTEM_EIGEN_INSTALL:BOOL=ON"
cmake_flags="${cmake_flags} -DUSE_ITT:BOOL=ON"
cmake_flags="${cmake_flags} -DUSE_DISTRIBUTED:BOOL=ON"
cmake_flags="${cmake_flags} -DUSE_UCC:BOOL=OFF"
cmake_flags="${cmake_flags} -DUSE_FBGEMM:BOOL=ON"
cmake_flags="${cmake_flags} -DUSE_GLOO:BOOL=ON"
cmake_flags="${cmake_flags} -DUSE_GLOO_WITH_OPENSSL:BOOL=ON"
cmake_flags="${cmake_flags} -DUSE_C10D_GLOO:BOOL=ON"
cmake_flags="${cmake_flags} -DUSE_C10D_NCCL:BOOL=OFF"
cmake_flags="${cmake_flags} -DUSE_MPI:BOOL=ON"
cmake_flags="${cmake_flags} -DUSE_MPS:BOOL=ON"
cmake_flags="${cmake_flags} -DUSE_C10D_MPI:BOOL=ON"
cmake_flags="${cmake_flags} -DUSE_C10D_GLOO:BOOL=ON"
cmake_flags="${cmake_flags} -DUSE_C10D_NCCL:BOOL=OFF"
cmake_flags="${cmake_flags} -DUSE_NUMPY:BOOL=ON"
cmake_flags="${cmake_flags} -DUSE_TENSORPIPE:BOOL=ON"
cmake_flags="${cmake_flags} -DHAVE_SOVERSION:BOOL=ON"
cmake_flags="${cmake_flags} -DUSE_CCACHE:BOOL=OFF"
cmake_flags="${cmake_flags} -DUSE_MKLDNN:BOOL=ON"
cmake_flags="${cmake_flags} -DUSE_KLEIDIAI:BOOL=ON"
cmake_flags="${cmake_flags} -DBUILD_FUNCTORCH:BOOL=ON"
cmake_flags="${cmake_flags} -DBUILD_BUNDLE_PTXAS:BOOL=OFF"
cmake_flags="${cmake_flags} -DUSE_SYSTEM_LIBS:BOOL=ON"
cmake_flags="${cmake_flags} -DUSE_SYSTEM_SLEEF:BOOL=ON"
cmake_flags="${cmake_flags} -DUSE_SYSTEM_PTHREADPOOL:BOOL=OFF"
cmake_flags="${cmake_flags} -DUSE_SYSTEM_CPUINFO:BOOL=OFF"
cmake_flags="${cmake_flags} -DUSE_SYSTEM_PYBIND11:BOOL=ON"
cmake_flags="${cmake_flags} -DUSE_SYSTEM_NVTX:BOOL=OFF"
cmake_flags="${cmake_flags} -DCOMPILER_SUPPORTS_OPENMP:BOOL=True"
cmake_flags="${cmake_flags} -DUSE_GOLD_LINKER:BOOL=OFF"

cat /dev/null > ${output_file}
echo "Running: cmake ${CMAKE_FLAGS} ${cmake_flags} ${srcdir} >> ${output_file} 2>&1"
cmake ${CMAKE_FLAGS} ${cmake_flags} ${srcdir} >> ${output_file} 2>&1
cret=$?

if [ ${cret} -ne 0 ] ; then
  echo "CMake configuration failed."
  exit 1
fi

echo "Fixing bad compiler flags from CMake ..."
echo "Fixing bad compiler flags from CMake ..." >> ${output_file} 2>&1
listfile="/tmp/badcompilerflags.$$"

cat /dev/null > ${listfile}

find . -type f -name "*.make" -print >> ${listfile} 2>&1

while read -r line
do
  cp -fp ${line} "${line}.orig"
  /opt/homebrew/bin/gsed -i 's#\"-g -O2\"#-g -O2#g' ${line}
  touch -r "${line}.orig" -acm ${line}
  rm -f "${line}.orig"
done < ${listfile}

while read -r line
do
  cp -fp ${line} "${line}.orig"
  /opt/homebrew/bin/gsed -i 's#-O2#-O3#g' ${line}
  /opt/homebrew/bin/gsed -i 's#-Os#-O3#g' ${line}
  /opt/homebrew/bin/gsed -i 's#-fvisibility-inlines-hidden##g' ${line}
  /opt/homebrew/bin/gsed -i 's#-fvisibility=hidden##g' ${line}
  /opt/homebrew/bin/gsed -i 's#-fcolor-diagnostics#-fno-color-diagnostics#g' ${line}
  /opt/homebrew/bin/gsed -i 's#-Werror=return-type ##g' ${line}
  /opt/homebrew/bin/gsed -i 's#-Werror=non-virtual-dtor ##g' ${line}
  /opt/homebrew/bin/gsed -i 's#-Werror=braced-scalar-init ##g' ${line}
  /opt/homebrew/bin/gsed -i 's#-Werror=range-loop-construct##g' ${line}
  /opt/homebrew/bin/gsed -i 's#-Werror=bool-operation ##g' ${line}
  /opt/homebrew/bin/gsed -i 's#/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang #/usr/bin/clang #g' ${line}
  /opt/homebrew/bin/gsed -i 's#/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang++ #/usr/bin/clang++ #g' ${line}
  /opt/homebrew/bin/gsed -i 's#"-g -O3 -fuse-ld=lld -Wno-unused-command-line-argument -std=gnu11 -Wall -Wextra -DNDEBUG"#-g -O3 -fuse-ld=lld -Wno-unused-command-line-argument -std=gnu11 -Wall -Wextra -DNDEBUG#g' ${line}
  touch -r "${line}.orig" -acm ${line}
  rm -f "${line}.orig"
done < ${listfile}

rm -f ${listfile}

if [ -d ${here}/sleef ] ; then
  echo "Fixing bad SLEEF PIC flags from CMake ..."
  echo "Fixing bad SLEEF PIC flags from CMake ..." >> ${output_file} 2>&1

  listfile="/tmp/badsleefpic.$$"

  cd ${here}/sleef
  cat /dev/null > ${listfile}
  find . -type f -name "*.make" -print >> ${listfile}
  find . -type f -name "link.txt" -print >> ${listfile} 2>&1

  while read -r line
  do
    cp -fp ${line} "${line}.orig"
    /opt/homebrew/bin/gsed -i 's#-fPIE#-fno-pie#g' ${line}
    /opt/homebrew/bin/gsed -i 's#-fPIC#-fno-PIC -fno-pie#g' ${line}
    /opt/homebrew/bin/gsed -i 's#-rdynamic##g' ${line}
    /opt/homebrew/bin/gsed -i 's#-Wl,-search_paths_first##g' ${line}
    /opt/homebrew/bin/gsed -i 's#-Wl,-headerpad_max_install_names##g' ${line}
    /opt/homebrew/bin/gsed -i 's#-Wl,-no_fixup_chains##g' ${line}
    /opt/homebrew/bin/gsed -i 's#-flto=thin##g' ${line}
    /opt/homebrew/bin/gsed -i 's#-fcolor-diagnostics#-fno-color-diagnostics#g' ${line}
    touch -r "${line}.orig" -acm ${line}
    rm -f "${line}.orig"
  done < ${listfile}

  rm -f ${listfile}
  cd ${here}
fi

echo "Fixing bad caffe2 flags from CMake ..."
echo "Fixing bad caffe2 flags from CMake ..." >> ${output_file} 2>&1

listfile="/tmp/badcaffe2flags.$$"

cd ${here}/caffe2
cat /dev/null > ${listfile}

find . -type f -name "*.make" -print >> ${listfile}
find . -type f -name "link.txt" -print >> ${listfile} 2>&1

while read -r line
do
  cp -fp ${line} "${line}.orig"
  /opt/homebrew/bin/gsed -i 's#-ftree-vectorize#-ftree-vectorize -fno-color-diagnostics#g' ${line}
  /opt/homebrew/bin/gsed -i 's#-ftree-vectorize#-ftree-vectorize -fuse-ld=lld#g' ${line}
  /opt/homebrew/bin/gsed -i 's#-Werror=missing-prototypes##g' ${line}
  /opt/homebrew/bin/gsed -i 's#pthreadpool_interface#pthreadpool#g' ${line}
  touch -r "${line}.orig" -acm ${line}
  rm -f "${line}.orig"
done < ${listfile}

rm -f ${listfile}
cd ${here}

echo "Fixing bad linker flags from CMake ..."
echo "Fixing bad linker flags from CMake ..." >> ${output_file} 2>&1

listfile="/tmp/link-relocations.$$"
cat /dev/null > ${listfile}

find . -type f -name "link.txt" -print >> ${listfile} 2>&1

while read -r line
do
  cp -fp ${line} "${line}.orig"
  /opt/homebrew/bin/gsed -i 's#-Wl,-flat_namespace##g' ${line}
  /opt/homebrew/bin/gsed -i 's#-rdynamic##g' ${line}
  /opt/homebrew/bin/gsed -i 's#-fvisibility=hidden##g' ${line}
  /opt/homebrew/bin/gsed -i 's#-fvisibility=default##g' ${line}
  /opt/homebrew/bin/gsed -i 's#-fvisibility-inlines-hidden##g' ${line}
  /opt/homebrew/bin/gsed -i 's#pthreadpool_interface#pthreadpool#g' ${line}
  /opt/homebrew/bin/gsed -i 's#-lsleef#/opt/homebrew/opt/sleef/lib/libsleef.dylib#g' ${line}
  touch -r "${line}.orig" -acm ${line}
  rm -f "${line}.orig"
done < ${listfile}

rm -f ${listfile}

echo "Creating Metal Kernels ..."
kerneldir="${srcdir}/aten/src/ATen/native/mps/kernels"
filelist=""

for file in \
  BinaryKernel.metal \
  Bucketization.metal \
  CrossKernel.metal \
  FusedOptimizerOps.metal \
  Gamma.metal \
  HistogramKernel.metal \
  Im2Col.metal \
  Indexing.metal \
  LinearAlgebra.metal \
  Quantized.metal \
  RMSNorm.metal \
  RenormKernel.metal \
  Repeat.metal \
  SpecialOps.metal \
  TriangularOps.metal \
  UnaryKernel.metal \
  UnfoldBackward.metal \
  UpSample.metal
do
  filelist="${filelist} ${kerneldir}/${file}"
done

# Trying with Metal 3.1 ONLY. Disabling Metal 3.0.
# echo "xcrun --verbose --log -sdk macosx metal -dynamiclib ${filelist} -I${srcdir} -o ${here}/lib/libKernelBasic.metallib -install_name libKernelBasic.metallib -std=metal3.0"
# xcrun --verbose --log -sdk macosx metal -dynamiclib ${filelist} -I${srcdir} -o ${here}/lib/libKernelBasic.metallib -install_name libKernelBasic.metallib -std=metal3.0 >> ${output_file} 2>&1

# This doesn't work.
# echo "xcrun -sdk macosx metal-tt ${here}/lib/libKernelBasic.metalir.metallib -o ${here}/lib/libKernelBasic.metallib $(xcrun -sdk macosx metal-config --native-arch-flags --gpu-family=metal3)"
# xcrun -sdk macosx metal-tt ${here}/lib/libKernelBasic.metalir.metallib -o ${here}/lib/libKernelBasic.metallib $(xcrun -sdk macosx metal-config --native-arch-flags --gpu-family=metal3)

echo "xcrun --verbose --log -sdk macosx metal -dynamiclib ${filelist} -I${srcdir} -o ${here}/lib/libKernelBasic.metallib -install_name libKernelBasic.metallib -std=metal3.1"
xcrun --verbose --log -sdk macosx metal -dynamiclib ${filelist} -I${srcdir} -o ${here}/lib/libKernelBasic.metallib -install_name libKernelBasic.metallib -std=metal3.1 >> ${output_file} 2>&1

echo "xcrun --verbose --log -sdk macosx metal -dynamiclib ${filelist} -I${srcdir} -o ${here}/lib/libKernelBFloat.metallib -install_name libKernelBFloat.metallib -std=metal3.1"
xcrun --verbose --log -sdk macosx metal -dynamiclib ${filelist} -I${srcdir} -o ${here}/lib/libKernelBFloat.metallib -install_name libKernelBFloat.metallib -std=metal3.1 >> ${output_file} 2>&1

# This doesn't work.
# echo "xcrun -sdk macosx metal-tt ${here}/lib/libKernelBFloat.metalir.metallib -o ${here}/lib/libKernelBFloat.metallib $(xcrun -sdk macosx metal-config --native-arch-flags --gpu-family=metal3)"
# xcrun -sdk macosx metal-tt ${here}/lib/libKernelBFloat.metalir.metallib -o ${here}/lib/libKernelBFloat.metallib $(xcrun -sdk macosx metal-config --native-arch-flags --gpu-family=metal3)

echo "Done."
echo "Done." >> ${output_file} 2>&1

