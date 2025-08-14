#!/bin/bash

distro="`uname -s`"

if [ "${distro}" != "Darwin" ] ; then
  echo "This Python Wheel Build script only works on MacOS."
  exit 1
fi

here="`pwd`"
srcdir="${here}"
basehere="`basename ${here}`"
builddir="${here}/build"
pytorch_version="2.7.1"
pytorch_string_version="2_7_1"
arch="`uname -m`"
macversion="macos-`uname -r`"
buildversion="700"
output_file="${here}/pytorch-wheel-libtorch-whl-build-macos.log"
kernellib="${here}/build/lib"
njobs=4
rc=0

if [ "${basehere}" != "pytorch-${pytorch_version}" ] ; then
  echo "You are in the wrong bdist_wheel build directory."
  exit 1
fi

if [ -e /opt/homebrew/bin/brew ] ; then
  /opt/homebrew/bin/brew shellenv >& /tmp/brewshellenv.$$
  source /tmp/brewshellenv.$$
  rm -f /tmp/brewshellenv.$$
fi

echo "find . -type d -name '__pycache__' -exec rm -rf {} \; -print > /dev/null 2>&1"
find . -type d -name '__pycache__' -exec rm -rf {} \; -print > /dev/null 2>&1

echo "rm -rf build-libtorch-macos libtorch_python_2.7.1_700.macos_24.5.0.arm64.egg-info dist-libtorch-macos build-pytorch-macos torch_python_2.7.1_700.macos_24.5.0.arm64.egg-info dist-pytorch-macos"
rm -rf build-libtorch-macos libtorch_python_2.7.1_700.macos_24.5.0.arm64.egg-info dist-libtorch-macos build-pytorch-macos torch_python_2.7.1_700.macos_24.5.0.arm64.egg-info dist-pytorch-macos

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

export PATH="/opt/homebrew/bin:/usr/bin:/bin:/usr/local/bin:/usr/sbin:${here}/bin"
export PATH="/opt/homebrew/opt/coreutils/libexec/gnubin:${PATH}"
export PATH="/opt/homebrew/opt/grep/libexec/gnubin:${PATH}"
export PATH="/opt/homebrew/opt/findutils/libexec/gnubin:${PATH}"
export PATH="/opt/homebrew/opt/diffutils/libexec/gnubin:${PATH}"
export PATH="/opt/homebrew/opt/openmpi/bin:${PATH}"
export PATH="/opt/homebrew/opt/cmake/bin:${PATH}"

export GMAKE="/opt/homebrew/bin/gmake"
export MAKE="${GMAKE}"
export MAKEFLAGS="j4"
export CMAKE="/opt/homebrew/opt/cmake/bin/cmake"
build_type="Release"

system_incdir="/usr/include"
system_libdir="/usr/lib"
pytorch_prefix="/opt/pytorch/${pytorch_version}"
pytorch_incdir="${pytorch_prefix}/include"
pytorch_libdir="${pytorch_prefix}/lib"
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
py_pytorch_site="/opt/homebrew/lib/python3.13/site-packages"
py_pytorch_libdir="${py_pytorch_site}/torch/lib"
py_pytorch_sitedir="${py_pytorch}"

lld_linker_flags="-Wl,-no_fixup_chains"
lld_linker_flags="${lld_linker_flags} -Wl,-undefined -Wl,dynamic_lookup"
lld_linker_flags="${lld_linker_flags} -Wl,--color-diagnostics=never"
exe_linker_flags="-Wl,-no_fixup_chains -Wl,--color-diagnostics=never"
linker_type="LLD"

cmake_install_rpath="${sleef_libdir};${openssl_libdir};${openblas_libdir};${openmpi_libdir};${openmp_libdir};${protobuf_libdir};${py_pytorch_libdir};${py_pytorch_sitedir};@loader_path;\$ORIGIN/../lib"

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

export MAX_JOBS=4
export USE_CUDA=0
export CFLAGS="-g -O3 -std=gnu11 -Wall -Wextra -fuse-ld=lld -fno-color-diagnostics"
export CXXFLAGS="-g -O3 -std=gnu++17 -Wall -Wextra -fuse-ld=lld -fno-color-diagnostics"
export LDFLAGS="${lld_linker_flags}"
export LDFLAGS="${LDFLAGS} -Wl,-rpath,${openblas_libdir}"
export LDFLAGS="${LDFLAGS} -Wl,-rpath,${openmpi_libdir}"
export LDFLAGS="${LDFLAGS} -Wl,-rpath,${openmp_libdir}"
export LDFLAGS="${LDFLAGS} -Wl,-rpath,${sleef_libdir}"
export LDFLAGS="${LDFLAGS} -Wl,-rpath,${abseil_libdir}"
export LDFLAGS="${LDFLAGS} -Wl,-rpath,${protobuf_libdir}"
export LDFLAGS="${LDFLAGS} -Wl,-rpath,${openssl_libdir}"
export CMAKE_BUILD_TYPE="${build_type}"
export CMAKE_MAKE_PROGRAM="${GMAKE}"
export CMAKE_C_COMPILER="${CC}"
export CMAKE_OBJC_COMPILER="${CC}"
export CMAKE_CXX_COMPILER="${CXX}"
export CMAKE_OBJCXX_COMPILER="${CXX}"
export CMAKE_C_FLAGS="${CFLAGS}"
export CMAKE_OBJC_FLAGS="${CFLAGS}"
export CMAKE_CXX_FLAGS="${CXXFLAGS}"
export CMAKE_OBJCXX_FLAGS="${CXXFLAGS}"
export CMAKE_C_FLAGS_RELEASE="${CFLAGS}"
export CMAKE_CXX_FLAGS_RELEASE="${CXXFLAGS}"
export CMAKE_EXE_LINKER_FLAGS="${exe_linker_flags}"
export CMAKE_SHARED_LINKER_FLAGS="${lld_linker_flags}"
export CMAKE_MODULE_LINKER_FLAGS="${lld_linker_flags}"
export CMAKE_LINKER_TYPE="LLD"
export CMAKE_GENERATOR="Unix Makefiles"
export CMAKE_C_STANDARD=11
export CMAKE_CXX_STANDARD=17
export CMAKE_C_EXTENSIONS=1
export CMAKE_CXX_EXTENSIONS=1
export CMAKE_LINKER_TYPE="${linker_type}"
export CMAKE_EXE_LINKER_FLAGS="${exe_linker_flags}"
export CMAKE_SHARED_LINKER_FLAGS="${lld_linker_flags}"
export CMAKE_MODULE_LINKER_FLAGS="${lld_linker_flags}"
export CMAKE_POSITION_INDEPENDENT_CODE=1
export CMAKE_VERBOSE_MAKEFILE=1
export CMAKE_SUPPRESS_REGENERATION=1
export CMAKE_INSTALL_RPATH="${cmake_install_rpath}"
export USE_COLORIZE_OUTPUT=0
export USE_PYTORCH_METAL=1
export USE_PYTORCH_METAL_EXPORT=0
export BUILD_BINARY=1
export BUILD_CUSTOM_PROTOBUF=0
export BUILD_PYTHON=1
export BUILD_SHARED_LIBS=1
export BUILD_TEST=1
export BUILD_AOT_INDUCTOR_TEST=0
export BUILD_STATIC_RUNTIME_BENCHMARK=0
export USE_CUDA=0
export USE_CUDNN=0
export BLAS="OpenBLAS"
export BUILD_LAZY_CUDA_LINALG=0
export USE_ROCM=0

# GLOO
export USE_TCP_OPENSSL_LINK=1
export USE_TCP_OPENSSL_LINK_DEFAULT=1
export USE_TCP_OPENSSL_LOAD=0
export USE_TCP_OPENSSL_LOAD_DEFAULT=0
export GLOO_USE_CUDA_TOOLKIT=0
export GLOO_USE_TORCH_DTYPES=1
export BUILD_BENCHMARK=0
export GLOO_INSTALL=1
# GLOO

# OpenBLAS
export PYTORCH_OPENBLAS="${openblas}"
# OpenBLAS

export INTERN_BUILD_MOBILE=0
export USE_LITE_PROTO=0
export USE_NCCL=0
export USE_SYSTEM_NCCL=0
export USE_XPU=0
export USE_NUMA=0
export USE_OPENCL=1
export USE_OPENMP=1
export USE_SYSTEM_EIGEN_INSTALL=1
export USE_ITT=1
export USE_DISTRIBUTED=1
export USE_UCC=0
export USE_GLOO=1
export USE_GLOO_WITH_OPENSSL=1
export USE_C10D_GLOO=1
export USE_C10D_NCCL=0
export USE_MPI=1
export USE_METAL=1
export USE_MPS=1
export USE_FBGEMM=1
export USE_C10D_MPI=1
export USE_C10D_GLOO=1
export USE_C10D_NCCL=0
export USE_NUMPY=1
export USE_TENSORPIPE=1
export HAVE_SOVERSION=1
export USE_CCACHE=0
export USE_MKLDNN=1
export USE_KLEIDIAI=1
export BUILD_FUNCTORCH=1
export BUILD_BUNDLE_PTXAS=0
export USE_SYSTEM_LIBS=1
export USE_SYSTEM_SLEEF=1
export USE_SYSTEM_EIGEN_INSTALL=1
export USE_SYSTEM_PTHREADPOOL=0
export USE_SYSTEM_CPUINFO=0
export USE_SYSTEM_PYBIND11=1
export USE_SYSTEM_NVTX=0
export COMPILER_SUPPORTS_OPENMP="True"
export USE_GOLD_LINKER=0
export Torch_DIR="${here}/torch/share/cmake/Torch"
export TORCH_USE_RTLD_GLOBAL=1
export ATEN_THREADING="OMP"
export USE_MIMALLOC=0

export BUILD_LIBTORCH_WHL=1
export BUILD_PYTHON_ONLY=0

export TORCH_PACKAGE_NAME="torch-python-${pytorch_version}-${buildversion}.${macversion}.${arch}"
export LIBTORCH_PACKAGE_NAME="libtorch-python-${pytorch_version}-${buildversion}.${macversion}.${arch}"
export TORCH_PACKAGE_VERSION="${pytorch_version}.${buildversion}"

echo "cat /dev/null > ${output_file}"
cat /dev/null > ${output_file}

echo "rm -rf ./build ./dist ${kernellib}"
rm -rf ./build ./dist ${kernellib}

echo "Clearing __pycache__ ..."
find . -type d -name '__pycache__' -exec rm -rf {} \; -print > /dev/null 2>&1

echo "env PATH=${PATH} MAKE=${GMAKE} MAKEFLAGS=${MAKEFLAGS} CMAKE=${CMAKE} PKG_CONFIG_PATH=${PKG_CONFIG_PATH} CMAKE_ARGS=\"${cmake_flags}\" python3 ./setup.py bdist_wheel --cmake-only --verbose"
env PATH=${PATH} MAKE=${GMAKE} MAKEFLAGS=${MAKEFLAGS} CMAKE=${CMAKE} PKG_CONFIG_PATH=${PKG_CONFIG_PATH} CMAKE_ARGS="${cmake_flags}" python3 ./setup.py bdist_wheel --cmake-only --verbose >> ${output_file} 2>&1
rc=$?

if [ ${rc} -ne 0 ] ; then
  echo "First stage python3 setup.py failed."
  exit 1
fi

if [ ! -d ${here}/build ] ; then
  echo "bdist_wheel --cmake-only failed."
  exit 1
fi

cd ${builddir}

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
  /opt/homebrew/bin/gsed -i 's#-O3#-O3 -fuse-ld=lld -Wno-unused-command-line-argument#g' ${line}
  /opt/homebrew/bin/gsed -i 's#-fvisibility-inlines-hidden##g' ${line}
  /opt/homebrew/bin/gsed -i 's#-fvisibility=hidden##g' ${line}
  /opt/homebrew/bin/gsed -i 's#-fvisibility=default##g' ${line}
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

if [ -d ${builddir}/sleef ] ; then
  echo "Fixing bad SLEEF PIC flags from CMake ..."
  echo "Fixing bad SLEEF PIC flags from CMake ..." >> ${output_file} 2>&1

  listfile="/tmp/badsleefpic.$$"

  cd ${builddir}/sleef
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
fi

cd ${here}

echo "Fixing bad microkernels flags from CMake ..."
echo "Fixing bad microkernels flags from CMake ..." >> ${output_file} 2>&1

listfile="/tmp/badmicrokernels.$$"

cd ${builddir}/confu-deps
cat /dev/null > ${listfile}

find . -type f -name "*.make" -print >> ${listfile}
find . -type f -name "link.txt" -print >> ${listfile} 2>&1

while read -r line
do
  cp -fp ${line} "${line}.orig"
  /opt/homebrew/bin/gsed -i 's#-Wno-unused-command-line-argument#-Wno-unused-command-line-argument -fno-color-diagnostics#g' ${line}
  touch -r "${line}.orig" -acm ${line}
  rm -f "${line}.orig"
done < ${listfile}

rm -f ${listfile}
cd ${here}

echo "Fixing bad caffe2 flags from CMake ..."
echo "Fixing bad caffe2 flags from CMake ..." >> ${output_file} 2>&1

listfile="/tmp/badcaffe2flags.$$"

cd ${builddir}/caffe2
cat /dev/null > ${listfile}

find . -type f -name "*.make" -print >> ${listfile}
find . -type f -name "link.txt" -print >> ${listfile} 2>&1

while read -r line
do
  cp -fp ${line} "${line}.orig"
  /opt/homebrew/bin/gsed -i 's#-ftree-vectorize#-ftree-vectorize -fno-color-diagnostics#g' ${line}
  /opt/homebrew/bin/gsed -i 's#-ftree-vectorize#-ftree-vectorize -fuse-ld=lld#g' ${line}
  /opt/homebrew/bin/gsed -i 's#-Werror=missing-prototypes#-fuse-ld=lld#g' ${line}
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
  /opt/homebrew/bin/gsed -i 's#-Wl,-headerpad_max_install_names##g' ${line}
  /opt/homebrew/bin/gsed -i 's#-Wl,-search_paths_first##g' ${line}
  /opt/homebrew/bin/gsed -i 's#-lsleef#/opt/homebrew/opt/sleef/lib/libsleef.dylib#g' ${line}
  /opt/homebrew/bin/gsed -i 's#pthreadpool_interface#pthreadpool#g' ${line}
  touch -r "${line}.orig" -acm ${line}
  rm -f "${line}.orig"
done < ${listfile}

rm -f ${listfile}

echo "Creating Metal Kernels ..."
cd ${here}
kerneldir="${here}/aten/src/ATen/native/mps/kernels"
kernellist=""

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
  kernellist="${kernellist} ${kerneldir}/${file}"
done

if [ ! -d ${kernellib} ] ; then
  mkdir -p ${kernellib}
fi

echo "xcrun --verbose --log -sdk macosx metal -dynamiclib ${kernellist} -I${srcdir} -o ${kernellib}/libKernelBasic.metallib -install_name libKernelBasic.metallib -std=metal3.1"
xcrun --verbose --log -sdk macosx metal -dynamiclib ${kernellist} -I${srcdir} -o ${kernellib}/libKernelBasic.metallib -install_name libKernelBasic.metallib -std=metal3.1 >> ${output_file} 2>&1
echo "xcrun --verbose --log -sdk macosx metal -dynamiclib ${kernellist} -I${srcdir} -o ${kernellib}/libKernelBFloat.metallib -install_name libKernelBFloat.metallib -std=metal3.1"
xcrun --verbose --log -sdk macosx metal -dynamiclib ${kernellist} -I${srcdir} -o ${kernellib}/libKernelBFloat.metallib -install_name libKernelBFloat.metallib -std=metal3.1 >> ${output_file} 2>&1

cd ${here}

echo "python3 ./setup.py bdist_wheel --verbose"
python3 ./setup.py bdist_wheel --verbose >> ${output_file} 2>&1
rc=$?

if [ ${rc} -eq 0 ] ; then
  ts="`date +%Y%m%d%H%M%S`"
  mv build build-libtorch-macos-wheel-${ts}
  mv dist dist-libtorch-macos-wheel-${ts}
fi


