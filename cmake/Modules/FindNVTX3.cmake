# Script to find the NVTX3 include directory (Version 3.0, header-only)
# NVTX also depends on lib dl.
# Defines the following:
#   NVTX_FOUND       - if the package is found
#   NVTX_INCLUDE_DIR - the directory for 'nvtx3'
#   NVTX_LIBRARIES   - the libraries needed to linked for using NVTX
#
# For use with HTGS, add the '-DUSE_NVTX' during compilation to enable
# profiling with NVTX.
# If your HTGS graph contains more than 24 tasks, then use both '-DUSE_NVTX'
# and '-DUSE_MINIMAL_NVTX'
# The visualization tool NsightSystems must be used to profile and display
# the output.

include(FindPackageHandleStandardArgs)
unset(NVTX_FOUND)

# typical root dirs of installations, exactly one of them is used

find_path(NVTX_INCLUDE_DIR
          NAMES
          nvtx3/nvToolsExt.h
          HINTS
          /usr/local/include
          /usr/local/cuda/include/nvtx3
          /usr/include)

# set NVTX_FOUND
find_package_handle_standard_args(NVTX3 DEFAULT_MSG NVTX_INCLUDE_DIR)

# set external variables for usage in CMakeLists.txt
if (NVTX3_FOUND)
  set(NVTX_FOUND true)
  set(nvtx3_dir ${NVTX_INCLUDE_DIR})
  message(STATUS "NVTX3 Found. NVTX_INCLUDE_DIR: ${NVTX_INCLUDE_DIR}")
  message(STATUS "nvtx3_dir: ${nvtx3_dir}")
  mark_as_advanced(nvtx3_dir)
  mark_as_advanced(NVTX_INCLUDE_DIR)
else()
  message(STATUS "NVTX3 not found.")
endif()

