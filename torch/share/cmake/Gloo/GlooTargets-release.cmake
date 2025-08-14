#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "gloo" for configuration "Release"
set_property(TARGET gloo APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(gloo PROPERTIES
  IMPORTED_LOCATION_RELEASE "/Volumes/CT4000P34TB/src/steleman/programming/libtorch/20250609/pytorch-2.7.1/torch/lib/libgloo.dylib"
  IMPORTED_SONAME_RELEASE "@rpath/libgloo.dylib"
  )

list(APPEND _cmake_import_check_targets gloo )
list(APPEND _cmake_import_check_files_for_gloo "/Volumes/CT4000P34TB/src/steleman/programming/libtorch/20250609/pytorch-2.7.1/torch/lib/libgloo.dylib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
