# ---[ Test binaries.

file(GLOB_RECURSE C10_ALL_TEST_FILES *_test.cpp)

if (BUILD_TEST)
  foreach(test_src ${C10_ALL_TEST_FILES})
    get_filename_component(test_file_name ${test_src} NAME_WE)
    set(test_name "c10_${test_file_name}")
    add_executable(${test_name} "${test_src}")
    if(NOT MSVC)
      target_compile_options(${test_name} PRIVATE -Wno-unused-variable)
    endif()

    if (${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")
      string(REPLACE "-fcolor-diagnostics" "" CMAKE_C_FLAGS ${CMAKE_C_FLAGS})
      string(REPLACE "-fcolor-diagnostics" "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
      string(APPEND CMAKE_C_FLAGS " -fno-color-diagnostics")
      string(APPEND CMAKE_CXX_FLAGS " -fno-color-diagnostics")
    endif()

    target_link_libraries(${test_name} ${C10_LIB} gmock gtest gtest_main)
    add_test(NAME ${test_name} COMMAND $<TARGET_FILE:${test_name}>)
    if(INSTALL_TEST)
      set_target_properties(${test_name} PROPERTIES INSTALL_RPATH "${CMAKE_INSTALL_RPATH}:${_rpath_portable_origin}/../lib")
      install(TARGETS ${test_name} DESTINATION test)
    endif()
  endforeach()
endif()
