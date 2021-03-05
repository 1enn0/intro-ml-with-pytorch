cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(esc-app)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(esc-app src/main.cpp)
target_link_libraries(esc-app "${TORCH_LIBRARIES}")

set_target_properties(esc-app
PROPERTIES 
    CXX_STANDARD 17
)

# On Windows, the LibTorch DLLs need to be copied to the binary dir
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET esc-app
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:esc-app>)
endif (MSVC)