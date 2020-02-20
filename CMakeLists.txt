cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(FFRNN)

# CXX FLAGS
if(NOT CMAKE_BUILD_TYPE)
set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(PYBIND11_CPP_STANDARD -std=c++14)

# pybind11
find_package(pybind11 CONFIG REQUIRED)
include_directories(${pybind11_INCLUDE_DIRS})

message( [MAIN] "Found pybind11 v${pybind11_VERSION}: ${pybind11_INCLUDE_DIRS}")
message( [Main] " pybind11_INCLUDE_DIRS = ${pybind11_INCLUDE_DIRS}")
message( [Main] " pybind11_LIBRARIES = ${pybind11_LIBRARIES}")

# pytorch
find_package(Torch REQUIRED)
include_directories(${Torch_INCLUDE_DIRS})


add_library(ffrnn MODULE ffrnn.cc)
target_link_libraries(ffrnn PRIVATE pybind11::module
                                    "${TORCH_LIBRARIES}")
set_target_properties(ffrnn PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}" 
                                       SUFFIX "${PYTHON_MODULE_EXTENSION}")