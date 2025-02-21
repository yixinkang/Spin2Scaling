cmake_minimum_required(VERSION 3.18)
project(Spin2GpeBCC LANGUAGES CXX CUDA)

file(GLOB SOURCES
    *.hpp
    *.cpp
	*.cu
	*.h
)

set(CMAKE_CUDA_ARCHITECTURES 52)

add_executable(Spin2GpeBCC ${SOURCES})

target_include_directories(Spin2GpeBCC PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/>
)

set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

target_link_libraries(Spin2GpeBCC Mesh)
target_link_libraries(Spin2GpeBCC Output)
target_link_libraries(Spin2GpeBCC Types)

set(HDF5_INCLUDE_DIR /cm/shared/apps/hdf5/1.14.0/include)
set(HDF5_LIB_DIR /cm/shared/apps/hdf5/1.14.0/lib)

include_directories(${HDF5_INCLUDE_DIR})
link_directories(${HDF5_LIB_DIR})

target_link_libraries(Spin2GpeBCC
  /cm/shared/apps/hdf5/1.14.0/lib/libhdf5_cpp.so)
target_link_libraries(Spin2GpeBCC
  /cm/shared/apps/hdf5/1.14.0/lib/libhdf5.so
)
