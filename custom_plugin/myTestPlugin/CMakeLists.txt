# We need cmake >= 3.8, since 3.8 introduced CUDA as a first class language
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(MsDeformIm2ColTRT LANGUAGES CXX CUDA)

# Enable all compile warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-long-long -pedantic -Wno-deprecated-declarations")

# Sets variable to a value if variable is unset.
macro(set_ifndef var val)
    if (NOT ${var})
        set(${var} ${val})
    endif()
    message(STATUS "Configurable variable ${var} set to ${${var}}")
endmacro()

# -------- CONFIGURATION --------
set_ifndef(TRT_LIB /usr/lib/x86_64-linux-gnu)
set_ifndef(TRT_INCLUDE /usr/lib/x86_64-linux-gnu)
set_ifndef(CUDA_INC_DIR /usr/local/cuda/include)
set_ifndef(CUDA_ARCH_SM 75) # should be fine for Tesla V100

# Find dependencies:
message("\nThe following variables are derived from the values of the previous variables unless provided explicitly:\n")

# TensorRT's nvinfer lib
find_library(_NVINFER_LIB nvinfer HINTS ${TRT_LIB} PATH_SUFFIXES lib lib64)
set_ifndef(NVINFER_LIB ${_NVINFER_LIB})


# -------- BUILDING --------

# Add include directories
include_directories(${CUDA_INC_DIR} ${TRT_INCLUDE} ${CMAKE_SOURCE_DIR}/sources/)
message(STATUS "CUDA_INC_DIR: ${CUDA_INC_DIR}")
# Define plugin library target
add_library(mytest_trt MODULE
~/project/TensorRT/custom_plugin/myTestPlugin/source/checkMacrosPlugin.cpp
~/project/TensorRT/custom_plugin/myTestPlugin/source/myTestKernel.cu
~/project/TensorRT/custom_plugin/myTestPlugin/source/myTestPlugin.cpp
~/project/TensorRT/custom_plugin/myTestPlugin/source/plugin.h
~/project/TensorRT/custom_plugin/myTestPlugin/source/checkMacrosPlugin.h
~/project/TensorRT/custom_plugin/myTestPlugin/source/myTestPlugin.h
~/project/TensorRT/custom_plugin/myTestPlugin/source/myTestKernel.h
)

# Use C++11
target_compile_features(mytest_trt PUBLIC cxx_std_11)

# Link TensorRT's nvinfer lib
target_link_libraries(mytest_trt PRIVATE ${NVINFER_LIB})

# We need to explicitly state that we need all CUDA files
# to be built with -dc as the member functions will be called by
# other libraries and executables (in our case, Python inference scripts)
set_target_properties(mytest_trt PROPERTIES
CUDA_SEPARABLE_COMPILATION ON
)

# CUDA ARCHITECTURE
set_target_properties(mytest_trt PROPERTIES
CUDA_ARCHITECTURES "${CUDA_ARCH_SM}")