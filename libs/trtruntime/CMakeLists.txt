cmake_minimum_required(VERSION 3.13)
project(
  frvf_trt
  VERSION 0.1
  LANGUAGES CXX
)

message(STATUS " - frvf_trt CMake Start")
set(CMAKE_BUILD_TYPE RelWithDebInfo)
set(CMAKE_CXX_STANDARD 17)
message(STATUS " - This lib name : ${PROJECT_NAME}")
set(src_dir source)
message(STATUS " - source dir : ${CMAKE_CURRENT_SOURCE_DIR}/${src_dir}")

find_package(CUDA)

# tensorRT
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES include)
MESSAGE(STATUS " - Found TensorRT headers at ${TENSORRT_INCLUDE_DIR}")

find_library(TENSORRT_LIBRARY_INFER nvinfer
  HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR} 
  PATH_SUFFIXES lib lib64 lib/x64)

find_library(TENSORRT_LIBRARY_INFER_PLUGIN nvinfer_plugin
  HINTS  ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 lib/x64)

find_library(TENSORRT_LIBRARY_PARSER nvparsers
  HINTS  ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 lib/x64)
  
find_library(TENSORRT_LIBRARY_ONNXPARSER nvonnxparser
  HINTS  ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 lib/x64)

set(TENSORRT_LIBRARY ${TENSORRT_LIBRARY_INFER} ${TENSORRT_LIBRARY_INFER_PLUGIN} ${TENSORRT_LIBRARY_PARSER} ${TENSORRT_LIBRARY_ONNXPARSER})

INCLUDE(FindPackageHandleStandardArgs)

find_package_handle_standard_args(
    TENSORRT 
    DEFAULT_MSG 
    TENSORRT_INCLUDE_DIR 
    TENSORRT_LIBRARY
    TENSORRT_LIBRARY_ONNXPARSER
)
message(STATUS " - ${TENSORRT_LIBRARY_INFER}")
message(STATUS " - ${TENSORRT_LIBRARY_INFER_PLUGIN}")
message(STATUS " - ${TENSORRT_LIBRARY_PARSER}")
message(STATUS " - ${TENSORRT_LIBRARY_ONNXPARSER}")

if(NOT TENSORRT_FOUND)
  message(ERROR
    "Cannot find TensorRT library.")
endif()

include_directories(include)

cuda_add_library(${PROJECT_NAME} SHARED ${src_dir}/frvf_trt.cpp)

target_include_directories(${PROJECT_NAME} PUBLIC ${OpenCV_INCLUDE_DIRS} ${TENSORRT_INCLUDE_DIR} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBRARIES} ${TENSORRT_LIBRARY})
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)

message(STATUS " - frvf_trt CMake Finish")