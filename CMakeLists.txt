cmake_minimum_required(VERSION 3.13)
set(WITH_INFERENCE_ENGINE "ONNX" CACHE STRING "What select inference engine?")
# 프로젝트 정보
project(
  main
  VERSION 0.1
  DESCRIPTION "Face recognition verify test"
  LANGUAGES CXX
)
add_compile_definitions(_DEBUG_)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
set(CMAKE_BUILD_TYPE RelWithDebInfo)
set(CMAKE_CXX_FLAGS -pthread)
set(CMAKE_CXX_STANDARD 14)
message("\n\n")

message(STATUS "Configuration: ${CMAKE_BUILD_TYPE}") # -- Configuration: Debug
message(STATUS "Compiler")
message(STATUS " - ID       \t: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS " - Version  \t: ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS " - Path     \t: ${CMAKE_CXX_COMPILER}")
message(STATUS " - Inference engine     \t: ${WITH_INFERENCE_ENGINE}")
message("\n\n")
message(STATUS "OpenCV")
find_package(OpenCV REQUIRED)
message("\n\n")

if(${WITH_INFERENCE_ENGINE} STREQUAL "TENSORRT")
    add_definitions(-DTENSORRT)
elseif(${WITH_INFERENCE_ENGINE} STREQUAL "ONNX")
    add_definitions(-DONNX)
else()
    message(FATAL_ERROR)
endif()

set (logger_Header "${CMAKE_SOURCE_DIR}/libs/spdlog/include")
set (config_Header "${CMAKE_SOURCE_DIR}/libs/configparser/include")

include_directories(
    ${logger_Header}
    ${config_Header}
)

message(STATUS "Utils")
message(STATUS " - Logger Header     \t: ${logger_Header}")
message(STATUS " - Configure Header     \t: ${config_Header}")

add_subdirectory(
    libs/configparser
)
message(STATUS " - configparser library Done\n\n")


if(${WITH_INFERENCE_ENGINE} STREQUAL "TENSORRT")
    find_package(CUDA REQUIRED)

    message(STATUS "TensorRT")
    set (trt_Header "${CMAKE_CURRENT_SOURCE_DIR}/libs/trtruntime/include")
    message(STATUS " - TensorRT Header     \t: ${trt_Header}")

    include_directories(/usr/local/cuda-11.1/include)
    link_directories(/usr/local/cuda-11.1/lib64)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")
    set(
        CUDA_NVCC_FLAGS
        ${CUDA_NVCC_FLAGS};
        -O3
        -gencode arch=compute_75,code=sm_75
        # -gencode arch=compute_62,code=sm_62
    )

    find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    # HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR} /usr/include
    HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR} /home/luke/SDK_Tools/TensorRT-7.2.3.4/targets/x86_64-linux-gnu/include
    PATH_SUFFIXES include)
    find_path(TENSORRT_INCLUDE_DIR NvInferPlugin.h
    HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES include)
    MESSAGE(STATUS "Found TensorRT headers at ${TENSORRT_INCLUDE_DIR}")
    find_library(TENSORRT_LIBRARY_INFER nvinfer
    HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR} /home/luke/SDK_Tools/TensorRT-7.2.3.4/targets/x86_64-linux-gnu/lib/
    PATH_SUFFIXES lib lib64 lib/x64 lib/aarch64-linux-gnu)
    find_library(TENSORRT_LIBRARY_INFER_PLUGIN nvinfer_plugin
    HINTS  ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR} /home/luke/SDK_Tools/TensorRT-7.2.3.4/targets/x86_64-linux-gnu/lib/
    PATH_SUFFIXES lib lib64 lib/x64 lib/aarch64-linux-gnu)
    find_library(TENSORRT_LIBRARY_PARSER nvparsers
    HINTS  ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR} /home/luke/SDK_Tools/TensorRT-7.2.3.4/targets/x86_64-linux-gnu/lib/
    PATH_SUFFIXES lib lib64 lib/x64 lib/aarch64-linux-gnu)
    find_library(TENSORRT_LIBRARY_ONNXPARSER nvonnxparser
    HINTS  ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR} /home/luke/SDK_Tools/TensorRT-7.2.3.4/targets/x86_64-linux-gnu/lib/
    PATH_SUFFIXES lib lib64 lib/x64 lib/aarch64-linux-gnu)
    set(TENSORRT_LIBRARY ${TENSORRT_LIBRARY_INFER} ${TENSORRT_LIBRARY_INFER_PLUGIN} ${TENSORRT_LIBRARY_PARSER} ${TENSORRT_LIBRARY_ONNXPARSER})
    # message(STATUS "Find TensorRT libs at ${TENSORRT_LIBRARY}")
    include(FindPackageHandleStandardArgs)
    find_package_handle_standard_args(
        TENSORRT 
        DEFAULT_MSG 
        TENSORRT_INCLUDE_DIR 
        TENSORRT_LIBRARY
        TENSORRT_LIBRARY_ONNXPARSER
    )

    message(STATUS " -> ${TENSORRT_LIBRARY_INFER}")
    message(STATUS " -> ${TENSORRT_LIBRARY_INFER_PLUGIN}")
    message(STATUS " -> ${TENSORRT_LIBRARY_PARSER}")
    message(STATUS " -> ${TENSORRT_LIBRARY_ONNXPARSER}")

    if(NOT TENSORRT_FOUND)
    message(ERROR
        "Cannot find TensorRT library.")
    endif()

    include_directories(
      ${TENSORRT_INCLUDE_DIR}
    )
    
    add_subdirectory(
        libs/trtruntime
    )

    message(STATUS " - TRT runtime library Done\n\n")

elseif(${WITH_INFERENCE_ENGINE} STREQUAL "ONNX")
    message(STATUS "ONNX")
    set (onnx_Header "${CMAKE_CURRENT_SOURCE_DIR}/libs/onnxruntime/include")

    message(STATUS " - ONNX Header     \t: ${onnx_Header}")

    find_path(ONNX_RUNTIME_SESSION_INCLUDE_DIRS onnxruntime_cxx_api.h HINTS /usr/local/include/onnxruntime/core/session/)
    find_library(ONNX_RUNTIME_LIB onnxruntime HINTS /usr/local/lib)

    add_subdirectory(
        libs/onnxruntime
    )
    message(STATUS " - ONNX runtime library Done\n\n")

else()
    message(FATAL_ERROR "Please select inference engine")
endif()


if(${WITH_INFERENCE_ENGINE} STREQUAL "TENSORRT")
    cuda_add_executable (${PROJECT_NAME} main.cpp)

    include_directories(
        ${trt_Header}
    )
    target_link_libraries(
        ${PROJECT_NAME}
        frvf_trt
        configparser
        pthread
    )
    target_include_directories(
        ${PROJECT_NAME} 
        PUBLIC 
            ${trt_Header}
            ${logger_Header}
            ${config_Header}
    )

elseif(${WITH_INFERENCE_ENGINE} STREQUAL "ONNX")
    add_executable (${PROJECT_NAME} main.cpp)
    
    include_directories(
        ${onnx_Header}
    )
    target_link_libraries(
        ${PROJECT_NAME}
        frvf_onnx
        configparser
        pthread
    )
    target_include_directories(
        ${PROJECT_NAME} 
        PUBLIC 
            ${onnx_Header}
            ${logger_Header}
            ${config_Header}
    )

else()
    message(FATAL_ERROR)
endif()

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)