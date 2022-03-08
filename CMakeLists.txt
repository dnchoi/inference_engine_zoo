cmake_minimum_required(VERSION 3.13)

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

message(STATUS "Configuration: ${CMAKE_BUILD_TYPE}") # -- Configuration: Debug
message(STATUS "Compiler")
message(STATUS " - ID       \t: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS " - Version  \t: ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS " - Path     \t: ${CMAKE_CXX_COMPILER}\n\n")


set (onnx_Header "${CMAKE_CURRENT_SOURCE_DIR}/libs/onnxruntime/include")
set (logger_Header "${CMAKE_SOURCE_DIR}/libs/spdlog/include")
set (config_Header "${CMAKE_SOURCE_DIR}/libs/configparser/include")

message("${onnx_Header}")
message("${logger_Header}")
message("${config_Header}")

include_directories(
    ${onnx_Header}
    ${logger_Header}
    ${config_Header}
)

find_package(OpenCV REQUIRED)
find_path(ONNX_RUNTIME_SESSION_INCLUDE_DIRS onnxruntime_cxx_api.h HINTS /usr/local/include/onnxruntime/core/session/)
find_library(ONNX_RUNTIME_LIB onnxruntime HINTS /usr/local/lib)

#INIT
# Lib 등록을 위한 작업 -> 해당 Lib Cmake 동작 후 build
add_subdirectory(
    libs/onnxruntime
)
message("ONNX runtime lib")
add_subdirectory(
    libs/configparser
)
message("configparser lib")
# add_subdirectory(
#     lib/ThreadPool
# )
# message("Thread pool shared lib build done")
# #INIT

add_executable (${PROJECT_NAME} main.cpp)

# include lib file in program
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

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)