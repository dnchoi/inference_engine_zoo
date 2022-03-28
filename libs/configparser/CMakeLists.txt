cmake_minimum_required(VERSION 3.13)
project(
  configparser
  VERSION 0.1
  LANGUAGES CXX
)
message(STATUS " - configparser CMake Start")
set(CMAKE_BUILD_TYPE RelWithDebInfo)
set(CMAKE_CXX_STANDARD 14)
message(STATUS " - This lib name : ${PROJECT_NAME}")

set(src_dir source)
message(STATUS " - source dir : ${CMAKE_CURRENT_SOURCE_DIR}/${src_dir}")
include_directories(include)

add_library(${PROJECT_NAME} SHARED ${src_dir}/configparser.cpp)
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)

message(STATUS " - configparser CMake Finish")