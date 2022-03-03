cmake_minimum_required(VERSION 3.13)
project(
  configparser
  VERSION 0.1
  LANGUAGES CXX
)
message("@@ configparser CMake Start @@")
set(CMAKE_BUILD_TYPE RelWithDebInfo)
set(CMAKE_CXX_STANDARD 14)
message("This lib name : ${PROJECT_NAME}")

include_directories(include)

add_library(${PROJECT_NAME} SHARED configparser.cpp)
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)

message("@@ configparser CMake Finish @@\n\n")