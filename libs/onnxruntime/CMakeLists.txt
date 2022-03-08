cmake_minimum_required(VERSION 3.13)
project(
  frvf_onnx
  VERSION 0.1
  LANGUAGES CXX
)
message("@@ frvf_onnx CMake Start @@")
set(CMAKE_BUILD_TYPE RelWithDebInfo)
set(CMAKE_CXX_STANDARD 14)
message("This lib name : ${PROJECT_NAME}")

include_directories(include)

add_library(${PROJECT_NAME} SHARED frvf_onnx.cpp)
message("ONNX RUNTIME INCLUDE DIRS : ${ONNX_RUNTIME_SESSION_INCLUDE_DIRS}")
message("OPENCV INCLUDE DIRS : ${OpenCV_INCLUDE_DIRS}")
message("ONNX RUNTIME LIB : ${ONNX_RUNTIME_LIB}")
message("OpenCV LIBRARIES : ${OpenCV_LIBRARIES}")
target_include_directories(${PROJECT_NAME} PUBLIC ${ONNX_RUNTIME_SESSION_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PUBLIC ${ONNX_RUNTIME_LIB} ${OpenCV_LIBRARIES})
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)

message("@@ frvf_onnx CMake Finish @@\n\n")