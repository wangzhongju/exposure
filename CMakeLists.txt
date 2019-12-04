cmake_minimum_required(VERSION 2.8)
project(exposure_test)
add_definitions(-std=c++11)
find_package(OpenCV REQUIRED)

add_executable(exposure_test main.cpp)
target_link_libraries(exposure_test ${OpenCV_LIBS})

