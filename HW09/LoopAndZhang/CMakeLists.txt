# By FDL 11/14/2022
cmake_minimum_required(VERSION 2.8.12)

project(main)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(HEADER_FILES inc/util.hpp)
include_directories(inc/)

add_executable(main src/main.cpp src/util.cpp ${HEADER_FILES})
target_link_libraries(main ${OpenCV_LIBS})