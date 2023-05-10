cmake_minimum_required(VERSION 3.0)

project(main)

set(CMAKE_CXX_STANDARD 17)

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Find Eigen3 package
find_package(Eigen3 REQUIRED)

# Include OpenCV headers
include_directories(${OpenCV_INCLUDE_DIRS})

# Include Eigen3 headers
include_directories(${EIGEN3_INCLUDE_DIRS})

# Set source files
set(SOURCES
    src/main.cc
    include/image_dataset_handler.cc
    include/image_dataset_handler.hpp
    )

# Create executable target
add_executable(main ${SOURCES})

# Link with OpenCV libraries
target_link_libraries(main ${OpenCV_LIBS} Eigen3::Eigen)

