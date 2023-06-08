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

# Find PCL package
find_package(PCL  REQUIRED COMPONENTS common visualization)

# Add the include directories for the PCL headers
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})


# Set source files
set(SOURCES
    src/main.cc
    include/image_dataset_handler.cc
    include/image_dataset_handler.hpp
    )


# Create executable target
add_executable(main ${SOURCES})

add_executable(pointcloud_to_image 
    src/poiintcloud_to_image.cc
    include/image_dataset_handler.hpp
)
# Link with OpenCV libraries
target_link_libraries(main ${OpenCV_LIBS} Eigen3::Eigen  ${PCL_LIBRARIES} )
target_link_libraries(pointcloud_to_image ${OpenCV_LIBS} Eigen3::Eigen  ${PCL_LIBRARIES})

