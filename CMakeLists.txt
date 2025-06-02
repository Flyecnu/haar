cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(haar)

set(CMAKE_CXX_STANDARD 17)

# Include dirs
include_directories(${PROJECT_SOURCE_DIR}/include)

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

message(STATUS "OpenCV libraries: ${OpenCV_LIBRARIES}")

# Release优化
if(CMAKE_BUILD_TYPE STREQUAL "Release")
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-O3)
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    add_compile_options(/O2)
  endif()
endif()

# ✅ 编译 main.cpp
add_executable(main src/main.cpp)
target_link_libraries(main ${OpenCV_LIBRARIES})

# ✅ 编译 detect.cpp
add_executable(detect src/detect.cpp)
target_link_libraries(detect ${OpenCV_LIBRARIES})


add_executable(detect_haar_roi src/haar_tracking_roi.cpp)
target_link_libraries(detect_haar_roi ${OpenCV_LIBRARIES})



# ✅ 编译 detect_haar_threshold_roi.cpp（阈值分割+ROI+Haar检测）
add_executable(detect_haar_threshold_roi src/detect_haar_threshold_roi.cpp)
target_link_libraries(detect_haar_threshold_roi ${OpenCV_LIBRARIES})

add_executable(detect_drone_template src/detect_drone_template.cpp)
target_link_libraries(detect_drone_template ${OpenCV_LIBRARIES})

add_executable(detect_drone_with_box_estimation src/detect_drone_with_box_estimation.cpp)
target_link_libraries(detect_drone_with_box_estimation ${OpenCV_LIBRARIES})
