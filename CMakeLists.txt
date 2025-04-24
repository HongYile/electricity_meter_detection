cmake_minimum_required(VERSION 3.30)
project(electricity_meter_detection)

set(CMAKE_CXX_STANDARD 20)


find_package(OpenCV)

include_directories(
        ${OpenCV_INCLUDE_DIRS}
)


add_executable(electricity_meter_detection
        main.cpp
)

target_link_libraries(
        electricity_meter_detection ${OpenCV_LIBS}
)

target_include_directories(electricity_meter_detection PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)