cmake_minimum_required(VERSION 3.5)
project(DBoW3)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3")

find_package(OpenCV 4 QUIET)
if(NOT OpenCV_FOUND)
   find_package(OpenCV 3.0 QUIET)
   if(NOT OpenCV_FOUND)
      message(FATAL_ERROR "OpenCV > 3.0 not found.")
   endif()
endif()

set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

include_directories(${OpenCV_INCLUDE_DIRS})

add_library(${PROJECT_NAME} SHARED
  src/BowVector.cpp
  src/Database.cpp
  src/DescManip.cpp
  src/FeatureVector.cpp
  src/QueryResults.cpp
  src/ScoringObject.cpp
  src/Vocabulary.cpp
  src/quicklz.c
)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
