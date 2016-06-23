cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
project(HoughCNN_project)

SET(CMAKE_BUILD_TYPE Release)

add_definitions(-std=c++11)

set( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin )
set( CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib )

SET(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" "${CMAKE_MODULE_PATH}")
set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} "/usr/lib/x86_64-linux-gnu/")
set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} "/usr/local/cuda/lib64/")
include_directories(/usr/local/cuda/include)

############################
## OPENMP
############################
if(UNIX)
	#use openMP
	find_package( OpenMP REQUIRED)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif(UNIX)


#################
## LUA LIB
#################
#set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")
find_package(Lua 5.1 REQUIRED)
include_directories(${LUA_INCLUDE_DIR})
link_directories(${LUA_LIBRARIES})
set(PROJECT_LINK_LIB ${PROJECT_LINK_LIB} ${LUA_LIBRARIES})

FIND_PACKAGE(Torch REQUIRED)
INCLUDE_DIRECTORIES("${Torch_INSTALL_INCLUDE}/TH")
LINK_DIRECTORIES("${Torch_INSTALL_LIB}")
set(PROJECT_LINK_LIB ${PROJECT_LINK_LIB} ${Torch_INSTALL_LIB})

LINK_DIRECTORIES("/usr/local/lib")


add_executable(HoughCNN_Exec
main.cpp
houghCNN.cpp
houghCNN.h
pc_io.cpp
pc_io.h
timeMeasure.cpp
timeMeasure.h)
target_link_libraries(HoughCNN_Exec ${EXTRA_LIBS} ${Boost_LIBRARIES} ${PCL_LIBRARIES} ${PROJECT_LINK_LIB} thpp TH luaT THC  )
