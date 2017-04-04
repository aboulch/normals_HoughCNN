# License Information
#
#  Copyright (C) ONERA, The French Aerospace Lab
#  Author: Alexandre BOULCH
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this
#  software and associated documentation files (the "Software"), to deal in the Software
#  without restriction, including without limitation the rights to use, copy, modify, merge,
#  publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons
#  to whom the Software is furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all copies or
#  substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
#  INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
#  PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
#  LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
#  TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
#  OR OTHER DEALINGS IN THE SOFTWARE.
#
#  Note that this library relies on external libraries subject to their own license.
#  To use this software, you are subject to the dependencies license, these licenses
#  applies to the dependency ONLY  and NOT this code.
#  Please refer below to the web sites for license informations:
#       NANOFLANN, EIGEN, TORCH
#
# When using the software please aknowledge the  corresponding publication:
# "Deep Learning for Robust Normal Estimation in Unstructured Point Clouds "
# by Alexandre Boulch and Renaud Marlet
# Symposium of Geometry Processing 2016, Computer Graphics Forum

cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
project(HoughCNN_project)

SET(CMAKE_BUILD_TYPE Release)

set( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin )
set( CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib )

set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} "/usr/lib/x86_64-linux-gnu/")


############################
## OPENMP
if(UNIX)
	#use openMP
	find_package( OpenMP REQUIRED)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif(UNIX)

include_directories("include")


# Make the scripts available in the 'cmake' directory available for the
# 'include()' command, 'find_package()' command.
set( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_LIST_DIR}/cmake )

# Include the CMake script UseCython.cmake.  This defines add_cython_module().
# Instruction for use can be found at the top of cmake/UseCython.cmake.
include( UseCython )

include_directories("python")

### build the libraies in the package directory
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python")

add_subdirectory(src)
