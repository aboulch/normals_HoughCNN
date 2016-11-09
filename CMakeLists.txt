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

set(CMAKE_BUILD_TYPE Release)

############################
## OUTPUT DIRECTORIES
set( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin )
set( CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib )

############################
## CUDA
FIND_PACKAGE(CUDA REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")

############################
## OPENMP
if(UNIX)
	#use openMP
	find_package( OpenMP REQUIRED)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif(UNIX)

#################
## TORCH
FIND_PACKAGE(Torch REQUIRED)
INCLUDE_DIRECTORIES("${Torch_INSTALL_INCLUDE}/TH")
LINK_DIRECTORIES("${Torch_INSTALL_LIB}")
set(PROJECT_LINK_LIB ${PROJECT_LINK_LIB} ${Torch_INSTALL_LIB})

####################
## EXECUTABLE
add_executable(HoughCNN_Exec
main.cpp
houghCNN.cpp
houghCNN.h
pc_io.cpp
pc_io.h
timeMeasure.cpp
timeMeasure.h)
target_link_libraries( HoughCNN_Exec TH luajit luaT )
