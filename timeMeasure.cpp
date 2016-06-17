/* License Information
 *
 *  Copyright (C) ONERA, The French Aerospace Lab
 *  Author: Alexandre BOULCH
 *
 *  Permission is hereby granted, free of charge, to any person obtaining a copy of this
 *  software and associated documentation files (the "Software"), to deal in the Software
 *  without restriction, including without limitation the rights to use, copy, modify, merge,
 *  publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons
 *  to whom the Software is furnished to do so, subject to the following conditions:
 *
 *  The above copyright notice and this permission notice shall be included in all copies or
 *  substantial portions of the Software.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
 *  INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
 *  PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 *  LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 *  TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
 *  OR OTHER DEALINGS IN THE SOFTWARE.
 *
 *
 *  Note that this library relies on external libraries subject to their own license.
 *  To use this software, you are subject to the dependencies license, these licenses
 *  applies to the dependency ONLY  and NOT this code.
 *  Please refer below to the web sites for license informations:
 *       PCL, BOOST,NANOFLANN, EIGEN, LUA TORCH
 *
 * When using the software please aknowledge the  corresponding publication:
 * "Deep Learning for Robust Normal Estimation in Unstructured Point Clouds "
 * by Alexandre Boulch and Renaud Marlet
 * Symposium of Geometry Processing 2016, Computer Graphics Forum
 */

#include "timeMeasure.h"



#if 1
#if defined(_WIN32) || defined(_WIN64)

void TMeasure::tic(){
	GetSystemTime(&t_begin);
}

void TMeasure::tac(){
	GetSystemTime(&t_end);
}

size_t recons3D::TMeasure::elapsed(){
	return t_end.wMilliseconds - t_begin.wMilliseconds + 1000*(t_end.wSecond - t_begin.wSecond) + 60000*(t_end.wMinute - t_begin.wMinute);
}

#else

void TMeasure::tic(){
  gettimeofday(&t_begin, 0);
}

void TMeasure::tac(){
	gettimeofday(&t_end, 0);
}

unsigned int TMeasure::elapsed(){
	return (t_end.tv_sec-t_begin.tv_sec)*1000+(t_end.tv_usec-t_begin.tv_usec)/1000;
}

unsigned int TMeasure::elapsed_s(){
	return (t_end.tv_sec-t_begin.tv_sec);
}

#endif
#endif
