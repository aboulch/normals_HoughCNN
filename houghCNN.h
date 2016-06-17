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

#ifndef NORMALS_EST_HEADER
#define NORMALS_EST_HEADER



// LUA
#include <TH.h>
#include <THTensor.h>
#include <luaT.h>
extern "C" {
#include "lua.h"
#include "lualib.h"
#include "lauxlib.h"
}

#include"Eigen/Dense"
#include<vector>

typedef Eigen::Vector3d Vector3;
typedef Eigen::Vector2d Vector2;
typedef Eigen::Matrix3d Matrix3;
typedef Eigen::Matrix2d Matrix2;
typedef Eigen::MatrixX3d MatrixX3;
typedef Eigen::MatrixX3i MatrixX3i;
typedef Eigen::VectorXd VectorX;
typedef Eigen::MatrixXd MatrixX;
typedef Eigen::Vector3i Vector3i;


class  NormEst{


private:
	const MatrixX3& _pc; //reference to the point cloud
	MatrixX3& _normals; //reference to the normal cloud

	int T; //number of triplets to pick, default is 700

	int K_aniso;

	int A; //side_size of accumulator

	std::vector<float> proba_vector;
	std::vector<int> counts_generated_elems;
	int batch_size;


public:
	std::vector<unsigned int> rand_ints;
	NormEst(const MatrixX3& pc, MatrixX3& normals):
		_pc(pc),
		_normals(normals),
		K_aniso(5),
		T(1000),
		A(33),
		batch_size(256)
		{}


	int estimate(const std::string& model, const std::vector<int>& Ks, bool use_aniso);


	int& access_T(){return T;}
	int& access_K_aniso(){return K_aniso;}
	int& access_A(){return A;}
	int& access_batchSize(){return batch_size;}

	const int& access_T()const {return T;}
	const int& access_K_aniso() const {return K_aniso;}
	const int& access_A() const{return A;}
	const int& access_batchSize() const{return batch_size;}

	const MatrixX3& pc() const{return _pc;}
	MatrixX3& normals(){return _normals;}

};


#endif
