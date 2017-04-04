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


#include"Eigen/Dense"
#include<vector>
#include<fstream>

#include "nanoflann.hpp"

typedef Eigen::Vector3d Vector3;
typedef Eigen::Vector2d Vector2;
typedef Eigen::Matrix3d Matrix3;
typedef Eigen::Matrix2d Matrix2;
typedef Eigen::MatrixX3d MatrixX3;
typedef Eigen::MatrixX3i MatrixX3i;
typedef Eigen::VectorXd VectorX;
typedef Eigen::MatrixXd MatrixX;
typedef Eigen::Vector3i Vector3i;

class HoughAccum{
public:
	VectorX accum;
	MatrixX3 accum_vec;
	Matrix3 P;

	int A;

};

class  NormEst{

	typedef typename nanoflann::KDTreeEigenMatrixAdaptor< MatrixX3 > kd_tree;

private:
	MatrixX3 _pc; //reference to the point cloud
	MatrixX3 _normals; //reference to the normal cloud

	int T; //number of triplets to pick, default is 700

	int K_aniso;

	int A; //side_size of accumulator

	std::vector<float> proba_vector;
	std::vector<int> counts_generated_elems;

public:

	std::vector<int> Ks; // TODO

	void set_Ks(int* array, int m);
	int get_Ks_size(){return Ks.size();}
	void get_Ks(int* array, int m);

	bool use_aniso; // TODO
	bool get_density_sensitive(){return use_aniso;}
	void set_density_sensitive(bool d_s){use_aniso=d_s;}

	int maxK;
	kd_tree* tree;
	bool is_tree_initialized;
	unsigned int randPos;
	std::vector<HoughAccum> accums;

	void initialize();
	void get_batch(int batch_id, int batch_size, double* array);
	void set_batch(int batch_id, int batch_size, double* array);

	std::vector<unsigned int> rand_ints;

	NormEst():
		K_aniso(5),T(1000),A(33), is_tree_initialized(false){}

	~NormEst(){
		if(is_tree_initialized){
	        delete tree;
	    }
	}


	int size();
	int size_normals();

	int estimate(const std::string& model, const std::vector<int>& Ks, bool use_aniso);

	void set_T(int T_){T=T_;}
	void set_K_aniso(int Kaniso){K_aniso=Kaniso;}
	void set_A(int A_){A=A_;}

	const int get_T()const {return T;}
	const int get_K_aniso() const {return K_aniso;}
	const int get_A() const{return A;}

    void get_points(double* array, int m, int n);
    void get_normals(double* array, int m, int n);
    void set_points(double* array, int m, int n);
    void set_normals(double* array, int m, int n);

	const MatrixX3& pc() const{return _pc;}
	MatrixX3& normals(){return _normals;}



	// io
	void loadXYZ(const std::string& filename){
		std::ifstream istr(filename.c_str());
		std::vector<Eigen::Vector3d> points;
		std::string line;
		double x,y,z;
		while(getline(istr, line))
		{
			std::stringstream sstr("");
			sstr << line;
			sstr >> x >> y >> z;
			points.push_back(Eigen::Vector3d(x,y,z));
		}
		istr.close();
		_pc.resize(points.size(),3);
		for(int i=0; i<points.size(); i++){
			_pc.row(i) = points[i];
		}
	}

	void saveXYZ(const std::string& filename){
		std::ofstream ofs(filename.c_str());
		for(int i=0; i<_pc.rows(); i++){
			ofs << _pc(i,0) << " ";
			ofs << _pc(i,1) << " ";
			ofs << _pc(i,2) << " ";
			ofs << _normals(i,0) << " ";
			ofs << _normals(i,1) << " ";
			ofs << _normals(i,2) << std::endl;
		}
		ofs.close();
	}

};


#endif
