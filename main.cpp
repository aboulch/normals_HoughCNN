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


#include<iostream>
#include<vector>
#include<string>

#include<time.h>
#include <cstdlib>
#include <ctime>

using namespace std;

#include "pc_io.h"

#include "houghCNN.h"


#include "boost/program_options.hpp"

int main(int argc, char** argv){
	srand (time(NULL));
	try{

		string input = "";
		string output = "out.ply";
		int nc = 1;
		int k = 100;
		int s = 33;
		int T = 1000;
		bool ua = false; // use aniso
		string model = "";

		// parse options
		namespace po = boost::program_options;
		po::options_description desc("Options");
		desc.add_options()
			("input,i",po::value<string>(&input)->required(), "input model ply")
			("output,o",po::value<string>(&output), "output model ply")
			("size,k",po::value<int>(&k) , "neighborhood size")
			("imSize,s", po::value<int>(&s), "accumulator size")
			("nbrT,t", po::value<int>(&T), "nbr T")
			("model,m",po::value<string>(&model)->required(), "model torch")
			("nbrScales,c", po::value<int>(&nc), "nbr of scales")
			("aniso,a", po::value<bool>(&ua), "use aniso")
			;
		po::variables_map vm;
		po::store(po::parse_command_line(argc, argv, desc),vm);
		po::notify(vm);

		if(nc!=1 && nc!=3 && nc!=5){
			cerr << "Error bad number of scales, should be 1,3 or 5" << endl;
		}

		// load the point cloud
		MatrixX3 pc, normals_gt, normals;
		ply_load(input,pc, normals_gt);

		// create the estimator
		NormEst ne(pc,normals);

		ne.access_A() = s; // accumulator size
		ne.access_T() = T; // number of hypothesis
		ne.access_K_aniso() = 5; // anisotropy nbr of neighborhoods

		// neighborhoods
		std::vector<int> Ks;
		Ks.push_back(k);
		if(nc==3){
			Ks.push_back(k/2);
			Ks.push_back(k*2);
		}else if(nc==5){
			Ks.push_back(k/4);
			Ks.push_back(k/2);
			Ks.push_back(k*2);
			Ks.push_back(k*4);
		}

		// estimation
		ne.estimate(model,Ks, ua);

		// save the point cloud
		ply_save(output,pc, normals);

	}catch(std::exception& e){
		std::cerr << "Unhandled Exception reached the top of main: "
				<< e.what() << ", application will now exit " << std::endl;
		return 1;
	}

	return 0;
}
