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
 *       NANOFLANN, EIGEN, LUA TORCH
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


int main(int argc, char** argv){
	srand (time(NULL));

		string input = "";
		string output = "out.xyz";
		int nc = 1;
		int k = 100;
		int s = 33;
		int T = 1000;
		bool ua = false; // use aniso
		string model = "-1";
		int k_density = 5;

		int c;

        opterr = 0;
        while ((c = getopt (argc, argv, "i:o:m:k:t:d:p:r:a:e:")) != -1)
        switch (c){
            case 'i':{
                input = optarg;
                break;
            }
            case 'o':{
                output = optarg;
                break;
            }
            case 'm':{
                model = optarg;
                break;
            }
            case 'k':{
                stringstream sstr("");
                sstr << optarg;
                sstr >> k;
                break;
            }
            case 't':{
                stringstream sstr("");
                sstr << optarg;
                sstr >> T;
                break;
            }
            case 'd':{
                stringstream sstr("");
                sstr << optarg;
                sstr >> ua;
                break;
            }
            case 'c':{
                stringstream sstr("");
                sstr << optarg;
                sstr >> nc;
                break;
            }
            case 's':{
                stringstream sstr("");
                sstr << optarg;
                sstr >> s;
                break;
            }
            case 'e':{
                stringstream sstr("");
                sstr << optarg;
                sstr >> k_density;
                break;
            }
            default:{
                cout << "Unknown option character" << endl;
                return 1;
                break;
            }
            }

        if(input=="-1"){
            cout << "Error need input file" << endl;
            return 1;
        }
        if(model=="-1"){
            cout << "Error need model file" << endl;
            return 1;
        }

		if(nc!=1 && nc!=3 && nc!=5){
			cerr << "Error bad number of scales, should be 1,3 or 5" << endl;
		}

		// load the point cloud
		MatrixX3 pc, normals;
		pc_load(input,pc);

		// create the estimator
		NormEst ne(pc,normals);

		ne.access_A() = s; // accumulator size
		ne.access_T() = T; // number of hypothesis
		ne.access_K_aniso() = k_density; // anisotropy nbr of neighborhoods

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
		pc_save(output,pc, normals);



	return 0;
}
