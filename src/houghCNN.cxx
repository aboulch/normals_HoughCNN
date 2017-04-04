// Deep Learning for Robust Normal Estimation in Unstructured Point Clouds
// Copyright (c) 2016 Alexande Boulch and Renaud Marlet
//
// This program is free software; you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the Free Software
// Foundation; either version 3 of the License, or any later version.
// This program is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
// You should have received a copy of the GNU General Public License along with this
// program; if not, write to the Free Software Foundation, Inc., 51 Franklin Street,
// Fifth Floor, Boston, MA 02110-1301  USA
//
// PLEASE ACKNOWLEDGE THE ORIGINAL AUTHORS AND PUBLICATION:
// "Deep Learning for Robust Normal Estimation in Unstructured Point Clouds "
// by Alexandre Boulch and Renaud Marlet, Symposium of Geometry Processing 2016,
// Computer Graphics Forum

#include "houghCNN.h"

// STL
#include <iostream>
#include <time.h>
#include <stdlib.h>
#include <string>
#include <unistd.h>
using namespace std;
using std::string;

// OPENMP
#include <omp.h>


typedef typename nanoflann::KDTreeEigenMatrixAdaptor< MatrixX3 > kd_tree;

int NormEst::size(){
	return _pc.rows();
}

int NormEst::size_normals(){
	return _normals.rows();
}

inline void fill_accum_aniso(HoughAccum& hd, std::vector<long int>& nbh, int nbh_size,
		const NormEst* est, unsigned int& randPos,
		const vector<float>& proba_vector,
		bool compute_P = true, Matrix3 P_ref=Matrix3())
{

	//references
	VectorX& accum = hd.accum;
	Matrix3& P = hd.P;
	MatrixX3 &accum_vec = hd.accum_vec;
	int& A = hd.A;

	//init
	A = est->get_A();
	accum_vec = MatrixX3::Zero(A*A,3);
	accum = VectorX::Zero(A*A);
	Vector3 mean = Vector3::Zero();
	Matrix3 cov = Matrix3::Zero();

	//other refs
	const MatrixX3& _pc = est->pc();
	const int T = est->get_T();
	const vector<unsigned int>& rand_ints = est->rand_ints;




	//regression
	float mean_sum = 0;
	for(int i=0; i<nbh_size; i++){
		const Vector3& v =  proba_vector[nbh[i]] * _pc.row(nbh[i]);
		mean+=v;
		mean_sum+=proba_vector[nbh[i]] ;
	}
	mean /= mean_sum;

	for(int i=0; i<nbh_size; i++){
		Vector3 v =  _pc.row(nbh[i]).transpose()-mean;
		cov+= v*v.transpose()*proba_vector[nbh[i]];
	}

	Eigen::JacobiSVD<Matrix3> svd(cov, Eigen::ComputeFullV);
	if(compute_P){
		P =  svd.matrixV().transpose();
	}else{
		P = P_ref;
	}
	//Vector3 nor_reg = P.row(2);


    float min_prob = 1e7;
    float max_prob = -1e7;
    for(int i=0; i<nbh_size; i++){
        if(min_prob > proba_vector[nbh[i]]) min_prob = proba_vector[nbh[i]];
        if(max_prob < proba_vector[nbh[i]]) max_prob = proba_vector[nbh[i]];
    }
    if(max_prob == min_prob)
        max_prob += 1;


    int segNbr = 5;
    vector<float> sums(segNbr,0);
    vector<vector<int> > seg_nbh(segNbr);
    for(int i=0; i<nbh_size; i++){
        int pos = min(segNbr-1, int((proba_vector[nbh[i]]-min_prob)/(max_prob-min_prob)*segNbr));
        seg_nbh[pos].push_back(i);
        sums[pos] += proba_vector[nbh[i]];
    }

	// create the cumulative vector
	vector<float> cumulative_proba(segNbr);
	cumulative_proba[0] = sums[0];
	for(int i=1; i<segNbr; i++){
		cumulative_proba[i] = sums[i] + cumulative_proba[i-1];
	}
	for(int i=0; i<segNbr; i++){
		cumulative_proba[i] /= cumulative_proba[segNbr-1];
	}


	std::vector<Vector3> accum_points;
	for(int i=0; i<T; i++){
		//get a triplet of points in the neighborhood
		std::vector<int> pt_ids(3);
		do{
			for(int j=0; j<pt_ids.size(); j++){

				// select a point
				float pos = float(rand_ints[randPos])/RAND_MAX;
                randPos = (randPos+1)%rand_ints.size();



                int seg = 0;
                for(int i=0; i<segNbr; i++)
                    if(cumulative_proba[i]>=pos){
                        seg = i;
                        break;
                    }

                //seg = lower_bound(cumulative_proba.begin(), cumulative_proba.end(), pos)-cumulative_proba.begin();


                pt_ids[j] = seg_nbh[seg][rand_ints[randPos]%seg_nbh[seg].size()];

				//pt_ids[j] = rand_ints[randPos]%nbh_size();
				randPos = (randPos+1)%rand_ints.size();
			}
		}while(pt_ids[0]==pt_ids[1] || pt_ids[1]==pt_ids[2] || pt_ids[2]==pt_ids[0]);

		//normal to plane
		Vector3 v1 = _pc.row(nbh[pt_ids[1]])-_pc.row(nbh[pt_ids[0]]);
		Vector3 v2 = _pc.row(nbh[pt_ids[2]])-_pc.row(nbh[pt_ids[0]]);

		Vector3 nl = v1.cross(v2);
		nl = P*nl;
		nl.normalize();
		if(nl.dot(Vector3(0,0,1))<0) nl*=-1; //reorient normal
		accum_points.push_back(nl);
	}

	Matrix3 P2;
	if(compute_P){
		mean *= 0;
		for(int i=0; i<T; i++){
			double c1 = std::max(0.,std::min(1-1e-8,(accum_points[i][0]+1.)/2))*A;
			double c2 = std::max(0.,std::min(1-1e-8,(accum_points[i][1]+1.)/2))*A;
			mean+=Vector3(c1,c2,0);
		}
		mean /= T;
		cov*=0;
		for(int i=0; i<T; i++){
			double c1 = std::max(0.,std::min(1-1e-8,(accum_points[i][0]+1.)/2))*A;
			double c2 = std::max(0.,std::min(1-1e-8,(accum_points[i][1]+1.)/2))*A;
			Vector3 v =  Vector3(c1,c2,0)-mean;
			cov+= v*v.transpose();
		}
		Eigen::JacobiSVD<Matrix3> svd2(cov, Eigen::ComputeFullV);
		P2 =  svd2.matrixV().transpose();
		P = P2*P;
	}

	for(int i=0; i<T; i++){

		Vector3& nl = accum_points[i];
		if(compute_P)
			nl = P2*nl; // change coordinate system
		if(nl.dot(Vector3(0,0,1))<0) nl*=-1; //reorient normal
		nl.normalize();//normalize



		// get the position in accum
		int c1 = std::max(0.,std::min(1-1e-8,(nl[0]+1.)/2.))*A;
		int c2 = std::max(0.,std::min(1-1e-8,(nl[1]+1.)/2.))*A;
		//int pos = c1*A + c2;
		int pos = c1 + c2*A;

		//fill the patch
		accum[pos] ++;
		accum_vec.row(pos) += nl;
	}

	//renorm patch
	accum /= accum.maxCoeff();

}

// not aniso
inline void fill_accum_not_aniso(HoughAccum& hd, std::vector<long int>& nbh, int nbh_size,
		const NormEst* est, unsigned int& randPos,
		bool compute_P = true, Matrix3 P_ref=Matrix3())
{

	//references
	VectorX& accum = hd.accum;
	Matrix3& P = hd.P;
	MatrixX3 &accum_vec = hd.accum_vec;
	int& A = hd.A;

	//init
	A = est->get_A();
	accum_vec = MatrixX3::Zero(A*A,3);
	accum = VectorX::Zero(A*A);
	Vector3 mean = Vector3::Zero();
	Matrix3 cov = Matrix3::Zero();

	//other refs
	const MatrixX3& _pc = est->pc();
	const int T = est->get_T();
	const vector<unsigned int>& rand_ints = est->rand_ints;

	//regression
	for(int i=0; i<nbh_size; i++){
		const Vector3& v =  _pc.row(nbh[i]);
		mean+=v;
	}
	mean /= nbh_size;

	for(int i=0; i<nbh_size; i++){
		Vector3 v =  _pc.row(nbh[i]).transpose()-mean;
		cov+= v*v.transpose();
	}

	Eigen::JacobiSVD<Matrix3> svd(cov, Eigen::ComputeFullV);
	if(compute_P){
		P =  svd.matrixV().transpose();
	}else{
		P = P_ref;
	}
	//Vector3 nor_reg = P.row(2);



	std::vector<Vector3> accum_points;
	for(int i=0; i<T; i++){
		//get a triplet of points in the neighborhood
		std::vector<int> pt_ids(3);
		do{
			for(int j=0; j<pt_ids.size(); j++){

				pt_ids[j] = rand_ints[randPos]%nbh_size;
				randPos = (randPos+1)%rand_ints.size();
			}
		}while(pt_ids[0]==pt_ids[1] || pt_ids[1]==pt_ids[2] || pt_ids[2]==pt_ids[0]);

		//normal to plane
		Vector3 v1 = _pc.row(nbh[pt_ids[1]])-_pc.row(nbh[pt_ids[0]]);
		Vector3 v2 = _pc.row(nbh[pt_ids[2]])-_pc.row(nbh[pt_ids[0]]);

		Vector3 nl = v1.cross(v2);
		nl = P*nl;
		nl.normalize();
		if(nl.dot(Vector3(0,0,1))<0) nl*=-1; //reorient normal
		accum_points.push_back(nl);
	}

	Matrix3 P2;
	if(compute_P){
		mean *= 0;
		for(int i=0; i<T; i++){
			double c1 = std::max(0.,std::min(1-1e-8,(accum_points[i][0]+1.)/2))*A;
			double c2 = std::max(0.,std::min(1-1e-8,(accum_points[i][1]+1.)/2))*A;
			mean+=Vector3(c1,c2,0);
		}
		mean /= T;
		cov*=0;
		for(int i=0; i<T; i++){
			double c1 = std::max(0.,std::min(1-1e-8,(accum_points[i][0]+1.)/2))*A;
			double c2 = std::max(0.,std::min(1-1e-8,(accum_points[i][1]+1.)/2))*A;
			Vector3 v =  Vector3(c1,c2,0)-mean;
			cov+= v*v.transpose();
		}
		Eigen::JacobiSVD<Matrix3> svd2(cov, Eigen::ComputeFullV);
		P2 =  svd2.matrixV().transpose();
		P = P2*P;
	}

	for(int i=0; i<T; i++){

		Vector3& nl = accum_points[i];
		if(compute_P)
			nl = P2*nl; // change coordinate system
		if(nl.dot(Vector3(0,0,1))<0) nl*=-1; //reorient normal
		nl.normalize();//normalize



		// get the position in accum
		int c1 = std::max(0.,std::min(1-1e-8,(nl[0]+1.)/2.))*A;
		int c2 = std::max(0.,std::min(1-1e-8,(nl[1]+1.)/2.))*A;
		//int pos = c1*A + c2;
		int pos = c1 + c2*A;

		//fill the patch
		accum[pos] ++;
		accum_vec.row(pos) += nl;
	}

	//renorm patch
	accum /= accum.maxCoeff();

}

//return the square distance to the farthest point
inline double searchKNN(const kd_tree& tree, const Vector3& pt, int K, vector<long int>& indices, vector<double>& distances){
	indices.resize(K);
	distances.resize(K);
	tree.index->knnSearch(&pt[0], K, &indices[0], &distances[0]);
	double r=0;
	for(int i=0; i<distances.size(); i++)
		if(distances[i]>r) r=distances[i];
	return r;
}


bool compare_pair_int_double(const pair<long int,double>& p1, const pair<long int,double>& p2){
    return p1.second < p2.second;
}

inline void sort_indices_by_distances(vector<long int>& indices, const vector<double>& distances){
    vector< pair<long int,double> > v(distances.size());
    for(int i=0; i<indices.size(); i++){
        v[i].first = indices[i];
        v[i].second = distances[i];
    }
    sort(v.begin(), v.end(), compare_pair_int_double);
    for(int i=0; i<indices.size(); i++){
        indices[i] = v[i].first;
    }
}

void NormEst::initialize(){

    // compute max K
    maxK = 0;
    for(int i=0; i<Ks.size(); i++)
        if(maxK<Ks[i])
            maxK = Ks[i];

    // compute the randints
	int nbr = 1e7;
	rand_ints.resize(nbr);
	for(int i=0; i<nbr; i++){
		rand_ints[i]  = rand();
	}

	// resize the normals
	int N = _pc.rows();
	_normals.resize(N,3);

	// create the tree
    if(is_tree_initialized){
        delete tree;
    }
	tree = new kd_tree(3, _pc, 10 /* max leaf */ );
	tree->index->buildIndex();

    // estimate the probas
    if(use_aniso){
    	proba_vector.resize(N);
        #pragma omp parallel for
    	for(int pt_id=0; pt_id<N; pt_id++){

    		const Vector3& pt = _pc.row(pt_id); //reference to the current point

    		//get the neighborhood
    		vector<long int> indices;
    		vector<double> distances;
    		searchKNN(*tree,pt,K_aniso, indices, distances);

    		float md = 0;
    		for(int i=0; i<distances.size(); i++)
    			if(md<distances[i])
    				md = distances[i];

    		proba_vector[pt_id] = md;
    	}
    }

    randPos = 0;
}

void NormEst::get_batch(int batch_id, int batch_size, double* array) { // array batch_size, Ks.size, A, A

    accums.resize(batch_size);

    // create forward tensor
    unsigned int randPos2 = randPos;
    //#pragma omp parallel for firstprivate(randPos2)
    for(int pt_id=batch_id; pt_id<batch_id+batch_size; pt_id++){
        if(pt_id>=_pc.rows()) continue;

        //reference to the current point
        const Vector3& pt = _pc.row(pt_id);

        //get the max neighborhood
        vector<long int> indices;
        vector<double> distances;
        searchKNN(*tree,pt,maxK, indices, distances);

        // for knn search distances appear to be sorted
        sort_indices_by_distances(indices, distances);

        if(use_aniso){
            for(int k_id=0; k_id<Ks.size(); k_id++){
                //fill the patch and get the rotation matrix
                HoughAccum hd;
                if(k_id==0){
                    fill_accum_aniso(hd,indices,Ks[k_id], this, randPos2, proba_vector);
                    accums[pt_id-batch_id] = hd;
                }else{
                    fill_accum_aniso(hd,indices,Ks[k_id], this, randPos2,  proba_vector, false, accums[pt_id-batch_id].P);
                }

                for(int i=0; i<A*A; i++){
                    array[A*A*Ks.size()*(pt_id-batch_id)+ A*A*k_id +i] = hd.accum[i];
                }
            }
        }else{

            for(int k_id=0; k_id<Ks.size(); k_id++){
                //fill the patch and get the rotation matrix
                HoughAccum hd;
                if(k_id==0){
                    fill_accum_not_aniso(hd,indices,Ks[k_id], this, randPos);
                    accums[pt_id-batch_id] = hd;
                }else{
                    fill_accum_not_aniso(hd,indices,Ks[k_id], this, randPos, false, accums[pt_id-batch_id].P);
                }

                for(int i=0; i<A*A; i++){
                    array[A*A*Ks.size()*(pt_id-batch_id)+ A*A*k_id +i] = hd.accum[i];
                }
            }
        }
    }

}

void NormEst::set_batch(int batch_id, int batch_size, double* array){

    // fill the normal
    #pragma omp parallel for
    for(int pt_id=batch_id; pt_id<batch_id+batch_size; pt_id++){
        if(pt_id>=_pc.rows()) continue;

        // compute final normal
        Vector3 nl(array[2*(pt_id-batch_id)],array[2*(pt_id-batch_id)+1],0);
        double squaredNorm = nl.squaredNorm();
        if(squaredNorm>1){
            nl.normalize();
        }else{
            nl[2] = sqrt(1.-squaredNorm);
        }
        nl = accums[pt_id-batch_id].P.inverse()*nl;
        nl.normalize();

        // store the normals
        _normals.row(pt_id) =nl;
    }
}



void NormEst::get_points(double* array, int m, int n) {

    int i, j ;
    int index = 0 ;

    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            array[index] = _pc(i,j);
            index ++ ;
            }
        }
    return ;
}

void NormEst::get_normals(double* array, int m, int n) {

    int i, j ;
    int index = 0 ;

    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            array[index] = _normals(i,j);
            index ++ ;
            }
        }
    return ;
}

void NormEst::set_points(double* array, int m, int n){
    // resize the point cloud
    _pc.resize(m,3);

    // fill the point cloud
    int i, j ;
    int index = 0 ;
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            _pc(i,j) = array[index];
            index ++ ;
        }
    }
    return ;
}

void NormEst::set_normals(double* array, int m, int n){
    // resize the point cloud
    _normals.resize(m,3);

    // fill the point cloud
    int i, j ;
    int index = 0 ;
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            _normals(i,j) = array[index];
            index ++ ;
        }
    }
    return ;
}


void NormEst::set_Ks(int* array, int m){
	Ks.resize(m);
	for (int i = 0; i < m; i++){
		Ks[i] = array[i];
	}
}
void NormEst::get_Ks(int* array, int m){
    for (int i = 0; i < m; i++){
		array[i] = Ks[i];
	}
}
