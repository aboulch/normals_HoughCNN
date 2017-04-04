from libcpp.string cimport string
import numpy as np
cimport numpy as np

cdef extern from "houghCNN.h":
    cdef cppclass NormEst:

        NormEst()

        void loadXYZ(string)
        void saveXYZ(string)

        void get_points(double*, int,int)
        void set_points(double*, int, int)
        void get_normals(double*, int,int)
        void set_normals(double*, int, int)

        int size()
        int size_normals()

        int get_T()
        void set_T(int)

        int get_A()
        void set_A(int)

        int get_density_sensitive()
        void set_density_sensitive(bool)

        int get_K_aniso()
        void set_K_aniso(int)

        void initialize()
        void get_batch(int,int, double*)
        void set_batch(int,int, double*)

        void get_Ks(int*,int)
        void set_Ks(int*, int)
        int get_Ks_size()
