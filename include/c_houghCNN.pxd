# Deep Learning for Robust Normal Estimation in Unstructured Point Clouds
# Copyright (c) 2016 Alexande Boulch and Renaud Marlet
#
# This program is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation; either version 3 of the License, or any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this
# program; if not, write to the Free Software Foundation, Inc., 51 Franklin Street,
# Fifth Floor, Boston, MA 02110-1301  USA
#
# PLEASE ACKNOWLEDGE THE ORIGINAL AUTHORS AND PUBLICATION:
# "Deep Learning for Robust Normal Estimation in Unstructured Point Clouds "
# by Alexandre Boulch and Renaud Marlet, Symposium of Geometry Processing 2016,
# Computer Graphics Forum

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
