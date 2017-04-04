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

# distutils: language = c++
# distutils: sources = normEstHough.cxx

cimport c_houghCNN
import numpy as np
cimport numpy as np
import cython

cdef class NormalEstimatorHoughCNN:
    cdef c_houghCNN.NormEst *thisptr

    def __cinit__(self):
        self.thisptr = new c_houghCNN.NormEst()

    def __dealloc__(self):
        del self.thisptr

    cpdef loadXYZ(self, filename):
        self.thisptr.loadXYZ(str.encode(filename))

    cpdef saveXYZ(self,filename):
        self.thisptr.saveXYZ(str.encode(filename))

    cpdef size(self):
        return self.thisptr.size()

    cpdef size_normals(self):
        return self.thisptr.size_normals()

    cpdef get_T(self):
        return self.thisptr.get_T()
    cpdef set_T(self, T):
        self.thisptr.set_T(T)

    cpdef get_A(self):
        return self.thisptr.get_A()
    cpdef set_A(self, A):
        self.thisptr.set_A(A)

    cpdef get_density_sensitive(self):
        return self.thisptr.get_density_sensitive()
    cpdef set_density_sensitive(self, d_s):
        self.thisptr.set_density_sensitive(d_s)

    cpdef get_K_density(self):
        return self.thisptr.get_K_aniso()
    cpdef set_K_density(self, K_d):
        self.thisptr.set_K_aniso(K_d)

    def get_Ks(self):
        cdef m
        m = self.get_Ks_size()
        d = np.zeros(m, dtype = np.int32)
        cdef np.ndarray[np.int32_t, ndim=1] d2 = d
        self.thisptr.get_Ks(<int *> d2.data, m)
        return d

    def get_Ks_size(self):
        return self.thisptr.get_Ks_size()

    def set_Ks(self, Ks):
        cdef np.ndarray[np.int32_t, ndim = 1] d2 = Ks.astype(np.int32)
        self.thisptr.set_Ks(<int *> d2.data, Ks.shape[0])

    def get_points(self):
        cdef int m, n
        m = self.size()
        n = 3
        d = np.zeros((m,n),dtype=np.double)
        cdef np.ndarray[np.float64_t, ndim = 2] d2 = d
        self.thisptr.get_points(<double *> d2.data, m,n)
        return d

    def get_normals(self):
        cdef int m, n
        m = self.size()
        n = 3
        d = np.zeros((m,n),dtype=np.double)
        cdef np.ndarray[np.float64_t, ndim = 2] d2 = d
        self.thisptr.get_normals(<double *> d2.data, m,n)
        return d

    def initialize(self):
        self.thisptr.initialize()

    def get_batch(self, pt_id, batch_size):
        cdef int ptid, bs, ks, A
        ptid = pt_id
        bs=batch_size
        ks= self.get_Ks_size()
        A = self.get_A()
        d = np.zeros((bs,ks,A,A), dtype=np.double)
        cdef np.ndarray[np.float64_t, ndim = 4] d2 = d
        self.thisptr.get_batch(ptid, bs, <double *> d2.data)
        return d

    def set_batch(self, pt_id, batch_size, batch):
        cdef np.ndarray[np.float64_t, ndim = 2] d2 = batch
        self.thisptr.set_batch(pt_id, batch_size, <double *> d2.data)


    def set_points(self, points):
        cdef np.ndarray[np.float64_t, ndim = 2] d2 = points
        self.thisptr.set_points(<double *> d2.data, points.shape[0], points.shape[1])

    def set_normals(self, normals):
        cdef np.ndarray[np.float64_t, ndim = 2] d2 = normals
        self.thisptr.set_normals(<double *> d2.data, normals.shape[0], normals.shape[1])
