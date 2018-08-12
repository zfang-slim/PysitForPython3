# import both numpy and the Cython declarations for numpy
import numpy as np
cimport numpy as np

# if you want to use the Numpy-C-API from Cython
# (not strictly necessary for this example)
np.import_array()

ctypedef fused T:
    float
    double

# cdefine the signature of our c function
cdef extern from "constant_density_acoustic_time_scalar_1D_4.h":
     void cda_time_scalar_1D_OMP_4[T]\
                             (T* km1_u,  int nr_km1_u,  int nc_km1_u,      # in - padded wavefield shape
                              T* k_Phiz, int nr_k_Phiz, int nc_k_Phiz,     # in - padded wavefield shape
                              T* k_u,    int nr_k_u,    int nc_k_u,        # in - padded wavefield shape
                              T* C,      int nr_C,      int nc_C,          # in - padded wavefield shape
                              T* rhs,    int nr_rhs,    int nc_rhs,        # in - padded wavefield shape
                              T* zlpml,  int n_zlpml,                      # in - length is the number of nodes inside the padding that the pml value is defined.
                              T* zrpml,  int n_zrpml,                      # in - length is the number of nodes inside the padding that the pml value is defined.
                              double  dt,                            # in
                              double  dz,                            # in
                              int  nz,                               # in
                              T* kp1_Phiz, int nr_kp1_Phiz,  int nc_kp1_Phiz,  # out
                              T* kp1_u,    int nr_kp1_u,     int nc_kp1_u   )  # out

def constant_density_acoustic_time_scalar_1D_4omp(km1_u, k_Phiz, k_u,
                                                  C,     rhs,    zlpml,
                                                  zrpml, dt,     dz,
                                                  nz,    kp1_Phiz, kp1_u):
    if km1_u.dtype == 'double':
        cda_time_scalar_1D_OMP_4(<double*>np.PyArray_DATA(km1_u),  km1_u.size,  1,      # in - padded wavefield shape
                                 <double*>np.PyArray_DATA(k_Phiz), k_Phiz.size, 1,     # in - padded wavefield shape
                                 <double*>np.PyArray_DATA(k_u),    k_u.size,    1,        # in - padded wavefield shape
                                 <double*>np.PyArray_DATA(C),      C.size,      1,          # in - padded wavefield shape
                                 <double*>np.PyArray_DATA(rhs),    rhs.size,    1,        # in - padded wavefield shape
                                 <double*>np.PyArray_DATA(zlpml),  zlpml.size,                      # in - length is the number of nodes inside the padding that the pml value is defined.
                                 <double*>np.PyArray_DATA(zrpml),  zrpml.size,                      # in - length is the number of nodes inside the padding that the pml value is defined.
                                 dt,                            # in
                                 dz,                            # in
                                 nz,                               # in
                                 <double*>np.PyArray_DATA(kp1_Phiz), kp1_Phiz.size,  1,  # out
                                 <double*>np.PyArray_DATA(kp1_u),    kp1_u.size,     1)
    else:
        cda_time_scalar_1D_OMP_4(<float*>np.PyArray_DATA(km1_u),  km1_u.size,  1,      # in - padded wavefield shape
                                 <float*>np.PyArray_DATA(k_Phiz), k_Phiz.size, 1,     # in - padded wavefield shape
                                 <float*>np.PyArray_DATA(k_u),    k_u.size,    1,        # in - padded wavefield shape
                                 <float*>np.PyArray_DATA(C),      C.size,      1,          # in - padded wavefield shape
                                 <float*>np.PyArray_DATA(rhs),    rhs.size,    1,        # in - padded wavefield shape
                                 <float*>np.PyArray_DATA(zlpml),  zlpml.size,                      # in - length is the number of nodes inside the padding that the pml value is defined.
                                 <float*>np.PyArray_DATA(zrpml),  zrpml.size,                      # in - length is the number of nodes inside the padding that the pml value is defined.
                                 dt,                            # in
                                 dz,                            # in
                                 nz,                               # in
                                 <float*>np.PyArray_DATA(kp1_Phiz), kp1_Phiz.size,  1,  # out
                                 <float*>np.PyArray_DATA(kp1_u),    kp1_u.size,     1)
