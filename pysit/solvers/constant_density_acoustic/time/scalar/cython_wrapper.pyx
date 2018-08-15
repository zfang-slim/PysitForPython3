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

# 1D
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


cdef extern from "constant_density_acoustic_time_scalar_1D_6.h":
    void cda_time_scalar_1D_OMP_6[T]\
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


cdef extern from "constant_density_acoustic_time_scalar_1D.h":
    void cda_time_scalar_1D_OS_2[T]\
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


cdef extern from "constant_density_acoustic_time_scalar_1D.h":
    void cda_time_scalar_1D_OS_4[T]\
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


cdef extern from "constant_density_acoustic_time_scalar_1D.h":
    void cda_time_scalar_1D_OS_6[T]\
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


cdef extern from "constant_density_acoustic_time_scalar_1D.h":
    void cda_time_scalar_1D_OS_8[T]\
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


# 2D
cdef extern from "constant_density_acoustic_time_scalar_2D_4.h":
     void cda_time_scalar_2D_OMP_4[T]\
                             (T* km1_u,  int nr_km1_u,  int nc_km1_u,      # in - padded wavefield shape
                              T* k_Phix, int nr_k_Phix, int nc_k_Phix,     # in - padded wavefield shape
                              T* k_Phiz, int nr_k_Phiz, int nc_k_Phiz,     # in - padded wavefield shape
                              T* k_u,    int nr_k_u,    int nc_k_u,        # in - padded wavefield shape
                              T* C,      int nr_C,      int nc_C,          # in - padded wavefield shape
                              T* rhs,    int nr_rhs,    int nc_rhs,        # in - padded wavefield shape
                              T* xlpml,  int n_xlpml,                      # in - length is the number of nodes inside the padding that the pml value is defined.
                              T* xrpml,  int n_xrpml,                      # in - length is the number of nodes inside the padding that the pml value is defined.
                              T* zlpml,  int n_zlpml,                      # in - length is the number of nodes inside the padding that the pml value is defined.
                              T* zrpml,  int n_zrpml,                      # in - length is the number of nodes inside the padding that the pml value is defined.
                              double  dt,                            # in
                              double  dx,                            # in
                              double  dz,                            # in
                              int  nx,                               # in
                              int  nz,                               # in
                              T* kp1_Phix, int nr_kp1_Phix,  int nc_kp1_Phix,  # out
                              T* kp1_Phiz, int nr_kp1_Phiz,  int nc_kp1_Phiz,  # out
                              T* kp1_u,    int nr_kp1_u,     int nc_kp1_u  )  # out


cdef extern from "constant_density_acoustic_time_scalar_2D_6.h":
     void cda_time_scalar_2D_OMP_6[T]\
                             (T* km1_u,  int nr_km1_u,  int nc_km1_u,      # in - padded wavefield shape
                              T* k_Phix, int nr_k_Phix, int nc_k_Phix,     # in - padded wavefield shape
                              T* k_Phiz, int nr_k_Phiz, int nc_k_Phiz,     # in - padded wavefield shape
                              T* k_u,    int nr_k_u,    int nc_k_u,        # in - padded wavefield shape
                              T* C,      int nr_C,      int nc_C,          # in - padded wavefield shape
                              T* rhs,    int nr_rhs,    int nc_rhs,        # in - padded wavefield shape
                              T* xlpml,  int n_xlpml,                      # in - length is the number of nodes inside the padding that the pml value is defined.
                              T* xrpml,  int n_xrpml,                      # in - length is the number of nodes inside the padding that the pml value is defined.
                              T* zlpml,  int n_zlpml,                      # in - length is the number of nodes inside the padding that the pml value is defined.
                              T* zrpml,  int n_zrpml,                      # in - length is the number of nodes inside the padding that the pml value is defined.
                              double  dt,                            # in
                              double  dx,                            # in
                              double  dz,                            # in
                              int  nx,                               # in
                              int  nz,                               # in
                              T* kp1_Phix, int nr_kp1_Phix,  int nc_kp1_Phix,  # out
                              T* kp1_Phiz, int nr_kp1_Phiz,  int nc_kp1_Phiz,  # out
                              T* kp1_u,    int nr_kp1_u,     int nc_kp1_u  )  # out


cdef extern from "constant_density_acoustic_time_scalar_2D.h":
     void cda_time_scalar_2D_OS_2[T]\
                             (T* km1_u,  int nr_km1_u,  int nc_km1_u,      # in - padded wavefield shape
                              T* k_Phix, int nr_k_Phix, int nc_k_Phix,     # in - padded wavefield shape
                              T* k_Phiz, int nr_k_Phiz, int nc_k_Phiz,     # in - padded wavefield shape
                              T* k_u,    int nr_k_u,    int nc_k_u,        # in - padded wavefield shape
                              T* C,      int nr_C,      int nc_C,          # in - padded wavefield shape
                              T* rhs,    int nr_rhs,    int nc_rhs,        # in - padded wavefield shape
                              T* xlpml,  int n_xlpml,                      # in - length is the number of nodes inside the padding that the pml value is defined.
                              T* xrpml,  int n_xrpml,                      # in - length is the number of nodes inside the padding that the pml value is defined.
                              T* zlpml,  int n_zlpml,                      # in - length is the number of nodes inside the padding that the pml value is defined.
                              T* zrpml,  int n_zrpml,                      # in - length is the number of nodes inside the padding that the pml value is defined.
                              double  dt,                            # in
                              double  dx,                            # in
                              double  dz,                            # in
                              int  nx,                               # in
                              int  nz,                               # in
                              T* kp1_Phix, int nr_kp1_Phix,  int nc_kp1_Phix,  # out
                              T* kp1_Phiz, int nr_kp1_Phiz,  int nc_kp1_Phiz,  # out
                              T* kp1_u,    int nr_kp1_u,     int nc_kp1_u  )  # out


cdef extern from "constant_density_acoustic_time_scalar_2D.h":
     void cda_time_scalar_2D_OS_4[T]\
                             (T* km1_u,  int nr_km1_u,  int nc_km1_u,      # in - padded wavefield shape
                              T* k_Phix, int nr_k_Phix, int nc_k_Phix,     # in - padded wavefield shape
                              T* k_Phiz, int nr_k_Phiz, int nc_k_Phiz,     # in - padded wavefield shape
                              T* k_u,    int nr_k_u,    int nc_k_u,        # in - padded wavefield shape
                              T* C,      int nr_C,      int nc_C,          # in - padded wavefield shape
                              T* rhs,    int nr_rhs,    int nc_rhs,        # in - padded wavefield shape
                              T* xlpml,  int n_xlpml,                      # in - length is the number of nodes inside the padding that the pml value is defined.
                              T* xrpml,  int n_xrpml,                      # in - length is the number of nodes inside the padding that the pml value is defined.
                              T* zlpml,  int n_zlpml,                      # in - length is the number of nodes inside the padding that the pml value is defined.
                              T* zrpml,  int n_zrpml,                      # in - length is the number of nodes inside the padding that the pml value is defined.
                              double  dt,                            # in
                              double  dx,                            # in
                              double  dz,                            # in
                              int  nx,                               # in
                              int  nz,                               # in
                              T* kp1_Phix, int nr_kp1_Phix,  int nc_kp1_Phix,  # out
                              T* kp1_Phiz, int nr_kp1_Phiz,  int nc_kp1_Phiz,  # out
                              T* kp1_u,    int nr_kp1_u,     int nc_kp1_u  )  # out


cdef extern from "constant_density_acoustic_time_scalar_2D.h":
     void cda_time_scalar_2D_OS_6[T]\
                             (T* km1_u,  int nr_km1_u,  int nc_km1_u,      # in - padded wavefield shape
                              T* k_Phix, int nr_k_Phix, int nc_k_Phix,     # in - padded wavefield shape
                              T* k_Phiz, int nr_k_Phiz, int nc_k_Phiz,     # in - padded wavefield shape
                              T* k_u,    int nr_k_u,    int nc_k_u,        # in - padded wavefield shape
                              T* C,      int nr_C,      int nc_C,          # in - padded wavefield shape
                              T* rhs,    int nr_rhs,    int nc_rhs,        # in - padded wavefield shape
                              T* xlpml,  int n_xlpml,                      # in - length is the number of nodes inside the padding that the pml value is defined.
                              T* xrpml,  int n_xrpml,                      # in - length is the number of nodes inside the padding that the pml value is defined.
                              T* zlpml,  int n_zlpml,                      # in - length is the number of nodes inside the padding that the pml value is defined.
                              T* zrpml,  int n_zrpml,                      # in - length is the number of nodes inside the padding that the pml value is defined.
                              double  dt,                            # in
                              double  dx,                            # in
                              double  dz,                            # in
                              int  nx,                               # in
                              int  nz,                               # in
                              T* kp1_Phix, int nr_kp1_Phix,  int nc_kp1_Phix,  # out
                              T* kp1_Phiz, int nr_kp1_Phiz,  int nc_kp1_Phiz,  # out
                              T* kp1_u,    int nr_kp1_u,     int nc_kp1_u  )  # out


cdef extern from "constant_density_acoustic_time_scalar_2D.h":
     void cda_time_scalar_2D_OS_8[T]\
                             (T* km1_u,  int nr_km1_u,  int nc_km1_u,      # in - padded wavefield shape
                              T* k_Phix, int nr_k_Phix, int nc_k_Phix,     # in - padded wavefield shape
                              T* k_Phiz, int nr_k_Phiz, int nc_k_Phiz,     # in - padded wavefield shape
                              T* k_u,    int nr_k_u,    int nc_k_u,        # in - padded wavefield shape
                              T* C,      int nr_C,      int nc_C,          # in - padded wavefield shape
                              T* rhs,    int nr_rhs,    int nc_rhs,        # in - padded wavefield shape
                              T* xlpml,  int n_xlpml,                      # in - length is the number of nodes inside the padding that the pml value is defined.
                              T* xrpml,  int n_xrpml,                      # in - length is the number of nodes inside the padding that the pml value is defined.
                              T* zlpml,  int n_zlpml,                      # in - length is the number of nodes inside the padding that the pml value is defined.
                              T* zrpml,  int n_zrpml,                      # in - length is the number of nodes inside the padding that the pml value is defined.
                              double  dt,                            # in
                              double  dx,                            # in
                              double  dz,                            # in
                              int  nx,                               # in
                              int  nz,                               # in
                              T* kp1_Phix, int nr_kp1_Phix,  int nc_kp1_Phix,  # out
                              T* kp1_Phiz, int nr_kp1_Phiz,  int nc_kp1_Phiz,  # out
                              T* kp1_u,    int nr_kp1_u,     int nc_kp1_u  )  # out


# 3D
cdef extern from "constant_density_acoustic_time_scalar_3D_4.h":
     void cda_time_scalar_3D_OMP_4[T]\
                             (T* km1_u,  int nr_km1_u,  int nc_km1_u,      # in - padded wavefield shape
                              T* k_Phix, int nr_k_Phix, int nc_k_Phix,     # in - padded wavefield shape
                              T* k_Phiy, int nr_k_Phiy, int nc_k_Phiy,     # in - padded wavefield shape
                              T* k_Phiz, int nr_k_Phiz, int nc_k_Phiz,     # in - padded wavefield shape
                              T* k_psi,  int nr_k_psi,  int nc_k_psi,      # in - padded wavefield shape
                              T* k_u,    int nr_k_u,    int nc_k_u,        # in - padded wavefield shape
                              T* C,      int nr_C,      int nc_C,          # in - padded wavefield shape
                              T* rhs,    int nr_rhs,    int nc_rhs,        # in - padded wavefield shape
                              T* xlpml,  int n_xlpml,                      # in - length is the number of nodes inside the padding that the pml value is defined.
                              T* xrpml,  int n_xrpml,                      # in - length is the number of nodes inside the padding that the pml value is defined.
                              T* ylpml,  int n_ylpml,                      # in - length is the number of nodes inside the padding that the pml value is defined.
                              T* yrpml,  int n_yrpml,                      # in - length is the number of nodes inside the padding that the pml value is defined.
                              T* zlpml,  int n_zlpml,                      # in - length is the number of nodes inside the padding that the pml value is defined.
                              T* zrpml,  int n_zrpml,                      # in - length is the number of nodes inside the padding that the pml value is defined.
                              double  dt,                            # in
                              double  dx,                            # in
                              double  dy,                            # in
                              double  dz,                            # in
                              int  nx,                               # in
                              int  ny,                               # in
                              int  nz,                               # in
                              T* kp1_Phix, int nr_kp1_Phix,  int nc_kp1_Phix,  # out
                              T* kp1_Phiy, int nr_kp1_Phiy,  int nc_kp1_Phiy,  # out
                              T* kp1_Phiz, int nr_kp1_Phiz,  int nc_kp1_Phiz,  # out
                              T* kp1_psi,  int nr_kp1_psi,   int nc_kp1_psi,   # out
                              T* kp1_u,    int nr_kp1_u,     int nc_kp1_u  )   # out


cdef extern from "constant_density_acoustic_time_scalar_3D_6.h":
     void cda_time_scalar_3D_OMP_6[T]\
                             (T* km1_u,  int nr_km1_u,  int nc_km1_u,      # in - padded wavefield shape
                              T* k_Phix, int nr_k_Phix, int nc_k_Phix,     # in - padded wavefield shape
                              T* k_Phiy, int nr_k_Phiy, int nc_k_Phiy,     # in - padded wavefield shape
                              T* k_Phiz, int nr_k_Phiz, int nc_k_Phiz,     # in - padded wavefield shape
                              T* k_psi,  int nr_k_psi,  int nc_k_psi,      # in - padded wavefield shape
                              T* k_u,    int nr_k_u,    int nc_k_u,        # in - padded wavefield shape
                              T* C,      int nr_C,      int nc_C,          # in - padded wavefield shape
                              T* rhs,    int nr_rhs,    int nc_rhs,        # in - padded wavefield shape
                              T* xlpml,  int n_xlpml,                      # in - length is the number of nodes inside the padding that the pml value is defined.
                              T* xrpml,  int n_xrpml,                      # in - length is the number of nodes inside the padding that the pml value is defined.
                              T* ylpml,  int n_ylpml,                      # in - length is the number of nodes inside the padding that the pml value is defined.
                              T* yrpml,  int n_yrpml,                      # in - length is the number of nodes inside the padding that the pml value is defined.
                              T* zlpml,  int n_zlpml,                      # in - length is the number of nodes inside the padding that the pml value is defined.
                              T* zrpml,  int n_zrpml,                      # in - length is the number of nodes inside the padding that the pml value is defined.
                              double  dt,                            # in
                              double  dx,                            # in
                              double  dy,                            # in
                              double  dz,                            # in
                              int  nx,                               # in
                              int  ny,                               # in
                              int  nz,                               # in
                              T* kp1_Phix, int nr_kp1_Phix,  int nc_kp1_Phix,  # out
                              T* kp1_Phiy, int nr_kp1_Phiy,  int nc_kp1_Phiy,  # out
                              T* kp1_Phiz, int nr_kp1_Phiz,  int nc_kp1_Phiz,  # out
                              T* kp1_psi,  int nr_kp1_psi,   int nc_kp1_psi,   # out
                              T* kp1_u,    int nr_kp1_u,     int nc_kp1_u  )   # out


cdef extern from "constant_density_acoustic_time_scalar_3D.h":
     void cda_time_scalar_3D_OS_2[T]\
                             (T* km1_u,  int nr_km1_u,  int nc_km1_u,      # in - padded wavefield shape
                              T* k_Phix, int nr_k_Phix, int nc_k_Phix,     # in - padded wavefield shape
                              T* k_Phiy, int nr_k_Phiy, int nc_k_Phiy,     # in - padded wavefield shape
                              T* k_Phiz, int nr_k_Phiz, int nc_k_Phiz,     # in - padded wavefield shape
                              T* k_psi,  int nr_k_psi,  int nc_k_psi,      # in - padded wavefield shape
                              T* k_u,    int nr_k_u,    int nc_k_u,        # in - padded wavefield shape
                              T* C,      int nr_C,      int nc_C,          # in - padded wavefield shape
                              T* rhs,    int nr_rhs,    int nc_rhs,        # in - padded wavefield shape
                              T* xlpml,  int n_xlpml,                      # in - length is the number of nodes inside the padding that the pml value is defined.
                              T* xrpml,  int n_xrpml,                      # in - length is the number of nodes inside the padding that the pml value is defined.
                              T* ylpml,  int n_ylpml,                      # in - length is the number of nodes inside the padding that the pml value is defined.
                              T* yrpml,  int n_yrpml,                      # in - length is the number of nodes inside the padding that the pml value is defined.
                              T* zlpml,  int n_zlpml,                      # in - length is the number of nodes inside the padding that the pml value is defined.
                              T* zrpml,  int n_zrpml,                      # in - length is the number of nodes inside the padding that the pml value is defined.
                              double  dt,                            # in
                              double  dx,                            # in
                              double  dy,                            # in
                              double  dz,                            # in
                              int  nx,                               # in
                              int  ny,                               # in
                              int  nz,                               # in
                              T* kp1_Phix, int nr_kp1_Phix,  int nc_kp1_Phix,  # out
                              T* kp1_Phiy, int nr_kp1_Phiy,  int nc_kp1_Phiy,  # out
                              T* kp1_Phiz, int nr_kp1_Phiz,  int nc_kp1_Phiz,  # out
                              T* kp1_psi,  int nr_kp1_psi,   int nc_kp1_psi,   # out
                              T* kp1_u,    int nr_kp1_u,     int nc_kp1_u  )   # out


cdef extern from "constant_density_acoustic_time_scalar_3D.h":
     void cda_time_scalar_3D_OS_4[T]\
                             (T* km1_u,  int nr_km1_u,  int nc_km1_u,      # in - padded wavefield shape
                              T* k_Phix, int nr_k_Phix, int nc_k_Phix,     # in - padded wavefield shape
                              T* k_Phiy, int nr_k_Phiy, int nc_k_Phiy,     # in - padded wavefield shape
                              T* k_Phiz, int nr_k_Phiz, int nc_k_Phiz,     # in - padded wavefield shape
                              T* k_psi,  int nr_k_psi,  int nc_k_psi,      # in - padded wavefield shape
                              T* k_u,    int nr_k_u,    int nc_k_u,        # in - padded wavefield shape
                              T* C,      int nr_C,      int nc_C,          # in - padded wavefield shape
                              T* rhs,    int nr_rhs,    int nc_rhs,        # in - padded wavefield shape
                              T* xlpml,  int n_xlpml,                      # in - length is the number of nodes inside the padding that the pml value is defined.
                              T* xrpml,  int n_xrpml,                      # in - length is the number of nodes inside the padding that the pml value is defined.
                              T* ylpml,  int n_ylpml,                      # in - length is the number of nodes inside the padding that the pml value is defined.
                              T* yrpml,  int n_yrpml,                      # in - length is the number of nodes inside the padding that the pml value is defined.
                              T* zlpml,  int n_zlpml,                      # in - length is the number of nodes inside the padding that the pml value is defined.
                              T* zrpml,  int n_zrpml,                      # in - length is the number of nodes inside the padding that the pml value is defined.
                              double  dt,                            # in
                              double  dx,                            # in
                              double  dy,                            # in
                              double  dz,                            # in
                              int  nx,                               # in
                              int  ny,                               # in
                              int  nz,                               # in
                              T* kp1_Phix, int nr_kp1_Phix,  int nc_kp1_Phix,  # out
                              T* kp1_Phiy, int nr_kp1_Phiy,  int nc_kp1_Phiy,  # out
                              T* kp1_Phiz, int nr_kp1_Phiz,  int nc_kp1_Phiz,  # out
                              T* kp1_psi,  int nr_kp1_psi,   int nc_kp1_psi,   # out
                              T* kp1_u,    int nr_kp1_u,     int nc_kp1_u  )   # out


cdef extern from "constant_density_acoustic_time_scalar_3D.h":
     void cda_time_scalar_3D_OS_6[T]\
                             (T* km1_u,  int nr_km1_u,  int nc_km1_u,      # in - padded wavefield shape
                              T* k_Phix, int nr_k_Phix, int nc_k_Phix,     # in - padded wavefield shape
                              T* k_Phiy, int nr_k_Phiy, int nc_k_Phiy,     # in - padded wavefield shape
                              T* k_Phiz, int nr_k_Phiz, int nc_k_Phiz,     # in - padded wavefield shape
                              T* k_psi,  int nr_k_psi,  int nc_k_psi,      # in - padded wavefield shape
                              T* k_u,    int nr_k_u,    int nc_k_u,        # in - padded wavefield shape
                              T* C,      int nr_C,      int nc_C,          # in - padded wavefield shape
                              T* rhs,    int nr_rhs,    int nc_rhs,        # in - padded wavefield shape
                              T* xlpml,  int n_xlpml,                      # in - length is the number of nodes inside the padding that the pml value is defined.
                              T* xrpml,  int n_xrpml,                      # in - length is the number of nodes inside the padding that the pml value is defined.
                              T* ylpml,  int n_ylpml,                      # in - length is the number of nodes inside the padding that the pml value is defined.
                              T* yrpml,  int n_yrpml,                      # in - length is the number of nodes inside the padding that the pml value is defined.
                              T* zlpml,  int n_zlpml,                      # in - length is the number of nodes inside the padding that the pml value is defined.
                              T* zrpml,  int n_zrpml,                      # in - length is the number of nodes inside the padding that the pml value is defined.
                              double  dt,                            # in
                              double  dx,                            # in
                              double  dy,                            # in
                              double  dz,                            # in
                              int  nx,                               # in
                              int  ny,                               # in
                              int  nz,                               # in
                              T* kp1_Phix, int nr_kp1_Phix,  int nc_kp1_Phix,  # out
                              T* kp1_Phiy, int nr_kp1_Phiy,  int nc_kp1_Phiy,  # out
                              T* kp1_Phiz, int nr_kp1_Phiz,  int nc_kp1_Phiz,  # out
                              T* kp1_psi,  int nr_kp1_psi,   int nc_kp1_psi,   # out
                              T* kp1_u,    int nr_kp1_u,     int nc_kp1_u  )   # out


cdef extern from "constant_density_acoustic_time_scalar_3D.h":
     void cda_time_scalar_3D_OS_8[T]\
                             (T* km1_u,  int nr_km1_u,  int nc_km1_u,      # in - padded wavefield shape
                              T* k_Phix, int nr_k_Phix, int nc_k_Phix,     # in - padded wavefield shape
                              T* k_Phiy, int nr_k_Phiy, int nc_k_Phiy,     # in - padded wavefield shape
                              T* k_Phiz, int nr_k_Phiz, int nc_k_Phiz,     # in - padded wavefield shape
                              T* k_psi,  int nr_k_psi,  int nc_k_psi,      # in - padded wavefield shape
                              T* k_u,    int nr_k_u,    int nc_k_u,        # in - padded wavefield shape
                              T* C,      int nr_C,      int nc_C,          # in - padded wavefield shape
                              T* rhs,    int nr_rhs,    int nc_rhs,        # in - padded wavefield shape
                              T* xlpml,  int n_xlpml,                      # in - length is the number of nodes inside the padding that the pml value is defined.
                              T* xrpml,  int n_xrpml,                      # in - length is the number of nodes inside the padding that the pml value is defined.
                              T* ylpml,  int n_ylpml,                      # in - length is the number of nodes inside the padding that the pml value is defined.
                              T* yrpml,  int n_yrpml,                      # in - length is the number of nodes inside the padding that the pml value is defined.
                              T* zlpml,  int n_zlpml,                      # in - length is the number of nodes inside the padding that the pml value is defined.
                              T* zrpml,  int n_zrpml,                      # in - length is the number of nodes inside the padding that the pml value is defined.
                              double  dt,                            # in
                              double  dx,                            # in
                              double  dy,                            # in
                              double  dz,                            # in
                              int  nx,                               # in
                              int  ny,                               # in
                              int  nz,                               # in
                              T* kp1_Phix, int nr_kp1_Phix,  int nc_kp1_Phix,  # out
                              T* kp1_Phiy, int nr_kp1_Phiy,  int nc_kp1_Phiy,  # out
                              T* kp1_Phiz, int nr_kp1_Phiz,  int nc_kp1_Phiz,  # out
                              T* kp1_psi,  int nr_kp1_psi,   int nc_kp1_psi,   # out
                              T* kp1_u,    int nr_kp1_u,     int nc_kp1_u  )   # out

# 1D
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


def constant_density_acoustic_time_scalar_1D_6omp(km1_u, k_Phiz, k_u,
                                                  C,     rhs,    zlpml,
                                                  zrpml, dt,     dz,
                                                  nz,    kp1_Phiz, kp1_u):
    if km1_u.dtype == 'double':
        cda_time_scalar_1D_OMP_6(<double*>np.PyArray_DATA(km1_u),  km1_u.size,  1,      # in - padded wavefield shape
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
        cda_time_scalar_1D_OMP_6(<float*>np.PyArray_DATA(km1_u),  km1_u.size,  1,      # in - padded wavefield shape
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


def constant_density_acoustic_time_scalar_1D_2os( km1_u, k_Phiz, k_u,
                                                  C,     rhs,    zlpml,
                                                  zrpml, dt,     dz,
                                                  nz,    kp1_Phiz, kp1_u):

    
    if km1_u.dtype == 'double':
        cda_time_scalar_1D_OS_2( <double*>np.PyArray_DATA(km1_u),  km1_u.size,  1,      # in - padded wavefield shape
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
        cda_time_scalar_1D_OS_2( <float*>np.PyArray_DATA(km1_u),  km1_u.size,  1,      # in - padded wavefield shape
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


def constant_density_acoustic_time_scalar_1D_4os( km1_u, k_Phiz, k_u,
                                                  C,     rhs,    zlpml,
                                                  zrpml, dt,     dz,
                                                  nz,    kp1_Phiz, kp1_u):
    if km1_u.dtype == 'double':
        cda_time_scalar_1D_OS_4( <double*>np.PyArray_DATA(km1_u),  km1_u.size,  1,      # in - padded wavefield shape
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
        cda_time_scalar_1D_OS_4( <float*>np.PyArray_DATA(km1_u),  km1_u.size,  1,      # in - padded wavefield shape
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


def constant_density_acoustic_time_scalar_1D_6os( km1_u, k_Phiz, k_u,
                                                  C,     rhs,    zlpml,
                                                  zrpml, dt,     dz,
                                                  nz,    kp1_Phiz, kp1_u):
    if km1_u.dtype == 'double':
        cda_time_scalar_1D_OS_6( <double*>np.PyArray_DATA(km1_u),  km1_u.size,  1,      # in - padded wavefield shape
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
        cda_time_scalar_1D_OS_6( <float*>np.PyArray_DATA(km1_u),  km1_u.size,  1,      # in - padded wavefield shape
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


def constant_density_acoustic_time_scalar_1D_8os( km1_u, k_Phiz, k_u,
                                                  C,     rhs,    zlpml,
                                                  zrpml, dt,     dz,
                                                  nz,    kp1_Phiz, kp1_u):
    if km1_u.dtype == 'double':
        cda_time_scalar_1D_OS_8( <double*>np.PyArray_DATA(km1_u),  km1_u.size,  1,      # in - padded wavefield shape
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
        cda_time_scalar_1D_OS_8( <float*>np.PyArray_DATA(km1_u),  km1_u.size,  1,      # in - padded wavefield shape
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


# 2D
def constant_density_acoustic_time_scalar_2D_4omp(km1_u, k_Phix, k_Phiz, k_u,
                                                  C,     rhs,    xlpml,  xrpml,
                                                  zlpml, zrpml,  dt,     dx,
                                                  dz,    nx,     nz,     kp1_Phix,
                                                  kp1_Phiz, kp1_u):
    if km1_u.dtype == 'double':
        cda_time_scalar_2D_OMP_4(<double*>np.PyArray_DATA(km1_u),  km1_u.size,  1,      # in - padded wavefield shape
                                 <double*>np.PyArray_DATA(k_Phix), k_Phix.size, 1,      # in - padded wavefield shape
                                 <double*>np.PyArray_DATA(k_Phiz), k_Phiz.size, 1,      # in - padded wavefield shape
                                 <double*>np.PyArray_DATA(k_u),    k_u.size,    1,      # in - padded wavefield shape
                                 <double*>np.PyArray_DATA(C),      C.size,      1,      # in - padded wavefield shape
                                 <double*>np.PyArray_DATA(rhs),    rhs.size,    1,      # in - padded wavefield shape
                                 <double*>np.PyArray_DATA(xlpml),  xlpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 <double*>np.PyArray_DATA(xrpml),  xrpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 <double*>np.PyArray_DATA(zlpml),  zlpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 <double*>np.PyArray_DATA(zrpml),  zrpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 dt,                                                    # in
                                 dx,                                                    # in
                                 dz,                                                    # in
                                 nx,                                                    # in
                                 nz,                                                    # in
                                 <double*>np.PyArray_DATA(kp1_Phix), kp1_Phix.size,  1, # out
                                 <double*>np.PyArray_DATA(kp1_Phiz), kp1_Phiz.size,  1, # out
                                 <double*>np.PyArray_DATA(kp1_u),    kp1_u.size,     1) # out
    else:
        cda_time_scalar_2D_OMP_4(<float*>np.PyArray_DATA(km1_u),  km1_u.size,  1,      # in - padded wavefield shape
                                 <float*>np.PyArray_DATA(k_Phix), k_Phix.size, 1,      # in - padded wavefield shape
                                 <float*>np.PyArray_DATA(k_Phiz), k_Phiz.size, 1,      # in - padded wavefield shape
                                 <float*>np.PyArray_DATA(k_u),    k_u.size,    1,      # in - padded wavefield shape
                                 <float*>np.PyArray_DATA(C),      C.size,      1,      # in - padded wavefield shape
                                 <float*>np.PyArray_DATA(rhs),    rhs.size,    1,      # in - padded wavefield shape
                                 <float*>np.PyArray_DATA(xlpml),  xlpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 <float*>np.PyArray_DATA(xrpml),  xrpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 <float*>np.PyArray_DATA(zlpml),  zlpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 <float*>np.PyArray_DATA(zrpml),  zrpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 dt,                                                    # in
                                 dx,                                                    # in
                                 dz,                                                    # in
                                 nx,                                                    # in
                                 nz,                                                    # in
                                 <float*>np.PyArray_DATA(kp1_Phix), kp1_Phix.size,  1, # out
                                 <float*>np.PyArray_DATA(kp1_Phiz), kp1_Phiz.size,  1, # out
                                 <float*>np.PyArray_DATA(kp1_u),    kp1_u.size,     1) # out


def constant_density_acoustic_time_scalar_2D_6omp(km1_u, k_Phix, k_Phiz, k_u,
                                                  C,     rhs,    xlpml,  xrpml,
                                                  zlpml, zrpml,  dt,     dx,
                                                  dz,    nx,     nz,     kp1_Phix,
                                                  kp1_Phiz, kp1_u):
    if km1_u.dtype == 'double':
        cda_time_scalar_2D_OMP_6(<double*>np.PyArray_DATA(km1_u),  km1_u.size,  1,      # in - padded wavefield shape
                                 <double*>np.PyArray_DATA(k_Phix), k_Phix.size, 1,      # in - padded wavefield shape
                                 <double*>np.PyArray_DATA(k_Phiz), k_Phiz.size, 1,      # in - padded wavefield shape
                                 <double*>np.PyArray_DATA(k_u),    k_u.size,    1,      # in - padded wavefield shape
                                 <double*>np.PyArray_DATA(C),      C.size,      1,      # in - padded wavefield shape
                                 <double*>np.PyArray_DATA(rhs),    rhs.size,    1,      # in - padded wavefield shape
                                 <double*>np.PyArray_DATA(xlpml),  xlpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 <double*>np.PyArray_DATA(xrpml),  xrpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 <double*>np.PyArray_DATA(zlpml),  zlpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 <double*>np.PyArray_DATA(zrpml),  zrpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 dt,                                                    # in
                                 dx,                                                    # in
                                 dz,                                                    # in
                                 nx,                                                    # in
                                 nz,                                                    # in
                                 <double*>np.PyArray_DATA(kp1_Phix), kp1_Phix.size,  1, # out
                                 <double*>np.PyArray_DATA(kp1_Phiz), kp1_Phiz.size,  1, # out
                                 <double*>np.PyArray_DATA(kp1_u),    kp1_u.size,     1) # out
    else:
        cda_time_scalar_2D_OMP_6(<float*>np.PyArray_DATA(km1_u),  km1_u.size,  1,      # in - padded wavefield shape
                                 <float*>np.PyArray_DATA(k_Phix), k_Phix.size, 1,      # in - padded wavefield shape
                                 <float*>np.PyArray_DATA(k_Phiz), k_Phiz.size, 1,      # in - padded wavefield shape
                                 <float*>np.PyArray_DATA(k_u),    k_u.size,    1,      # in - padded wavefield shape
                                 <float*>np.PyArray_DATA(C),      C.size,      1,      # in - padded wavefield shape
                                 <float*>np.PyArray_DATA(rhs),    rhs.size,    1,      # in - padded wavefield shape
                                 <float*>np.PyArray_DATA(xlpml),  xlpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 <float*>np.PyArray_DATA(xrpml),  xrpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 <float*>np.PyArray_DATA(zlpml),  zlpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 <float*>np.PyArray_DATA(zrpml),  zrpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 dt,                                                    # in
                                 dx,                                                    # in
                                 dz,                                                    # in
                                 nx,                                                    # in
                                 nz,                                                    # in
                                 <float*>np.PyArray_DATA(kp1_Phix), kp1_Phix.size,  1, # out
                                 <float*>np.PyArray_DATA(kp1_Phiz), kp1_Phiz.size,  1, # out
                                 <float*>np.PyArray_DATA(kp1_u),    kp1_u.size,     1) # out


def constant_density_acoustic_time_scalar_2D_2os(km1_u, k_Phix, k_Phiz, k_u,
                                                  C,     rhs,    xlpml,  xrpml,
                                                  zlpml, zrpml,  dt,     dx,
                                                  dz,    nx,     nz,     kp1_Phix,
                                                  kp1_Phiz, kp1_u):
    if km1_u.dtype == 'double':
        cda_time_scalar_2D_OS_2( <double*>np.PyArray_DATA(km1_u),  km1_u.size,  1,      # in - padded wavefield shape
                                 <double*>np.PyArray_DATA(k_Phix), k_Phix.size, 1,      # in - padded wavefield shape
                                 <double*>np.PyArray_DATA(k_Phiz), k_Phiz.size, 1,      # in - padded wavefield shape
                                 <double*>np.PyArray_DATA(k_u),    k_u.size,    1,      # in - padded wavefield shape
                                 <double*>np.PyArray_DATA(C),      C.size,      1,      # in - padded wavefield shape
                                 <double*>np.PyArray_DATA(rhs),    rhs.size,    1,      # in - padded wavefield shape
                                 <double*>np.PyArray_DATA(xlpml),  xlpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 <double*>np.PyArray_DATA(xrpml),  xrpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 <double*>np.PyArray_DATA(zlpml),  zlpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 <double*>np.PyArray_DATA(zrpml),  zrpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 dt,                                                    # in
                                 dx,                                                    # in
                                 dz,                                                    # in
                                 nx,                                                    # in
                                 nz,                                                    # in
                                 <double*>np.PyArray_DATA(kp1_Phix), kp1_Phix.size,  1, # out
                                 <double*>np.PyArray_DATA(kp1_Phiz), kp1_Phiz.size,  1, # out
                                 <double*>np.PyArray_DATA(kp1_u),    kp1_u.size,     1) # out
    else:
        cda_time_scalar_2D_OS_2( <float*>np.PyArray_DATA(km1_u),  km1_u.size,  1,      # in - padded wavefield shape
                                 <float*>np.PyArray_DATA(k_Phix), k_Phix.size, 1,      # in - padded wavefield shape
                                 <float*>np.PyArray_DATA(k_Phiz), k_Phiz.size, 1,      # in - padded wavefield shape
                                 <float*>np.PyArray_DATA(k_u),    k_u.size,    1,      # in - padded wavefield shape
                                 <float*>np.PyArray_DATA(C),      C.size,      1,      # in - padded wavefield shape
                                 <float*>np.PyArray_DATA(rhs),    rhs.size,    1,      # in - padded wavefield shape
                                 <float*>np.PyArray_DATA(xlpml),  xlpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 <float*>np.PyArray_DATA(xrpml),  xrpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 <float*>np.PyArray_DATA(zlpml),  zlpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 <float*>np.PyArray_DATA(zrpml),  zrpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 dt,                                                    # in
                                 dx,                                                    # in
                                 dz,                                                    # in
                                 nx,                                                    # in
                                 nz,                                                    # in
                                 <float*>np.PyArray_DATA(kp1_Phix), kp1_Phix.size,  1, # out
                                 <float*>np.PyArray_DATA(kp1_Phiz), kp1_Phiz.size,  1, # out
                                 <float*>np.PyArray_DATA(kp1_u),    kp1_u.size,     1) # out


def constant_density_acoustic_time_scalar_2D_4os(km1_u, k_Phix, k_Phiz, k_u,
                                                  C,     rhs,    xlpml,  xrpml,
                                                  zlpml, zrpml,  dt,     dx,
                                                  dz,    nx,     nz,     kp1_Phix,
                                                  kp1_Phiz, kp1_u):
    if km1_u.dtype == 'double':
        cda_time_scalar_2D_OS_4( <double*>np.PyArray_DATA(km1_u),  km1_u.size,  1,      # in - padded wavefield shape
                                 <double*>np.PyArray_DATA(k_Phix), k_Phix.size, 1,      # in - padded wavefield shape
                                 <double*>np.PyArray_DATA(k_Phiz), k_Phiz.size, 1,      # in - padded wavefield shape
                                 <double*>np.PyArray_DATA(k_u),    k_u.size,    1,      # in - padded wavefield shape
                                 <double*>np.PyArray_DATA(C),      C.size,      1,      # in - padded wavefield shape
                                 <double*>np.PyArray_DATA(rhs),    rhs.size,    1,      # in - padded wavefield shape
                                 <double*>np.PyArray_DATA(xlpml),  xlpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 <double*>np.PyArray_DATA(xrpml),  xrpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 <double*>np.PyArray_DATA(zlpml),  zlpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 <double*>np.PyArray_DATA(zrpml),  zrpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 dt,                                                    # in
                                 dx,                                                    # in
                                 dz,                                                    # in
                                 nx,                                                    # in
                                 nz,                                                    # in
                                 <double*>np.PyArray_DATA(kp1_Phix), kp1_Phix.size,  1, # out
                                 <double*>np.PyArray_DATA(kp1_Phiz), kp1_Phiz.size,  1, # out
                                 <double*>np.PyArray_DATA(kp1_u),    kp1_u.size,     1) # out
    else:
        cda_time_scalar_2D_OS_4( <float*>np.PyArray_DATA(km1_u),  km1_u.size,  1,      # in - padded wavefield shape
                                 <float*>np.PyArray_DATA(k_Phix), k_Phix.size, 1,      # in - padded wavefield shape
                                 <float*>np.PyArray_DATA(k_Phiz), k_Phiz.size, 1,      # in - padded wavefield shape
                                 <float*>np.PyArray_DATA(k_u),    k_u.size,    1,      # in - padded wavefield shape
                                 <float*>np.PyArray_DATA(C),      C.size,      1,      # in - padded wavefield shape
                                 <float*>np.PyArray_DATA(rhs),    rhs.size,    1,      # in - padded wavefield shape
                                 <float*>np.PyArray_DATA(xlpml),  xlpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 <float*>np.PyArray_DATA(xrpml),  xrpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 <float*>np.PyArray_DATA(zlpml),  zlpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 <float*>np.PyArray_DATA(zrpml),  zrpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 dt,                                                    # in
                                 dx,                                                    # in
                                 dz,                                                    # in
                                 nx,                                                    # in
                                 nz,                                                    # in
                                 <float*>np.PyArray_DATA(kp1_Phix), kp1_Phix.size,  1, # out
                                 <float*>np.PyArray_DATA(kp1_Phiz), kp1_Phiz.size,  1, # out
                                 <float*>np.PyArray_DATA(kp1_u),    kp1_u.size,     1) # out


def constant_density_acoustic_time_scalar_2D_6os(km1_u, k_Phix, k_Phiz, k_u,
                                                  C,     rhs,    xlpml,  xrpml,
                                                  zlpml, zrpml,  dt,     dx,
                                                  dz,    nx,     nz,     kp1_Phix,
                                                  kp1_Phiz, kp1_u):
    if km1_u.dtype == 'double':
        cda_time_scalar_2D_OS_6( <double*>np.PyArray_DATA(km1_u),  km1_u.size,  1,      # in - padded wavefield shape
                                 <double*>np.PyArray_DATA(k_Phix), k_Phix.size, 1,      # in - padded wavefield shape
                                 <double*>np.PyArray_DATA(k_Phiz), k_Phiz.size, 1,      # in - padded wavefield shape
                                 <double*>np.PyArray_DATA(k_u),    k_u.size,    1,      # in - padded wavefield shape
                                 <double*>np.PyArray_DATA(C),      C.size,      1,      # in - padded wavefield shape
                                 <double*>np.PyArray_DATA(rhs),    rhs.size,    1,      # in - padded wavefield shape
                                 <double*>np.PyArray_DATA(xlpml),  xlpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 <double*>np.PyArray_DATA(xrpml),  xrpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 <double*>np.PyArray_DATA(zlpml),  zlpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 <double*>np.PyArray_DATA(zrpml),  zrpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 dt,                                                    # in
                                 dx,                                                    # in
                                 dz,                                                    # in
                                 nx,                                                    # in
                                 nz,                                                    # in
                                 <double*>np.PyArray_DATA(kp1_Phix), kp1_Phix.size,  1, # out
                                 <double*>np.PyArray_DATA(kp1_Phiz), kp1_Phiz.size,  1, # out
                                 <double*>np.PyArray_DATA(kp1_u),    kp1_u.size,     1) # out
    else:
        cda_time_scalar_2D_OS_6( <float*>np.PyArray_DATA(km1_u),  km1_u.size,  1,      # in - padded wavefield shape
                                 <float*>np.PyArray_DATA(k_Phix), k_Phix.size, 1,      # in - padded wavefield shape
                                 <float*>np.PyArray_DATA(k_Phiz), k_Phiz.size, 1,      # in - padded wavefield shape
                                 <float*>np.PyArray_DATA(k_u),    k_u.size,    1,      # in - padded wavefield shape
                                 <float*>np.PyArray_DATA(C),      C.size,      1,      # in - padded wavefield shape
                                 <float*>np.PyArray_DATA(rhs),    rhs.size,    1,      # in - padded wavefield shape
                                 <float*>np.PyArray_DATA(xlpml),  xlpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 <float*>np.PyArray_DATA(xrpml),  xrpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 <float*>np.PyArray_DATA(zlpml),  zlpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 <float*>np.PyArray_DATA(zrpml),  zrpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 dt,                                                    # in
                                 dx,                                                    # in
                                 dz,                                                    # in
                                 nx,                                                    # in
                                 nz,                                                    # in
                                 <float*>np.PyArray_DATA(kp1_Phix), kp1_Phix.size,  1, # out
                                 <float*>np.PyArray_DATA(kp1_Phiz), kp1_Phiz.size,  1, # out
                                 <float*>np.PyArray_DATA(kp1_u),    kp1_u.size,     1) # out


def constant_density_acoustic_time_scalar_2D_8os(km1_u, k_Phix, k_Phiz, k_u,
                                                  C,     rhs,    xlpml,  xrpml,
                                                  zlpml, zrpml,  dt,     dx,
                                                  dz,    nx,     nz,     kp1_Phix,
                                                  kp1_Phiz, kp1_u):
    if km1_u.dtype == 'double':
        cda_time_scalar_2D_OS_8( <double*>np.PyArray_DATA(km1_u),  km1_u.size,  1,      # in - padded wavefield shape
                                 <double*>np.PyArray_DATA(k_Phix), k_Phix.size, 1,      # in - padded wavefield shape
                                 <double*>np.PyArray_DATA(k_Phiz), k_Phiz.size, 1,      # in - padded wavefield shape
                                 <double*>np.PyArray_DATA(k_u),    k_u.size,    1,      # in - padded wavefield shape
                                 <double*>np.PyArray_DATA(C),      C.size,      1,      # in - padded wavefield shape
                                 <double*>np.PyArray_DATA(rhs),    rhs.size,    1,      # in - padded wavefield shape
                                 <double*>np.PyArray_DATA(xlpml),  xlpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 <double*>np.PyArray_DATA(xrpml),  xrpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 <double*>np.PyArray_DATA(zlpml),  zlpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 <double*>np.PyArray_DATA(zrpml),  zrpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 dt,                                                    # in
                                 dx,                                                    # in
                                 dz,                                                    # in
                                 nx,                                                    # in
                                 nz,                                                    # in
                                 <double*>np.PyArray_DATA(kp1_Phix), kp1_Phix.size,  1, # out
                                 <double*>np.PyArray_DATA(kp1_Phiz), kp1_Phiz.size,  1, # out
                                 <double*>np.PyArray_DATA(kp1_u),    kp1_u.size,     1) # out
    else:
        cda_time_scalar_2D_OS_8( <float*>np.PyArray_DATA(km1_u),  km1_u.size,  1,      # in - padded wavefield shape
                                 <float*>np.PyArray_DATA(k_Phix), k_Phix.size, 1,      # in - padded wavefield shape
                                 <float*>np.PyArray_DATA(k_Phiz), k_Phiz.size, 1,      # in - padded wavefield shape
                                 <float*>np.PyArray_DATA(k_u),    k_u.size,    1,      # in - padded wavefield shape
                                 <float*>np.PyArray_DATA(C),      C.size,      1,      # in - padded wavefield shape
                                 <float*>np.PyArray_DATA(rhs),    rhs.size,    1,      # in - padded wavefield shape
                                 <float*>np.PyArray_DATA(xlpml),  xlpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 <float*>np.PyArray_DATA(xrpml),  xrpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 <float*>np.PyArray_DATA(zlpml),  zlpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 <float*>np.PyArray_DATA(zrpml),  zrpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 dt,                                                    # in
                                 dx,                                                    # in
                                 dz,                                                    # in
                                 nx,                                                    # in
                                 nz,                                                    # in
                                 <float*>np.PyArray_DATA(kp1_Phix), kp1_Phix.size,  1, # out
                                 <float*>np.PyArray_DATA(kp1_Phiz), kp1_Phiz.size,  1, # out
                                 <float*>np.PyArray_DATA(kp1_u),    kp1_u.size,     1) # out


# 3D
def constant_density_acoustic_time_scalar_3D_4omp(km1_u,   k_Phix,   k_Phiy,   k_Phiz,
                                                  k_psi,   k_u,      C,        rhs,
                                                  xlpml,   xrpml,    ylpml,    yrpml,
                                                  zlpml,   zrpml,    dt,       dx,
                                                  dy,      dz,       nx,       ny,
                                                  nz,      kp1_Phix, kp1_Phiy, kp1_Phiz,
                                                  kp1_psi, kp1_u):
    if km1_u.dtype == 'double':
        cda_time_scalar_3D_OMP_4(<double*>np.PyArray_DATA(km1_u),  km1_u.size,  1,      # in - padded wavefield shape
                                 <double*>np.PyArray_DATA(k_Phix), k_Phix.size, 1,      # in - padded wavefield shape
                                 <double*>np.PyArray_DATA(k_Phiy), k_Phiy.size, 1,      # in - padded wavefield shape
                                 <double*>np.PyArray_DATA(k_Phiz), k_Phiz.size, 1,      # in - padded wavefield shape
                                 <double*>np.PyArray_DATA(k_psi),  k_psi.size,  1,      # in - padded wavefield shape
                                 <double*>np.PyArray_DATA(k_u),    k_u.size,    1,      # in - padded wavefield shape
                                 <double*>np.PyArray_DATA(C),      C.size,      1,      # in - padded wavefield shape
                                 <double*>np.PyArray_DATA(rhs),    rhs.size,    1,      # in - padded wavefield shape
                                 <double*>np.PyArray_DATA(xlpml),  xlpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 <double*>np.PyArray_DATA(xrpml),  xrpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 <double*>np.PyArray_DATA(ylpml),  ylpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 <double*>np.PyArray_DATA(yrpml),  yrpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 <double*>np.PyArray_DATA(zlpml),  zlpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 <double*>np.PyArray_DATA(zrpml),  zrpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 dt,                                                    # in
                                 dx,                                                    # in
                                 dy,                                                    # in
                                 dz,                                                    # in
                                 nx,                                                    # in
                                 ny,                                                    # in
                                 nz,                                                    # in
                                 <double*>np.PyArray_DATA(kp1_Phix), kp1_Phix.size,  1, # out
                                 <double*>np.PyArray_DATA(kp1_Phiy), kp1_Phiy.size,  1, # out
                                 <double*>np.PyArray_DATA(kp1_Phiz), kp1_Phiz.size,  1, # out
                                 <double*>np.PyArray_DATA(kp1_psi),  kp1_psi.size,   1, # out
                                 <double*>np.PyArray_DATA(kp1_u),    kp1_u.size,     1) # out
    else:
        cda_time_scalar_3D_OMP_4(<float*>np.PyArray_DATA(km1_u),  km1_u.size,  1,      # in - padded wavefield shape
                                 <float*>np.PyArray_DATA(k_Phix), k_Phix.size, 1,      # in - padded wavefield shape
                                 <float*>np.PyArray_DATA(k_Phiy), k_Phiy.size, 1,      # in - padded wavefield shape
                                 <float*>np.PyArray_DATA(k_Phiz), k_Phiz.size, 1,      # in - padded wavefield shape
                                 <float*>np.PyArray_DATA(k_psi),  k_psi.size,  1,      # in - padded wavefield shape
                                 <float*>np.PyArray_DATA(k_u),    k_u.size,    1,      # in - padded wavefield shape
                                 <float*>np.PyArray_DATA(C),      C.size,      1,      # in - padded wavefield shape
                                 <float*>np.PyArray_DATA(rhs),    rhs.size,    1,      # in - padded wavefield shape
                                 <float*>np.PyArray_DATA(xlpml),  xlpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 <float*>np.PyArray_DATA(xrpml),  xrpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 <float*>np.PyArray_DATA(ylpml),  ylpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 <float*>np.PyArray_DATA(yrpml),  yrpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 <float*>np.PyArray_DATA(zlpml),  zlpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 <float*>np.PyArray_DATA(zrpml),  zrpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 dt,                                                    # in
                                 dx,                                                    # in
                                 dy,                                                    # in
                                 dz,                                                    # in
                                 nx,                                                    # in
                                 ny,                                                    # in
                                 nz,                                                    # in
                                 <float*>np.PyArray_DATA(kp1_Phix), kp1_Phix.size,  1, # out
                                 <float*>np.PyArray_DATA(kp1_Phiy), kp1_Phiy.size,  1, # out
                                 <float*>np.PyArray_DATA(kp1_Phiz), kp1_Phiz.size,  1, # out
                                 <float*>np.PyArray_DATA(kp1_psi),  kp1_psi.size,   1, # out
                                 <float*>np.PyArray_DATA(kp1_u),    kp1_u.size,     1) # out


def constant_density_acoustic_time_scalar_3D_6omp(km1_u,   k_Phix,   k_Phiy,   k_Phiz,
                                                  k_psi,   k_u,      C,        rhs,
                                                  xlpml,   xrpml,    ylpml,    yrpml,
                                                  zlpml,   zrpml,    dt,       dx,
                                                  dy,      dz,       nx,       ny,
                                                  nz,      kp1_Phix, kp1_Phiy, kp1_Phiz,
                                                  kp1_psi, kp1_u):
    if km1_u.dtype == 'double':
        cda_time_scalar_3D_OMP_6(<double*>np.PyArray_DATA(km1_u),  km1_u.size,  1,      # in - padded wavefield shape
                                 <double*>np.PyArray_DATA(k_Phix), k_Phix.size, 1,      # in - padded wavefield shape
                                 <double*>np.PyArray_DATA(k_Phiy), k_Phiy.size, 1,      # in - padded wavefield shape
                                 <double*>np.PyArray_DATA(k_Phiz), k_Phiz.size, 1,      # in - padded wavefield shape
                                 <double*>np.PyArray_DATA(k_psi),  k_psi.size,  1,      # in - padded wavefield shape
                                 <double*>np.PyArray_DATA(k_u),    k_u.size,    1,      # in - padded wavefield shape
                                 <double*>np.PyArray_DATA(C),      C.size,      1,      # in - padded wavefield shape
                                 <double*>np.PyArray_DATA(rhs),    rhs.size,    1,      # in - padded wavefield shape
                                 <double*>np.PyArray_DATA(xlpml),  xlpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 <double*>np.PyArray_DATA(xrpml),  xrpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 <double*>np.PyArray_DATA(ylpml),  ylpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 <double*>np.PyArray_DATA(yrpml),  yrpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 <double*>np.PyArray_DATA(zlpml),  zlpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 <double*>np.PyArray_DATA(zrpml),  zrpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 dt,                                                    # in
                                 dx,                                                    # in
                                 dy,                                                    # in
                                 dz,                                                    # in
                                 nx,                                                    # in
                                 ny,                                                    # in
                                 nz,                                                    # in
                                 <double*>np.PyArray_DATA(kp1_Phix), kp1_Phix.size,  1, # out
                                 <double*>np.PyArray_DATA(kp1_Phiy), kp1_Phiy.size,  1, # out
                                 <double*>np.PyArray_DATA(kp1_Phiz), kp1_Phiz.size,  1, # out
                                 <double*>np.PyArray_DATA(kp1_psi),  kp1_psi.size,   1, # out
                                 <double*>np.PyArray_DATA(kp1_u),    kp1_u.size,     1) # out
    else:
        cda_time_scalar_3D_OMP_6(<float*>np.PyArray_DATA(km1_u),  km1_u.size,  1,      # in - padded wavefield shape
                                 <float*>np.PyArray_DATA(k_Phix), k_Phix.size, 1,      # in - padded wavefield shape
                                 <float*>np.PyArray_DATA(k_Phiy), k_Phiy.size, 1,      # in - padded wavefield shape
                                 <float*>np.PyArray_DATA(k_Phiz), k_Phiz.size, 1,      # in - padded wavefield shape
                                 <float*>np.PyArray_DATA(k_psi),  k_psi.size,  1,      # in - padded wavefield shape
                                 <float*>np.PyArray_DATA(k_u),    k_u.size,    1,      # in - padded wavefield shape
                                 <float*>np.PyArray_DATA(C),      C.size,      1,      # in - padded wavefield shape
                                 <float*>np.PyArray_DATA(rhs),    rhs.size,    1,      # in - padded wavefield shape
                                 <float*>np.PyArray_DATA(xlpml),  xlpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 <float*>np.PyArray_DATA(xrpml),  xrpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 <float*>np.PyArray_DATA(ylpml),  ylpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 <float*>np.PyArray_DATA(yrpml),  yrpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 <float*>np.PyArray_DATA(zlpml),  zlpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 <float*>np.PyArray_DATA(zrpml),  zrpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 dt,                                                    # in
                                 dx,                                                    # in
                                 dy,                                                    # in
                                 dz,                                                    # in
                                 nx,                                                    # in
                                 ny,                                                    # in
                                 nz,                                                    # in
                                 <float*>np.PyArray_DATA(kp1_Phix), kp1_Phix.size,  1, # out
                                 <float*>np.PyArray_DATA(kp1_Phiy), kp1_Phiy.size,  1, # out
                                 <float*>np.PyArray_DATA(kp1_Phiz), kp1_Phiz.size,  1, # out
                                 <float*>np.PyArray_DATA(kp1_psi),  kp1_psi.size,   1, # out
                                 <float*>np.PyArray_DATA(kp1_u),    kp1_u.size,     1) # out


def constant_density_acoustic_time_scalar_3D_2os( km1_u,   k_Phix,   k_Phiy,   k_Phiz,
                                                  k_psi,   k_u,      C,        rhs,
                                                  xlpml,   xrpml,    ylpml,    yrpml,
                                                  zlpml,   zrpml,    dt,       dx,
                                                  dy,      dz,       nx,       ny,
                                                  nz,      kp1_Phix, kp1_Phiy, kp1_Phiz,
                                                  kp1_psi, kp1_u):
    if km1_u.dtype == 'double':
        cda_time_scalar_3D_OS_2( <double*>np.PyArray_DATA(km1_u),  km1_u.size,  1,      # in - padded wavefield shape
                                 <double*>np.PyArray_DATA(k_Phix), k_Phix.size, 1,      # in - padded wavefield shape
                                 <double*>np.PyArray_DATA(k_Phiy), k_Phiy.size, 1,      # in - padded wavefield shape
                                 <double*>np.PyArray_DATA(k_Phiz), k_Phiz.size, 1,      # in - padded wavefield shape
                                 <double*>np.PyArray_DATA(k_psi),  k_psi.size,  1,      # in - padded wavefield shape
                                 <double*>np.PyArray_DATA(k_u),    k_u.size,    1,      # in - padded wavefield shape
                                 <double*>np.PyArray_DATA(C),      C.size,      1,      # in - padded wavefield shape
                                 <double*>np.PyArray_DATA(rhs),    rhs.size,    1,      # in - padded wavefield shape
                                 <double*>np.PyArray_DATA(xlpml),  xlpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 <double*>np.PyArray_DATA(xrpml),  xrpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 <double*>np.PyArray_DATA(ylpml),  ylpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 <double*>np.PyArray_DATA(yrpml),  yrpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 <double*>np.PyArray_DATA(zlpml),  zlpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 <double*>np.PyArray_DATA(zrpml),  zrpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 dt,                                                    # in
                                 dx,                                                    # in
                                 dy,                                                    # in
                                 dz,                                                    # in
                                 nx,                                                    # in
                                 ny,                                                    # in
                                 nz,                                                    # in
                                 <double*>np.PyArray_DATA(kp1_Phix), kp1_Phix.size,  1, # out
                                 <double*>np.PyArray_DATA(kp1_Phiy), kp1_Phiy.size,  1, # out
                                 <double*>np.PyArray_DATA(kp1_Phiz), kp1_Phiz.size,  1, # out
                                 <double*>np.PyArray_DATA(kp1_psi),  kp1_psi.size,   1, # out
                                 <double*>np.PyArray_DATA(kp1_u),    kp1_u.size,     1) # out
    else:
        cda_time_scalar_3D_OS_2( <float*>np.PyArray_DATA(km1_u),  km1_u.size,  1,      # in - padded wavefield shape
                                 <float*>np.PyArray_DATA(k_Phix), k_Phix.size, 1,      # in - padded wavefield shape
                                 <float*>np.PyArray_DATA(k_Phiy), k_Phiy.size, 1,      # in - padded wavefield shape
                                 <float*>np.PyArray_DATA(k_Phiz), k_Phiz.size, 1,      # in - padded wavefield shape
                                 <float*>np.PyArray_DATA(k_psi),  k_psi.size,  1,      # in - padded wavefield shape
                                 <float*>np.PyArray_DATA(k_u),    k_u.size,    1,      # in - padded wavefield shape
                                 <float*>np.PyArray_DATA(C),      C.size,      1,      # in - padded wavefield shape
                                 <float*>np.PyArray_DATA(rhs),    rhs.size,    1,      # in - padded wavefield shape
                                 <float*>np.PyArray_DATA(xlpml),  xlpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 <float*>np.PyArray_DATA(xrpml),  xrpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 <float*>np.PyArray_DATA(ylpml),  ylpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 <float*>np.PyArray_DATA(yrpml),  yrpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 <float*>np.PyArray_DATA(zlpml),  zlpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 <float*>np.PyArray_DATA(zrpml),  zrpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 dt,                                                    # in
                                 dx,                                                    # in
                                 dy,                                                    # in
                                 dz,                                                    # in
                                 nx,                                                    # in
                                 ny,                                                    # in
                                 nz,                                                    # in
                                 <float*>np.PyArray_DATA(kp1_Phix), kp1_Phix.size,  1, # out
                                 <float*>np.PyArray_DATA(kp1_Phiy), kp1_Phiy.size,  1, # out
                                 <float*>np.PyArray_DATA(kp1_Phiz), kp1_Phiz.size,  1, # out
                                 <float*>np.PyArray_DATA(kp1_psi),  kp1_psi.size,   1, # out
                                 <float*>np.PyArray_DATA(kp1_u),    kp1_u.size,     1) # out


def constant_density_acoustic_time_scalar_3D_4os( km1_u,   k_Phix,   k_Phiy,   k_Phiz,
                                                  k_psi,   k_u,      C,        rhs,
                                                  xlpml,   xrpml,    ylpml,    yrpml,
                                                  zlpml,   zrpml,    dt,       dx,
                                                  dy,      dz,       nx,       ny,
                                                  nz,      kp1_Phix, kp1_Phiy, kp1_Phiz,
                                                  kp1_psi, kp1_u):
    if km1_u.dtype == 'double':
        cda_time_scalar_3D_OS_4( <double*>np.PyArray_DATA(km1_u),  km1_u.size,  1,      # in - padded wavefield shape
                                 <double*>np.PyArray_DATA(k_Phix), k_Phix.size, 1,      # in - padded wavefield shape
                                 <double*>np.PyArray_DATA(k_Phiy), k_Phiy.size, 1,      # in - padded wavefield shape
                                 <double*>np.PyArray_DATA(k_Phiz), k_Phiz.size, 1,      # in - padded wavefield shape
                                 <double*>np.PyArray_DATA(k_psi),  k_psi.size,  1,      # in - padded wavefield shape
                                 <double*>np.PyArray_DATA(k_u),    k_u.size,    1,      # in - padded wavefield shape
                                 <double*>np.PyArray_DATA(C),      C.size,      1,      # in - padded wavefield shape
                                 <double*>np.PyArray_DATA(rhs),    rhs.size,    1,      # in - padded wavefield shape
                                 <double*>np.PyArray_DATA(xlpml),  xlpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 <double*>np.PyArray_DATA(xrpml),  xrpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 <double*>np.PyArray_DATA(ylpml),  ylpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 <double*>np.PyArray_DATA(yrpml),  yrpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 <double*>np.PyArray_DATA(zlpml),  zlpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 <double*>np.PyArray_DATA(zrpml),  zrpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 dt,                                                    # in
                                 dx,                                                    # in
                                 dy,                                                    # in
                                 dz,                                                    # in
                                 nx,                                                    # in
                                 ny,                                                    # in
                                 nz,                                                    # in
                                 <double*>np.PyArray_DATA(kp1_Phix), kp1_Phix.size,  1, # out
                                 <double*>np.PyArray_DATA(kp1_Phiy), kp1_Phiy.size,  1, # out
                                 <double*>np.PyArray_DATA(kp1_Phiz), kp1_Phiz.size,  1, # out
                                 <double*>np.PyArray_DATA(kp1_psi),  kp1_psi.size,   1, # out
                                 <double*>np.PyArray_DATA(kp1_u),    kp1_u.size,     1) # out
    else:
        cda_time_scalar_3D_OS_4( <float*>np.PyArray_DATA(km1_u),  km1_u.size,  1,      # in - padded wavefield shape
                                 <float*>np.PyArray_DATA(k_Phix), k_Phix.size, 1,      # in - padded wavefield shape
                                 <float*>np.PyArray_DATA(k_Phiy), k_Phiy.size, 1,      # in - padded wavefield shape
                                 <float*>np.PyArray_DATA(k_Phiz), k_Phiz.size, 1,      # in - padded wavefield shape
                                 <float*>np.PyArray_DATA(k_psi),  k_psi.size,  1,      # in - padded wavefield shape
                                 <float*>np.PyArray_DATA(k_u),    k_u.size,    1,      # in - padded wavefield shape
                                 <float*>np.PyArray_DATA(C),      C.size,      1,      # in - padded wavefield shape
                                 <float*>np.PyArray_DATA(rhs),    rhs.size,    1,      # in - padded wavefield shape
                                 <float*>np.PyArray_DATA(xlpml),  xlpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 <float*>np.PyArray_DATA(xrpml),  xrpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 <float*>np.PyArray_DATA(ylpml),  ylpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 <float*>np.PyArray_DATA(yrpml),  yrpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 <float*>np.PyArray_DATA(zlpml),  zlpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 <float*>np.PyArray_DATA(zrpml),  zrpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 dt,                                                    # in
                                 dx,                                                    # in
                                 dy,                                                    # in
                                 dz,                                                    # in
                                 nx,                                                    # in
                                 ny,                                                    # in
                                 nz,                                                    # in
                                 <float*>np.PyArray_DATA(kp1_Phix), kp1_Phix.size,  1, # out
                                 <float*>np.PyArray_DATA(kp1_Phiy), kp1_Phiy.size,  1, # out
                                 <float*>np.PyArray_DATA(kp1_Phiz), kp1_Phiz.size,  1, # out
                                 <float*>np.PyArray_DATA(kp1_psi),  kp1_psi.size,   1, # out
                                 <float*>np.PyArray_DATA(kp1_u),    kp1_u.size,     1) # out


def constant_density_acoustic_time_scalar_3D_6os( km1_u,   k_Phix,   k_Phiy,   k_Phiz,
                                                  k_psi,   k_u,      C,        rhs,
                                                  xlpml,   xrpml,    ylpml,    yrpml,
                                                  zlpml,   zrpml,    dt,       dx,
                                                  dy,      dz,       nx,       ny,
                                                  nz,      kp1_Phix, kp1_Phiy, kp1_Phiz,
                                                  kp1_psi, kp1_u):
    if km1_u.dtype == 'double':
        cda_time_scalar_3D_OS_6( <double*>np.PyArray_DATA(km1_u),  km1_u.size,  1,      # in - padded wavefield shape
                                 <double*>np.PyArray_DATA(k_Phix), k_Phix.size, 1,      # in - padded wavefield shape
                                 <double*>np.PyArray_DATA(k_Phiy), k_Phiy.size, 1,      # in - padded wavefield shape
                                 <double*>np.PyArray_DATA(k_Phiz), k_Phiz.size, 1,      # in - padded wavefield shape
                                 <double*>np.PyArray_DATA(k_psi),  k_psi.size,  1,      # in - padded wavefield shape
                                 <double*>np.PyArray_DATA(k_u),    k_u.size,    1,      # in - padded wavefield shape
                                 <double*>np.PyArray_DATA(C),      C.size,      1,      # in - padded wavefield shape
                                 <double*>np.PyArray_DATA(rhs),    rhs.size,    1,      # in - padded wavefield shape
                                 <double*>np.PyArray_DATA(xlpml),  xlpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 <double*>np.PyArray_DATA(xrpml),  xrpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 <double*>np.PyArray_DATA(ylpml),  ylpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 <double*>np.PyArray_DATA(yrpml),  yrpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 <double*>np.PyArray_DATA(zlpml),  zlpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 <double*>np.PyArray_DATA(zrpml),  zrpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 dt,                                                    # in
                                 dx,                                                    # in
                                 dy,                                                    # in
                                 dz,                                                    # in
                                 nx,                                                    # in
                                 ny,                                                    # in
                                 nz,                                                    # in
                                 <double*>np.PyArray_DATA(kp1_Phix), kp1_Phix.size,  1, # out
                                 <double*>np.PyArray_DATA(kp1_Phiy), kp1_Phiy.size,  1, # out
                                 <double*>np.PyArray_DATA(kp1_Phiz), kp1_Phiz.size,  1, # out
                                 <double*>np.PyArray_DATA(kp1_psi),  kp1_psi.size,   1, # out
                                 <double*>np.PyArray_DATA(kp1_u),    kp1_u.size,     1) # out
    else:
        cda_time_scalar_3D_OS_6( <float*>np.PyArray_DATA(km1_u),  km1_u.size,  1,      # in - padded wavefield shape
                                 <float*>np.PyArray_DATA(k_Phix), k_Phix.size, 1,      # in - padded wavefield shape
                                 <float*>np.PyArray_DATA(k_Phiy), k_Phiy.size, 1,      # in - padded wavefield shape
                                 <float*>np.PyArray_DATA(k_Phiz), k_Phiz.size, 1,      # in - padded wavefield shape
                                 <float*>np.PyArray_DATA(k_psi),  k_psi.size,  1,      # in - padded wavefield shape
                                 <float*>np.PyArray_DATA(k_u),    k_u.size,    1,      # in - padded wavefield shape
                                 <float*>np.PyArray_DATA(C),      C.size,      1,      # in - padded wavefield shape
                                 <float*>np.PyArray_DATA(rhs),    rhs.size,    1,      # in - padded wavefield shape
                                 <float*>np.PyArray_DATA(xlpml),  xlpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 <float*>np.PyArray_DATA(xrpml),  xrpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 <float*>np.PyArray_DATA(ylpml),  ylpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 <float*>np.PyArray_DATA(yrpml),  yrpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 <float*>np.PyArray_DATA(zlpml),  zlpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 <float*>np.PyArray_DATA(zrpml),  zrpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 dt,                                                    # in
                                 dx,                                                    # in
                                 dy,                                                    # in
                                 dz,                                                    # in
                                 nx,                                                    # in
                                 ny,                                                    # in
                                 nz,                                                    # in
                                 <float*>np.PyArray_DATA(kp1_Phix), kp1_Phix.size,  1, # out
                                 <float*>np.PyArray_DATA(kp1_Phiy), kp1_Phiy.size,  1, # out
                                 <float*>np.PyArray_DATA(kp1_Phiz), kp1_Phiz.size,  1, # out
                                 <float*>np.PyArray_DATA(kp1_psi),  kp1_psi.size,   1, # out
                                 <float*>np.PyArray_DATA(kp1_u),    kp1_u.size,     1) # out


def constant_density_acoustic_time_scalar_3D_8os( km1_u,   k_Phix,   k_Phiy,   k_Phiz,
                                                  k_psi,   k_u,      C,        rhs,
                                                  xlpml,   xrpml,    ylpml,    yrpml,
                                                  zlpml,   zrpml,    dt,       dx,
                                                  dy,      dz,       nx,       ny,
                                                  nz,      kp1_Phix, kp1_Phiy, kp1_Phiz,
                                                  kp1_psi, kp1_u):
    if km1_u.dtype == 'double':
        cda_time_scalar_3D_OS_8( <double*>np.PyArray_DATA(km1_u),  km1_u.size,  1,      # in - padded wavefield shape
                                 <double*>np.PyArray_DATA(k_Phix), k_Phix.size, 1,      # in - padded wavefield shape
                                 <double*>np.PyArray_DATA(k_Phiy), k_Phiy.size, 1,      # in - padded wavefield shape
                                 <double*>np.PyArray_DATA(k_Phiz), k_Phiz.size, 1,      # in - padded wavefield shape
                                 <double*>np.PyArray_DATA(k_psi),  k_psi.size,  1,      # in - padded wavefield shape
                                 <double*>np.PyArray_DATA(k_u),    k_u.size,    1,      # in - padded wavefield shape
                                 <double*>np.PyArray_DATA(C),      C.size,      1,      # in - padded wavefield shape
                                 <double*>np.PyArray_DATA(rhs),    rhs.size,    1,      # in - padded wavefield shape
                                 <double*>np.PyArray_DATA(xlpml),  xlpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 <double*>np.PyArray_DATA(xrpml),  xrpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 <double*>np.PyArray_DATA(ylpml),  ylpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 <double*>np.PyArray_DATA(yrpml),  yrpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 <double*>np.PyArray_DATA(zlpml),  zlpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 <double*>np.PyArray_DATA(zrpml),  zrpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 dt,                                                    # in
                                 dx,                                                    # in
                                 dy,                                                    # in
                                 dz,                                                    # in
                                 nx,                                                    # in
                                 ny,                                                    # in
                                 nz,                                                    # in
                                 <double*>np.PyArray_DATA(kp1_Phix), kp1_Phix.size,  1, # out
                                 <double*>np.PyArray_DATA(kp1_Phiy), kp1_Phiy.size,  1, # out
                                 <double*>np.PyArray_DATA(kp1_Phiz), kp1_Phiz.size,  1, # out
                                 <double*>np.PyArray_DATA(kp1_psi),  kp1_psi.size,   1, # out
                                 <double*>np.PyArray_DATA(kp1_u),    kp1_u.size,     1) # out
    else:
        cda_time_scalar_3D_OS_8( <float*>np.PyArray_DATA(km1_u),  km1_u.size,  1,      # in - padded wavefield shape
                                 <float*>np.PyArray_DATA(k_Phix), k_Phix.size, 1,      # in - padded wavefield shape
                                 <float*>np.PyArray_DATA(k_Phiy), k_Phiy.size, 1,      # in - padded wavefield shape
                                 <float*>np.PyArray_DATA(k_Phiz), k_Phiz.size, 1,      # in - padded wavefield shape
                                 <float*>np.PyArray_DATA(k_psi),  k_psi.size,  1,      # in - padded wavefield shape
                                 <float*>np.PyArray_DATA(k_u),    k_u.size,    1,      # in - padded wavefield shape
                                 <float*>np.PyArray_DATA(C),      C.size,      1,      # in - padded wavefield shape
                                 <float*>np.PyArray_DATA(rhs),    rhs.size,    1,      # in - padded wavefield shape
                                 <float*>np.PyArray_DATA(xlpml),  xlpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 <float*>np.PyArray_DATA(xrpml),  xrpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 <float*>np.PyArray_DATA(ylpml),  ylpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 <float*>np.PyArray_DATA(yrpml),  yrpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 <float*>np.PyArray_DATA(zlpml),  zlpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 <float*>np.PyArray_DATA(zrpml),  zrpml.size,          # in - length is the number of nodes inside the padding that the pml value is defined.
                                 dt,                                                    # in
                                 dx,                                                    # in
                                 dy,                                                    # in
                                 dz,                                                    # in
                                 nx,                                                    # in
                                 ny,                                                    # in
                                 nz,                                                    # in
                                 <float*>np.PyArray_DATA(kp1_Phix), kp1_Phix.size,  1, # out
                                 <float*>np.PyArray_DATA(kp1_Phiy), kp1_Phiy.size,  1, # out
                                 <float*>np.PyArray_DATA(kp1_Phiz), kp1_Phiz.size,  1, # out
                                 <float*>np.PyArray_DATA(kp1_psi),  kp1_psi.size,   1, # out
                                 <float*>np.PyArray_DATA(kp1_u),    kp1_u.size,     1) # out
