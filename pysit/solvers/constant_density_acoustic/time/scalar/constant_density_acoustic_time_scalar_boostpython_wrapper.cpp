# include <boost/python.hpp>
// # include "constant_density_acoustic_time_scalar.i"
# include "constant_density_acoustic_time_scalar_1D.h"
# include "constant_density_acoustic_time_scalar_2D.h"
# include "constant_density_acoustic_time_scalar_3D.h"
using namespace boost::python;


// template< typename T, int ACCURACY >
// void cda_time_scalar_1D(      T* km1_u,  int nr_km1_u,  int nc_km1_u,      // in - padded wavefield shape
//                               T* k_Phiz, int nr_k_Phiz, int nc_k_Phiz,     // in - padded wavefield shape
//                               T* k_u,    int nr_k_u,    int nc_k_u,        // in - padded wavefield shape
//                               T* C,      int nr_C,      int nc_C,          // in - padded wavefield shape
//                               T* rhs,    int nr_rhs,    int nc_rhs,        // in - padded wavefield shape
//                               T* zlpml,  int n_zlpml,                      // in - length is the number of nodes inside the padding that the pml value is defined.
//                               T* zrpml,  int n_zrpml,                      // in - length is the number of nodes inside the padding that the pml value is defined.
//                               double const& dt,                            // in
//                               double const& dz,                            // in
//                               int const& nz,                               // in
//                               T* kp1_Phiz, int nr_kp1_Phiz,  int nc_kp1_Phiz,  // out
//                               T* kp1_u,    int nr_kp1_u,     int nc_kp1_u   )  // out
// {
//     return;
// }


BOOST_PYTHON_MODULE(_constant_density_acoustic_time_scalar_cpp)
{
    // 1D
    // def("constant_density_acoustic_time_scalar_1D_4omp", constant_density_acoustic_time_scalar_1D_4omp);
    // def("constant_density_acoustic_time_scalar_1D_6omp", constant_density_acoustic_time_scalar_1D_6omp);
    def("constant_density_acoustic_time_scalar_1D_2os", cda_time_scalar_1D<float, 2>);
    def("constant_density_acoustic_time_scalar_1D_2os", cda_time_scalar_1D<double, 2>);
    // def("constant_density_acoustic_time_scalar_1D_4os", constant_density_acoustic_time_scalar_1D_4os);
    // def("constant_density_acoustic_time_scalar_1D_6os", constant_density_acoustic_time_scalar_1D_6os);
    // def("constant_density_acoustic_time_scalar_1D_8os", constant_density_acoustic_time_scalar_1D_8os);
    //
    // // 2D
    // def("constant_density_acoustic_time_scalar_2D_4omp", constant_density_acoustic_time_scalar_2D_4omp);
    // def("constant_density_acoustic_time_scalar_2D_6omp", constant_density_acoustic_time_scalar_2D_6omp);
    def("constant_density_acoustic_time_scalar_2D_2os", cda_time_scalar_2D<float, 2>);
    // def("constant_density_acoustic_time_scalar_2D_4os", constant_density_acoustic_time_scalar_2D_4os);
    // def("constant_density_acoustic_time_scalar_2D_6os", constant_density_acoustic_time_scalar_2D_6os);
    // def("constant_density_acoustic_time_scalar_2D_8os", constant_density_acoustic_time_scalar_2D_8os);
    //
    // // 3D
    // def("constant_density_acoustic_time_scalar_3D_4omp", constant_density_acoustic_time_scalar_3D_4omp);
    // def("constant_density_acoustic_time_scalar_3D_6omp", constant_density_acoustic_time_scalar_3D_6omp);
    // def("constant_density_acoustic_time_scalar_3D_2os", cda_time_scalar_3D<float, 2>);
    // def("constant_density_acoustic_time_scalar_3D_4os", constant_density_acoustic_time_scalar_3D_4os);
    // def("constant_density_acoustic_time_scalar_3D_6os", constant_density_acoustic_time_scalar_3D_6os);
    // def("constant_density_acoustic_time_scalar_3D_8os", constant_density_acoustic_time_scalar_3D_8os);


}
