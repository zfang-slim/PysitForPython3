# include <boost/python.hpp>
// # include "constant_density_acoustic_time_scalar.i"
# include "constant_density_acoustic_time_scalar_1D.h"
using namespace boost::python;

template<typename T>
void printabc(T input)
{
    return;
}

BOOST_PYTHON_MODULE(_constant_density_acoustic_time_scalar_cpp)
{
    // 1D
    // def("constant_density_acoustic_time_scalar_1D_4omp", constant_density_acoustic_time_scalar_1D_4omp);
    // def("constant_density_acoustic_time_scalar_1D_6omp", constant_density_acoustic_time_scalar_1D_6omp);
    def("constant_density_acoustic_time_scalar_1D_2os", printabc<float>);
    // def("constant_density_acoustic_time_scalar_1D_4os", constant_density_acoustic_time_scalar_1D_4os);
    // def("constant_density_acoustic_time_scalar_1D_6os", constant_density_acoustic_time_scalar_1D_6os);
    // def("constant_density_acoustic_time_scalar_1D_8os", constant_density_acoustic_time_scalar_1D_8os);
    //
    // // 2D
    // def("constant_density_acoustic_time_scalar_2D_4omp", constant_density_acoustic_time_scalar_2D_4omp);
    // def("constant_density_acoustic_time_scalar_2D_6omp", constant_density_acoustic_time_scalar_2D_6omp);
    // def("constant_density_acoustic_time_scalar_2D_2os", constant_density_acoustic_time_scalar_2D_2os);
    // def("constant_density_acoustic_time_scalar_2D_4os", constant_density_acoustic_time_scalar_2D_4os);
    // def("constant_density_acoustic_time_scalar_2D_6os", constant_density_acoustic_time_scalar_2D_6os);
    // def("constant_density_acoustic_time_scalar_2D_8os", constant_density_acoustic_time_scalar_2D_8os);
    //
    // // 3D
    // def("constant_density_acoustic_time_scalar_3D_4omp", constant_density_acoustic_time_scalar_3D_4omp);
    // def("constant_density_acoustic_time_scalar_3D_6omp", constant_density_acoustic_time_scalar_3D_6omp);
    // def("constant_density_acoustic_time_scalar_3D_2os", constant_density_acoustic_time_scalar_3D_2os);
    // def("constant_density_acoustic_time_scalar_3D_4os", constant_density_acoustic_time_scalar_3D_4os);
    // def("constant_density_acoustic_time_scalar_3D_6os", constant_density_acoustic_time_scalar_3D_6os);
    // def("constant_density_acoustic_time_scalar_3D_8os", constant_density_acoustic_time_scalar_3D_8os);


}
