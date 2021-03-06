#ifndef __CDA_TIME_SCALAR_2D__
#define __CDA_TIME_SCALAR_2D__

#include <iostream>

#include <fstream>
//#include <time.h>
#include <ctime>
#include <chrono>
using namespace std::chrono;
using namespace std;

#include "../../../fd_tools/fd_manual.hpp"

template <typename T>
class FDArg2D
{
    public:
        T* U;
        T* Phi;

        T delta;
        T delta2;

        T& dU;
        T& dPhi;
        T& lapU;

        FDArg2D<T>(T* u, T* pPhi, T& du, T& dP, T& lapu, T const& del) : U(u),Phi(pPhi),dU(du),dPhi(dP),lapU(lapu),delta(del){delta2 = delta*delta;};

        void reset()
        {
            lapU = 0.0;
        };
};

template <typename T, int ACCURACY, int SHIFT>
class FDAction2D
{
    public:
        static
        void execute(FDArg2D<T>& arg, int const& idx, int const& stride)
        {
            arg.dU    = FD<T,1,ACCURACY,SHIFT>::apply(arg.U, idx, stride, arg.delta);
            arg.dPhi  = FD<T,1,ACCURACY,SHIFT>::apply(arg.Phi, idx, stride, arg.delta);
            arg.lapU += FD<T,2,ACCURACY,SHIFT>::apply(arg.U, idx, stride, arg.delta2);
        };
};

template< typename T, int ACCURACY >
void cda_time_scalar_2D(      T* km1_u,  int nr_km1_u,  int nc_km1_u,      // in - padded wavefield shape
                              T* k_Phix, int nr_k_Phix, int nc_k_Phix,     // in - padded wavefield shape
                              T* k_Phiz, int nr_k_Phiz, int nc_k_Phiz,     // in - padded wavefield shape
                              T* k_u,    int nr_k_u,    int nc_k_u,        // in - padded wavefield shape
                              T* C,      int nr_C,      int nc_C,          // in - padded wavefield shape
                              T* rhs,    int nr_rhs,    int nc_rhs,        // in - padded wavefield shape
                              T* xlpml,  int n_xlpml,                      // in - length is the number of nodes inside the padding that the pml value is defined.
                              T* xrpml,  int n_xrpml,                      // in - length is the number of nodes inside the padding that the pml value is defined.
                              T* zlpml,  int n_zlpml,                      // in - length is the number of nodes inside the padding that the pml value is defined.
                              T* zrpml,  int n_zrpml,                      // in - length is the number of nodes inside the padding that the pml value is defined.
                              double const& dt,                            // in
                              double const& dx,                            // in
                              double const& dz,                            // in
                              int const& nx,                               // in
                              int const& nz,                               // in
                              T* kp1_Phix, int nr_kp1_Phix,  int nc_kp1_Phix,  // out
                              T* kp1_Phiz, int nr_kp1_Phiz,  int nc_kp1_Phiz,  // out
                              T* kp1_u,    int nr_kp1_u,     int nc_kp1_u   )  // out
{
    enum {MAX_FD_SHIFT = ACCURACY/2};

    // Derivative variables
    T dUdx = 0.0;
    T dUdz = 0.0;
    T dPhixdx = 0.0;
    T dPhizdz = 0.0;
    T lapU = 0.0;

    // PML variable
    T sigmax = 0.0;
    T sigmaz = 0.0;

    // Time delta variables
    T dt2 = dt*dt;

    // Loop/index variables
    int idx;
    int zstride = 1;
    int xstride = nz;

    // Wrapper class for the arrays and references to the derivative variables.
    FDArg2D<T> x_args(k_u, k_Phix, dUdx, dPhixdx, lapU, dx);
    FDArg2D<T> z_args(k_u, k_Phiz, dUdz, dPhizdz, lapU, dz);

    for(int i=0; i < nx; ++i)
    {
        for(int k=0; k < nz; k++)
        {
            idx = i*xstride + k;

            kp1_Phix[idx] = 0.0;
            kp1_Phiz[idx] = 0.0;
            kp1_u[idx]    = 0.0;

            // This handles homogeneous Dirichlet BCs and non-updating in ghost regions.
            if ((i == 0) || (i == nx-1)) continue;
            if ((k == 0) || (k == nz-1)) continue;

            // Reset the derivative values
            x_args.reset();
            z_args.reset();

            // Do the X direction
            // Left side
            if     ((ACCURACY >= 2) && (i == 0)) FDAction2D<T, ACCURACY, -MAX_FD_SHIFT+0>::execute(x_args, idx, xstride);
            else if((ACCURACY >= 4) && (i == 1)) FDAction2D<T, ACCURACY, -MAX_FD_SHIFT+1>::execute(x_args, idx, xstride);
            else if((ACCURACY >= 6) && (i == 2)) FDAction2D<T, ACCURACY, -MAX_FD_SHIFT+2>::execute(x_args, idx, xstride);
            else if((ACCURACY >= 8) && (i == 3)) FDAction2D<T, ACCURACY, -MAX_FD_SHIFT+3>::execute(x_args, idx, xstride);
            // Right side
            else if((ACCURACY >= 2) && (i == nx-1)) FDAction2D<T, ACCURACY, MAX_FD_SHIFT-0>::execute(x_args, idx, xstride);
            else if((ACCURACY >= 4) && (i == nx-2)) FDAction2D<T, ACCURACY, MAX_FD_SHIFT-1>::execute(x_args, idx, xstride);
            else if((ACCURACY >= 6) && (i == nx-3)) FDAction2D<T, ACCURACY, MAX_FD_SHIFT-2>::execute(x_args, idx, xstride);
            else if((ACCURACY >= 8) && (i == nx-4)) FDAction2D<T, ACCURACY, MAX_FD_SHIFT-3>::execute(x_args, idx, xstride);
            // Bulk
            else FDAction2D<T, ACCURACY, 0>::execute(x_args, idx, xstride);

            // Do the Z direction
            // Left side
            if     ((ACCURACY >= 2) && (k == 0)) FDAction2D<T, ACCURACY, -MAX_FD_SHIFT+0>::execute(z_args, idx, zstride);
            else if((ACCURACY >= 4) && (k == 1)) FDAction2D<T, ACCURACY, -MAX_FD_SHIFT+1>::execute(z_args, idx, zstride);
            else if((ACCURACY >= 6) && (k == 2)) FDAction2D<T, ACCURACY, -MAX_FD_SHIFT+2>::execute(z_args, idx, zstride);
            else if((ACCURACY >= 8) && (k == 3)) FDAction2D<T, ACCURACY, -MAX_FD_SHIFT+3>::execute(z_args, idx, zstride);
            // Right side
            else if((ACCURACY >= 2) && (k == nz-1)) FDAction2D<T, ACCURACY, MAX_FD_SHIFT-0>::execute(z_args, idx, zstride);
            else if((ACCURACY >= 4) && (k == nz-2)) FDAction2D<T, ACCURACY, MAX_FD_SHIFT-1>::execute(z_args, idx, zstride);
            else if((ACCURACY >= 6) && (k == nz-3)) FDAction2D<T, ACCURACY, MAX_FD_SHIFT-2>::execute(z_args, idx, zstride);
            else if((ACCURACY >= 8) && (k == nz-4)) FDAction2D<T, ACCURACY, MAX_FD_SHIFT-3>::execute(z_args, idx, zstride);
            // Bulk
            else FDAction2D<T, ACCURACY, 0>::execute(z_args, idx, zstride);

            sigmax = 0.0;
            sigmaz = 0.0;

            // Check if in left PML-X
            if((n_xlpml>0) && (i < n_xlpml))
            {
                sigmax = xlpml[i];
            }
            // Check if in right PML-X
            else if((n_xrpml>0) && (i >= nx-n_xrpml))
            {
                sigmax = xrpml[n_xrpml-((nx-1)-i)];
            }

            // Check if in left PML-Z
            if((n_zlpml>0) && (k < n_zlpml))
            {
                sigmaz = zlpml[k];
            }
            // Check if in right PML-Z
            else if((n_zrpml>0) && (k >= nz-n_zrpml))
            {
                sigmaz = zrpml[n_zrpml-((nz-1)-k)]; // 0th element of the right pml array corresponds to n_zrpml'th node from the right boundary.
            }

            if((sigmaz != 0.0) || (sigmax != 0.0))
            {
                kp1_Phix[idx] = k_Phix[idx] - dt*sigmax*k_Phix[idx] + dt*(sigmaz-sigmax)*dUdx;
                kp1_Phiz[idx] = k_Phiz[idx] - dt*sigmaz*k_Phiz[idx] + dt*(sigmax-sigmaz)*dUdz;


                T fac1 = (2.0*dt2 / (2.0 + dt*(sigmax+sigmaz)));
                T fac2 = (C[idx]*C[idx])*(rhs[idx]+lapU+dPhixdx+dPhizdz) - (km1_u[idx]-2.0*k_u[idx])/dt2 + (sigmax+sigmaz)*km1_u[idx]/(2.0*dt) - (sigmax*sigmaz)*k_u[idx];
                kp1_u[idx] = fac1 * fac2;
            }
            else
            {
                kp1_Phix[idx] = k_Phix[idx];
                kp1_Phiz[idx] = k_Phiz[idx];
                kp1_u[idx] = dt2*(C[idx]*C[idx])*(rhs[idx]+lapU+dPhixdx+dPhizdz) - (km1_u[idx]-2.0*k_u[idx]);
            }



        }
    }
};


template< typename T>
void cda_time_scalar_2D_OS_2(     T* km1_u,  int nr_km1_u,  int nc_km1_u,      // in - padded wavefield shape
                                  T* k_Phix, int nr_k_Phix, int nc_k_Phix,     // in - padded wavefield shape
                                  T* k_Phiz, int nr_k_Phiz, int nc_k_Phiz,     // in - padded wavefield shape
                                  T* k_u,    int nr_k_u,    int nc_k_u,        // in - padded wavefield shape
                                  T* C,      int nr_C,      int nc_C,          // in - padded wavefield shape
                                  T* rhs,    int nr_rhs,    int nc_rhs,        // in - padded wavefield shape
                                  T* xlpml,  int n_xlpml,                      // in - length is the number of nodes inside the padding that the pml value is defined.
                                  T* xrpml,  int n_xrpml,                      // in - length is the number of nodes inside the padding that the pml value is defined.
                                  T* zlpml,  int n_zlpml,                      // in - length is the number of nodes inside the padding that the pml value is defined.
                                  T* zrpml,  int n_zrpml,                      // in - length is the number of nodes inside the padding that the pml value is defined.
                                  double const& dt,                            // in
                                  double const& dx,                            // in
                                  double const& dz,                            // in
                                  int const& nx,                               // in
                                  int const& nz,                               // in
                                  T* kp1_Phix, int nr_kp1_Phix,  int nc_kp1_Phix,  // out
                                  T* kp1_Phiz, int nr_kp1_Phiz,  int nc_kp1_Phiz,  // out
                                  T* kp1_u,    int nr_kp1_u,     int nc_kp1_u   )  // out
{
    // time_t start = time(NULL);
    // int start_s=clock();
    // auto start = high_resolution_clock::now();
    cda_time_scalar_2D<T,2>(      km1_u,  nr_km1_u,   nc_km1_u,      // in - padded wavefield shape
                                  k_Phix, nr_k_Phix,  nc_k_Phix,     // in - padded wavefield shape
                                  k_Phiz, nr_k_Phiz,  nc_k_Phiz,     // in - padded wavefield shape
                                  k_u,    nr_k_u,     nc_k_u,        // in - padded wavefield shape
                                  C,      nr_C,       nc_C,          // in - padded wavefield shape
                                  rhs,    nr_rhs,     nc_rhs,        // in - padded wavefield shape
                                  xlpml,  n_xlpml,                   // in - length is the number of nodes inside the padding that the pml value is defined.
                                  xrpml,  n_xrpml,                   // in - length is the number of nodes inside the padding that the pml value is defined.
                                  zlpml,  n_zlpml,                   // in - length is the number of nodes inside the padding that the pml value is defined.
                                  zrpml,  n_zrpml,                   // in - length is the number of nodes inside the padding that the pml value is defined.
                                  dt,                                 // in
                                  dx,                                 // in
                                  dz,                                 // in
                                  nx,                                 // in
                                  nz,                                 // in
                                  kp1_Phix, nr_kp1_Phix,  nc_kp1_Phix,   // out
                                  kp1_Phiz, nr_kp1_Phiz,  nc_kp1_Phiz,   // out
                                  kp1_u,    nr_kp1_u,     nc_kp1_u   );  // out

    // time_t end = time(NULL);
    // std::cout<<"Execution Time: "<< (double)(end-start)<<" Seconds"<<std::endl;

    // int stop_s=clock();
    // cout << "time: " << (stop_s-start_s)/double(CLOCKS_PER_SEC)*1000 << endl;

    // auto stop = high_resolution_clock::now();
    // auto duration = duration_cast<microseconds>(stop - start);
    // cout << duration.count() << endl;

}

template< typename T>
void cda_time_scalar_2D_OS_4(     T* km1_u,  int nr_km1_u,  int nc_km1_u,      // in - padded wavefield shape
                                  T* k_Phix, int nr_k_Phix, int nc_k_Phix,     // in - padded wavefield shape
                                  T* k_Phiz, int nr_k_Phiz, int nc_k_Phiz,     // in - padded wavefield shape
                                  T* k_u,    int nr_k_u,    int nc_k_u,        // in - padded wavefield shape
                                  T* C,      int nr_C,      int nc_C,          // in - padded wavefield shape
                                  T* rhs,    int nr_rhs,    int nc_rhs,        // in - padded wavefield shape
                                  T* xlpml,  int n_xlpml,                      // in - length is the number of nodes inside the padding that the pml value is defined.
                                  T* xrpml,  int n_xrpml,                      // in - length is the number of nodes inside the padding that the pml value is defined.
                                  T* zlpml,  int n_zlpml,                      // in - length is the number of nodes inside the padding that the pml value is defined.
                                  T* zrpml,  int n_zrpml,                      // in - length is the number of nodes inside the padding that the pml value is defined.
                                  double const& dt,                            // in
                                  double const& dx,                            // in
                                  double const& dz,                            // in
                                  int const& nx,                               // in
                                  int const& nz,                               // in
                                  T* kp1_Phix, int nr_kp1_Phix,  int nc_kp1_Phix,  // out
                                  T* kp1_Phiz, int nr_kp1_Phiz,  int nc_kp1_Phiz,  // out
                                  T* kp1_u,    int nr_kp1_u,     int nc_kp1_u   )  // out
{
    // auto start = high_resolution_clock::now();
    cda_time_scalar_2D<T,4>(      km1_u,  nr_km1_u,   nc_km1_u,      // in - padded wavefield shape
                                  k_Phix, nr_k_Phix,  nc_k_Phix,     // in - padded wavefield shape
                                  k_Phiz, nr_k_Phiz,  nc_k_Phiz,     // in - padded wavefield shape
                                  k_u,    nr_k_u,     nc_k_u,        // in - padded wavefield shape
                                  C,      nr_C,       nc_C,          // in - padded wavefield shape
                                  rhs,    nr_rhs,     nc_rhs,        // in - padded wavefield shape
                                  xlpml,  n_xlpml,                   // in - length is the number of nodes inside the padding that the pml value is defined.
                                  xrpml,  n_xrpml,                   // in - length is the number of nodes inside the padding that the pml value is defined.
                                  zlpml,  n_zlpml,                   // in - length is the number of nodes inside the padding that the pml value is defined.
                                  zrpml,  n_zrpml,                   // in - length is the number of nodes inside the padding that the pml value is defined.
                                  dt,                                // in
                                  dx,                                // in
                                  dz,                                // in
                                  nx,                                // in
                                  nz,                                // in
                                  kp1_Phix, nr_kp1_Phix,  nc_kp1_Phix,  // out
                                  kp1_Phiz, nr_kp1_Phiz,  nc_kp1_Phiz,  // out
                                  kp1_u,    nr_kp1_u,     nc_kp1_u   ); // out


    // auto stop = high_resolution_clock::now();
    // auto duration = duration_cast<microseconds>(stop - start);
    // cout << duration.count() << endl;
}

template< typename T>
void cda_time_scalar_2D_OS_6(     T* km1_u,  int nr_km1_u,  int nc_km1_u,      // in - padded wavefield shape
                                  T* k_Phix, int nr_k_Phix, int nc_k_Phix,     // in - padded wavefield shape
                                  T* k_Phiz, int nr_k_Phiz, int nc_k_Phiz,     // in - padded wavefield shape
                                  T* k_u,    int nr_k_u,    int nc_k_u,        // in - padded wavefield shape
                                  T* C,      int nr_C,      int nc_C,          // in - padded wavefield shape
                                  T* rhs,    int nr_rhs,    int nc_rhs,        // in - padded wavefield shape
                                  T* xlpml,  int n_xlpml,                      // in - length is the number of nodes inside the padding that the pml value is defined.
                                  T* xrpml,  int n_xrpml,                      // in - length is the number of nodes inside the padding that the pml value is defined.
                                  T* zlpml,  int n_zlpml,                      // in - length is the number of nodes inside the padding that the pml value is defined.
                                  T* zrpml,  int n_zrpml,                      // in - length is the number of nodes inside the padding that the pml value is defined.
                                  double const& dt,                            // in
                                  double const& dx,                            // in
                                  double const& dz,                            // in
                                  int const& nx,                               // in
                                  int const& nz,                               // in
                                  T* kp1_Phix, int nr_kp1_Phix,  int nc_kp1_Phix,  // out
                                  T* kp1_Phiz, int nr_kp1_Phiz,  int nc_kp1_Phiz,  // out
                                  T* kp1_u,    int nr_kp1_u,     int nc_kp1_u   )  // out
{
    // auto start = high_resolution_clock::now();
    cda_time_scalar_2D<T,6>(      km1_u,  nr_km1_u,   nc_km1_u,      // in - padded wavefield shape
                                  k_Phix, nr_k_Phix,  nc_k_Phix,     // in - padded wavefield shape
                                  k_Phiz, nr_k_Phiz,  nc_k_Phiz,     // in - padded wavefield shape
                                  k_u,    nr_k_u,     nc_k_u,        // in - padded wavefield shape
                                  C,      nr_C,       nc_C,          // in - padded wavefield shape
                                  rhs,    nr_rhs,     nc_rhs,        // in - padded wavefield shape
                                  xlpml,  n_xlpml,                   // in - length is the number of nodes inside the padding that the pml value is defined.
                                  xrpml,  n_xrpml,                   // in - length is the number of nodes inside the padding that the pml value is defined.
                                  zlpml,  n_zlpml,                   // in - length is the number of nodes inside the padding that the pml value is defined.
                                  zrpml,  n_zrpml,                   // in - length is the number of nodes inside the padding that the pml value is defined.
                                  dt,                                // in
                                  dx,                                // in
                                  dz,                                // in
                                  nx,                                // in
                                  nz,                                // in
                                  kp1_Phix, nr_kp1_Phix,  nc_kp1_Phix,  // out
                                  kp1_Phiz, nr_kp1_Phiz,  nc_kp1_Phiz,  // out
                                  kp1_u,    nr_kp1_u,     nc_kp1_u   ); // out

    // auto stop = high_resolution_clock::now();
    // auto duration = duration_cast<microseconds>(stop - start);
    // cout << duration.count() << endl;
}

template< typename T>
void cda_time_scalar_2D_OS_8(     T* km1_u,  int nr_km1_u,  int nc_km1_u,      // in - padded wavefield shape
                                  T* k_Phix, int nr_k_Phix, int nc_k_Phix,     // in - padded wavefield shape
                                  T* k_Phiz, int nr_k_Phiz, int nc_k_Phiz,     // in - padded wavefield shape
                                  T* k_u,    int nr_k_u,    int nc_k_u,        // in - padded wavefield shape
                                  T* C,      int nr_C,      int nc_C,          // in - padded wavefield shape
                                  T* rhs,    int nr_rhs,    int nc_rhs,        // in - padded wavefield shape
                                  T* xlpml,  int n_xlpml,                      // in - length is the number of nodes inside the padding that the pml value is defined.
                                  T* xrpml,  int n_xrpml,                      // in - length is the number of nodes inside the padding that the pml value is defined.
                                  T* zlpml,  int n_zlpml,                      // in - length is the number of nodes inside the padding that the pml value is defined.
                                  T* zrpml,  int n_zrpml,                      // in - length is the number of nodes inside the padding that the pml value is defined.
                                  double const& dt,                            // in
                                  double const& dx,                            // in
                                  double const& dz,                            // in
                                  int const& nx,                               // in
                                  int const& nz,                               // in
                                  T* kp1_Phix, int nr_kp1_Phix,  int nc_kp1_Phix,  // out
                                  T* kp1_Phiz, int nr_kp1_Phiz,  int nc_kp1_Phiz,  // out
                                  T* kp1_u,    int nr_kp1_u,     int nc_kp1_u   )  // out
{
    // auto start = high_resolution_clock::now();
    cda_time_scalar_2D<T,8>(      km1_u,  nr_km1_u,   nc_km1_u,      // in - padded wavefield shape
                                  k_Phix, nr_k_Phix,  nc_k_Phix,     // in - padded wavefield shape
                                  k_Phiz, nr_k_Phiz,  nc_k_Phiz,     // in - padded wavefield shape
                                  k_u,    nr_k_u,     nc_k_u,        // in - padded wavefield shape
                                  C,      nr_C,       nc_C,          // in - padded wavefield shape
                                  rhs,    nr_rhs,     nc_rhs,        // in - padded wavefield shape
                                  xlpml,  n_xlpml,                   // in - length is the number of nodes inside the padding that the pml value is defined.
                                  xrpml,  n_xrpml,                   // in - length is the number of nodes inside the padding that the pml value is defined.
                                  zlpml,  n_zlpml,                   // in - length is the number of nodes inside the padding that the pml value is defined.
                                  zrpml,  n_zrpml,                   // in - length is the number of nodes inside the padding that the pml value is defined.
                                  dt,                                // in
                                  dx,                                // in
                                  dz,                                // in
                                  nx,                                // in
                                  nz,                                // in
                                  kp1_Phix, nr_kp1_Phix,  nc_kp1_Phix,  // out
                                  kp1_Phiz, nr_kp1_Phiz,  nc_kp1_Phiz,  // out
                                  kp1_u,    nr_kp1_u,     nc_kp1_u   ); // out


    // auto stop = high_resolution_clock::now();
    // auto duration = duration_cast<microseconds>(stop - start);
    // cout << duration.count() << endl;
}

#endif
