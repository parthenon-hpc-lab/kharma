


#include "tetrads.hpp"

namespace Cameras {

/*
 * Make orthonormal basis for camera frame.
 *
 * e^0 along Ucam
 * e^3 outward (!) along radius vector
 * e^2 toward north pole of coordinate system ("y" in the image plane)
 * e^1 in the remaining direction ("x" in the image plane)
 *
 * This combination measures the final Stokes parameters correctly (IEEE/IAS).
 * These values are then translated if a different convention is to be output.
 *
 * Points the camera so that the angular momentum k_{th,phi} at FOV center is 0
 */
KOKKOS_INLINE_FUNCTION int make_camera_tetrad(const double Gcov[GR_DIM][GR_DIM],
    const double X[GR_DIM], double Econ[GR_DIM][GR_DIM], double Ecov[GR_DIM][GR_DIM])
{
  double Ucam[GR_DIM], Kcon[GR_DIM], trial[GR_DIM];
  double Gcon[GR_DIM][GR_DIM];
  gcon_func(Gcov, Gcon);

  // center the camera according to impact parameter, i.e., make it
  // so that Kcontetrad = ( 1, 0, 0, 1 ) corresponds to an outgoing
  // wavevector with zero angular momentum / zero impact parameter.

  // use normal observer velocity. this forces (Gcov.Econ[0])[3] = 0.
  trial[0] = -1.;
  trial[1] = 0.;
  trial[2] = 0.;
  trial[3] = 0.;
  flip_index(trial, Gcon, Ucam);

  // set Kcon (becomes Econ[3][mu]) outward directed with central 
  // pixel k_phi = 0. this ensures that a photon with zero impact 
  // parameter will be in the center of the field of view.
  trial[0] = 1.;
  trial[1] = 1.;
  trial[2] = 0.;
  trial[3] = 0.;
  flip_index(trial, Gcon, Kcon);

  // set the y camera direction to be parallel to the projected
  // spin axis of the black hole (on the image plane defined to
  // be normal to the Kcon vector above).
  trial[0] = 0.;
  trial[1] = 0.;
  trial[2] = 1.;
  trial[3] = 0.;

  int sing = make_plasma_tetrad(Ucam, Kcon, trial, Gcov, Econ, Ecov);
#if DEBUG
  if(sing) {
    fprintf(stderr, "\nError making Camera tetrad, something is wrong!\n");
    fprintf(stderr, "Used the following vectors:\n");
    print_vector("X", X);
    print_vector("Ucam", Ucam);
    print_vector("Kcon", Kcon);
    print_vector("trial", trial);
    print_matrix("gcov", Gcov);
    print_matrix("gcon", Gcon);
  }
#endif
  return sing;
}

/*
 * Make orthonormal basis for camera frame -- old implementation
 *
 * e^0 along Ucam
 * e^3 outward (!) along radius vector
 * e^2 toward north pole of coordinate system ("y" in the image plane)
 * e^1 in the remaining direction ("x" in the image plane)
 *
 * This combination measures the final Stokes parameters correctly (IEEE/IAS).
 * These values are then translated if a different convention is to be output.
 *
 * Points the camera so that the *contravariant wavevector* k^{th,phi} = 0
 */
KOKKOS_INLINE_FUNCTION int make_camera_tetrad_old(const double Gcov[GR_DIM][GR_DIM],
    const double X[GR_DIM], double Econ[GR_DIM][GR_DIM], double Ecov[GR_DIM][GR_DIM])
{
    double Ucam[GR_DIM], Kcon[GR_DIM], trial[GR_DIM];
    double Gcon[GR_DIM][GR_DIM];
    gcon_func(Gcov, Gcon);

    // old centering method
    Ucam[0] = 1.;
    Ucam[1] = 0.;
    Ucam[2] = 0.;
    Ucam[3] = 0.;

    trial[0] = 1.;
    trial[1] = 1.;
    trial[2] = 0.;
    trial[3] = 0.;
    flip_index(trial, Gcon, Kcon);

    trial[0] = 0.;
    trial[1] = 0.;
    trial[2] = 1.;
    trial[3] = 0.;

    int sing = make_plasma_tetrad(Ucam, Kcon, trial, Gcov, Econ, Ecov);
#if DEBUG
    if(sing) {
        fprintf(stderr, "\nError making Camera tetrad, something is wrong!\n");
        fprintf(stderr, "Used the following vectors:\n");
        print_vector("X", X);
        print_vector("Ucam", Ucam);
        print_vector("Kcon", Kcon);
        print_vector("trial", trial);
        print_matrix("gcov", Gcov);
        print_matrix("gcon", Gcon);
    }
#endif
    return sing;
}

} // Cameras
