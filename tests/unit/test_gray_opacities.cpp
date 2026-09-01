// ======================================================================
// © 2021-2024. Triad National Security, LLC. All rights reserved.  This
// program was produced under U.S. Government contract
// 89233218CNA000001 for Los Alamos National Laboratory (LANL), which
// is operated by Triad National Security, LLC for the U.S.
// Department of Energy/National Nuclear Security Administration. All
// rights in the program are reserved by Triad National Security, LLC,
// and the U.S. Department of Energy/National Nuclear Security
// Administration. The Government is granted for itself and others
// acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
// license in this material to reproduce, prepare derivative works,
// distribute copies to the public, perform publicly and display
// publicly, and to permit others to do so.
// ======================================================================

// This file was generated in part with the assistance of generative AI.

#include <cmath>
#include <iostream>

#include <catch2/catch_test_macros.hpp>

#include <ports-of-call/portability.hpp>
#include <ports-of-call/portable_arrays.hpp>
#include <spiner/databox.hpp>

#include <singularity-opac/base/indexers.hpp>
#include <singularity-opac/base/radiation_types.hpp>
#include <singularity-opac/chebyshev/chebyshev.hpp>
#include <singularity-opac/constants/constants.hpp>
#include <singularity-opac/neutrinos/opac_neutrinos.hpp>
#include <singularity-opac/photons/opac_photons.hpp>

using namespace singularity;

using pc = PhysicalConstantsCGS;
using DataBox = Spiner::DataBox<Real>;

#ifdef PORTABILITY_STRATEGY_KOKKOS
using atomic_view = Kokkos::MemoryTraits<Kokkos::Atomic>;
#endif

template<typename T>
PORTABLE_INLINE_FUNCTION T FractionalDifference(const T& a, const T& b)
{
    return 2 * std::abs(b - a) / (std::abs(a) + std::abs(b) + 1e-20);
}
constexpr Real EPS_TEST = 1e-3;

TEST_CASE("Gray neutrino opacities", "[GrayNeutrinos]")
{
    WHEN("We initialize a gray neutrino opacity")
    {
        constexpr Real MeV2K = 1e6 * pc::eV / pc::kb;
        constexpr Real MeV2Hz = 1e6 * pc::eV / pc::h;
        constexpr Real rho = 1e11;        // g/cc
        constexpr Real temp = 10 * MeV2K; // 10 MeV
        constexpr Real Ye = 0.1;
        constexpr RadiationType type = RadiationType::NU_ELECTRON;
        constexpr Real nu = 1.25 * MeV2Hz; // 1 MeV

        neutrinos::Opacity opac_host = neutrinos::Gray(1.);
        neutrinos::Opacity opac = opac_host.GetOnDevice();

        // Check constants from opacity
        THEN("Check constants from mean opacity for consistency")
        {
            auto constants = opac_host.GetRuntimePhysicalConstants();
            int n_wrong = 0;
            if (FractionalDifference(pc::eV, constants.eV) > EPS_TEST) {
                n_wrong += 1;
            }
            if (FractionalDifference(pc::kb, constants.kb) > EPS_TEST) {
                n_wrong += 1;
            }
            if (FractionalDifference(pc::h, constants.h) > EPS_TEST) {
                n_wrong += 1;
            }
            REQUIRE(n_wrong == 0);
        }

        THEN("The emissivity per nu omega is consistent with the emissity per nu")
        {
            int n_wrong_h = 0;
#ifdef PORTABILITY_STRATEGY_KOKKOS
            Kokkos::View<int, atomic_view> n_wrong_d("wrong");
#else
            PortableMDArray<int> n_wrong_d(&n_wrong_h, 1);
#endif

            portableFor(
                "calc emissivities", 0, 100, PORTABLE_LAMBDA(const int& i) {
                    Real jnu = opac.EmissivityPerNuOmega(rho, temp, Ye, type, nu);
                    Real Jnu = opac.EmissivityPerNu(rho, temp, Ye, type, nu);
                    if (FractionalDifference(Jnu, 4 * M_PI * jnu) > EPS_TEST) {
                        n_wrong_d() += 1;
                    }
                });

#ifdef PORTABILITY_STRATEGY_KOKKOS
            Kokkos::deep_copy(n_wrong_h, n_wrong_d);
#endif
            REQUIRE(n_wrong_h == 0);
        }

        WHEN("We create a gray opacity object in non-cgs units")
        {
            constexpr Real time_unit = 123;
            constexpr Real mass_unit = 456;
            constexpr Real length_unit = 789;
            constexpr Real temp_unit = 276;
            constexpr Real rho_unit =
                mass_unit / (length_unit * length_unit * length_unit);
            constexpr Real j_unit = mass_unit / (length_unit * time_unit * time_unit);
            neutrinos::Opacity funny_units_host = neutrinos::NonCGSUnits<neutrinos::Gray>(
                neutrinos::Gray(1), time_unit, mass_unit, length_unit, temp_unit);
            auto funny_units = funny_units_host.GetOnDevice();

            THEN("We can retrieve physical constants in code units")
            {
                auto noncgs_rpc = funny_units.GetRuntimePhysicalConstants();
                REQUIRE(FractionalDifference(noncgs_rpc.length, length_unit) < EPS_TEST);
                REQUIRE(FractionalDifference(noncgs_rpc.time, time_unit) < EPS_TEST);
                REQUIRE(FractionalDifference(noncgs_rpc.mass, mass_unit) < EPS_TEST);
                REQUIRE(FractionalDifference(noncgs_rpc.temp, temp_unit) < EPS_TEST);
                REQUIRE(FractionalDifference(noncgs_rpc.na, 6.022141e+23) < EPS_TEST);
                REQUIRE(FractionalDifference(noncgs_rpc.alpha, 7.297353e-03) < EPS_TEST);
                REQUIRE(FractionalDifference(noncgs_rpc.h, 2.871060e-33) < EPS_TEST);
                REQUIRE(FractionalDifference(noncgs_rpc.hbar, 4.569434e-34) < EPS_TEST);
                REQUIRE(FractionalDifference(noncgs_rpc.kb, 2.030877e-18) < EPS_TEST);
                REQUIRE(FractionalDifference(noncgs_rpc.r_gas, 4.431243e+03) < EPS_TEST);
                REQUIRE(FractionalDifference(noncgs_rpc.qe, 4.803205e-10) < EPS_TEST);
                REQUIRE(FractionalDifference(noncgs_rpc.c, 4.673571e+09) < EPS_TEST);
                REQUIRE(FractionalDifference(noncgs_rpc.g_newt, 4.508065e-15) < EPS_TEST);
                REQUIRE(FractionalDifference(noncgs_rpc.me, 1.997672e-30) < EPS_TEST);
                REQUIRE(FractionalDifference(noncgs_rpc.mp, 3.668030e-27) < EPS_TEST);
                REQUIRE(FractionalDifference(noncgs_rpc.mn, 3.673086e-27) < EPS_TEST);
                REQUIRE(FractionalDifference(noncgs_rpc.amu, 3.641533e-27) < EPS_TEST);
                REQUIRE(FractionalDifference(noncgs_rpc.sb, 1.342760e+09) < EPS_TEST);
                REQUIRE(FractionalDifference(noncgs_rpc.ar, 1.149237e+00) < EPS_TEST);
                REQUIRE(FractionalDifference(noncgs_rpc.eV, 8.538896e-17) < EPS_TEST);
                REQUIRE(FractionalDifference(noncgs_rpc.Fc, 1.189435e-09) < EPS_TEST);
                REQUIRE(
                    FractionalDifference(noncgs_rpc.nu_sigma0, 2.829094e-74) < EPS_TEST);
                REQUIRE(FractionalDifference(noncgs_rpc.gA, -1.23) < EPS_TEST);
            }

            THEN("We can convert meaningfully into and out of funny units")
            {
                int n_wrong_h = 0;
#ifdef PORTABILITY_STRATEGY_KOKKOS
                Kokkos::View<int, atomic_view> n_wrong_d("wrong");
#else
                PortableMDArray<int> n_wrong_d(&n_wrong_h, 1);
#endif

                portableFor(
                    "emissivities in funny units", 0, 100, PORTABLE_LAMBDA(const int& i) {
                        Real jnu_funny = funny_units.EmissivityPerNuOmega(
                            rho / rho_unit, temp / temp_unit, Ye, type, nu * time_unit);
                        Real jnu = opac.EmissivityPerNuOmega(rho, temp, Ye, type, nu);
                        if (FractionalDifference(jnu, jnu_funny * j_unit) > EPS_TEST) {
                            n_wrong_d() += 1;
                        }
                    });
#ifdef PORTABILITY_STRATEGY_KOKKOS
                Kokkos::deep_copy(n_wrong_h, n_wrong_d);
#endif
                REQUIRE(n_wrong_h == 0);
            }
        }

        THEN("We can fill an indexer allocated in a cell")
        {
            int n_wrong_h = 0;
#ifdef PORTABILITY_STRATEGY_KOKKOS
            Kokkos::View<int, atomic_view> n_wrong_d("wrong");
#else
            PortableMDArray<int> n_wrong_d(&n_wrong_h, 1);
#endif
            constexpr int nbins = 9;
            constexpr int ntemps = 100;

            Real lnu_min = 0 + std::log10(MeV2Hz);
            Real lnu_max = 1 + std::log10(MeV2Hz);
            Real nu_min = std::pow(10, lnu_min);
            Real nu_max = std::pow(10, lnu_max);

            constexpr Real lt_min = -1;
            constexpr Real lt_max = 2;
            Real dt = (lt_max - lt_min) / (Real)(ntemps - 1);

            Real* nu_bins = (Real*)PORTABLE_MALLOC(nbins * sizeof(Real));
            Real* lnu_bins = (Real*)PORTABLE_MALLOC(nbins * sizeof(Real));
            Real* temp_bins = (Real*)PORTABLE_MALLOC(ntemps * sizeof(Real));
            portableFor(
                "set nu bins", 0, 1, PORTABLE_LAMBDA(const int& i) {
                    chebyshev::GetPoints(lnu_min, lnu_max, nbins, lnu_bins);
                    for (int j = 0; j < nbins; ++j) {
                        nu_bins[j] = std::pow(10, lnu_bins[j]);
                    }
                });
            portableFor(
                "set temp bins", 0, ntemps, PORTABLE_LAMBDA(const int& i) {
                    temp_bins[i] = std::pow(10, lt_min + dt * i) * MeV2K;
                });

            Real* vm9 = (Real*)PORTABLE_MALLOC(9 * 9 * sizeof(Real));
            portableFor(
                "Fill vm", 0, 1,
                PORTABLE_LAMBDA(const int& i) { chebyshev::get_vmbox(vm9); });

            portableFor(
                "Fill the indexers", 0, ntemps, PORTABLE_LAMBDA(const int& i) {
                    Real temp = temp_bins[i];

                    Real* nu_data = (Real*)malloc(nbins * sizeof(Real));
                    Real* lnu_data = (Real*)malloc(nbins * sizeof(Real));
                    Real* nu_coeffs = (Real*)malloc(nbins * sizeof(Real));
                    indexers::LogCheb<nbins, Real*> J_cheb(
                        nu_data, lnu_data, nu_coeffs, nu_min, nu_max);
                    opac.EmissivityPerNu(rho, temp, Ye, type, nu_bins, J_cheb, nbins);
                    Real Jtrue = opac.EmissivityPerNu(rho, temp, Ye, type, nu);
                    J_cheb.SetInterpCoeffs(DataBox(vm9, 9, 9));
                    if (std::isnan(J_cheb(nu)) ||
                        ((std::abs(Jtrue) >= 1e-14 || J_cheb(nu) >= 1e-14) &&
                            FractionalDifference(J_cheb(nu), Jtrue) > EPS_TEST)) {
                        n_wrong_d() += 1;
                    }
                    free(nu_data);
                    free(lnu_data);
                    free(nu_coeffs);
                });

#ifdef PORTABILITY_STRATEGY_KOKKOS
            Kokkos::deep_copy(n_wrong_h, n_wrong_d);
#endif

            REQUIRE(n_wrong_h == 0);

            PORTABLE_FREE(vm9);
            PORTABLE_FREE(nu_bins);
            PORTABLE_FREE(lnu_bins);
            PORTABLE_FREE(temp_bins);
        }

        opac.Finalize();
    }
}

TEST_CASE("Gray photon opacities", "[GrayPhotons]")
{
    WHEN("We initialize a gray photon opacity")
    {
        constexpr Real rho = 1e3;  // g/cc.
        constexpr Real temp = 1e5; // Kelvin.
        constexpr Real nu = 3e9;   // Hz. UHF microwave

        photons::Opacity opac_host = photons::Gray(1);
        photons::Opacity opac = opac_host.GetOnDevice();

        // Check constants from mean opacity
        THEN("Check constants from mean opacity for consistency")
        {
            auto constants = opac_host.GetRuntimePhysicalConstants();
            int n_wrong = 0;
            if (FractionalDifference(pc::eV, constants.eV) > EPS_TEST) {
                n_wrong += 1;
            }
            if (FractionalDifference(pc::kb, constants.kb) > EPS_TEST) {
                n_wrong += 1;
            }
            if (FractionalDifference(pc::h, constants.h) > EPS_TEST) {
                n_wrong += 1;
            }
            REQUIRE(n_wrong == 0);
        }

        THEN("The emissivity per nu omega is consistent with the emissity per nu")
        {
            int n_wrong_h = 0;
#ifdef PORTABILITY_STRATEGY_KOKKOS
            Kokkos::View<int, atomic_view> n_wrong_d("wrong");
#else
            PortableMDArray<int> n_wrong_d(&n_wrong_h, 1);
#endif
            portableFor(
                "calc emissivities", 0, 100, PORTABLE_LAMBDA(const int& i) {
                    Real jnu = opac.EmissivityPerNuOmega(rho, temp, nu);
                    Real Jnu = opac.EmissivityPerNu(rho, temp, nu);
                    if (FractionalDifference(Jnu, 4 * M_PI * jnu) > EPS_TEST) {
                        n_wrong_d() += 1;
                    }
                });
#ifdef PORTABILITY_STRATEGY_KOKKOS
            Kokkos::deep_copy(n_wrong_h, n_wrong_d);
#endif
            REQUIRE(n_wrong_h == 0);
        }

        WHEN("We create a gray opacity object in non-cgs units")
        {
            constexpr Real time_unit = 123;
            constexpr Real mass_unit = 456;
            constexpr Real length_unit = 789;
            constexpr Real temp_unit = 276;
            constexpr Real rho_unit =
                mass_unit / (length_unit * length_unit * length_unit);
            constexpr Real j_unit = mass_unit / (length_unit * time_unit * time_unit);
            photons::Opacity funny_units_host = photons::NonCGSUnits<photons::Gray>(
                photons::Gray(1), time_unit, mass_unit, length_unit, temp_unit);
            auto funny_units = funny_units_host.GetOnDevice();

            THEN("We can retrieve physical constants in code units")
            {
                auto noncgs_rpc = funny_units.GetRuntimePhysicalConstants();
                REQUIRE(FractionalDifference(noncgs_rpc.length, length_unit) < EPS_TEST);
                REQUIRE(FractionalDifference(noncgs_rpc.time, time_unit) < EPS_TEST);
                REQUIRE(FractionalDifference(noncgs_rpc.mass, mass_unit) < EPS_TEST);
                REQUIRE(FractionalDifference(noncgs_rpc.temp, temp_unit) < EPS_TEST);
                REQUIRE(FractionalDifference(noncgs_rpc.na, 6.022141e+23) < EPS_TEST);
                REQUIRE(FractionalDifference(noncgs_rpc.alpha, 7.297353e-03) < EPS_TEST);
                REQUIRE(FractionalDifference(noncgs_rpc.h, 2.871060e-33) < EPS_TEST);
                REQUIRE(FractionalDifference(noncgs_rpc.hbar, 4.569434e-34) < EPS_TEST);
                REQUIRE(FractionalDifference(noncgs_rpc.kb, 2.030877e-18) < EPS_TEST);
                REQUIRE(FractionalDifference(noncgs_rpc.r_gas, 4.431243e+03) < EPS_TEST);
                REQUIRE(FractionalDifference(noncgs_rpc.qe, 4.803205e-10) < EPS_TEST);
                REQUIRE(FractionalDifference(noncgs_rpc.c, 4.673571e+09) < EPS_TEST);
                REQUIRE(FractionalDifference(noncgs_rpc.g_newt, 4.508065e-15) < EPS_TEST);
                REQUIRE(FractionalDifference(noncgs_rpc.me, 1.997672e-30) < EPS_TEST);
                REQUIRE(FractionalDifference(noncgs_rpc.mp, 3.668030e-27) < EPS_TEST);
                REQUIRE(FractionalDifference(noncgs_rpc.mn, 3.673086e-27) < EPS_TEST);
                REQUIRE(FractionalDifference(noncgs_rpc.amu, 3.641533e-27) < EPS_TEST);
                REQUIRE(FractionalDifference(noncgs_rpc.sb, 1.342760e+09) < EPS_TEST);
                REQUIRE(FractionalDifference(noncgs_rpc.ar, 1.149237e+00) < EPS_TEST);
                REQUIRE(FractionalDifference(noncgs_rpc.eV, 8.538896e-17) < EPS_TEST);
                REQUIRE(FractionalDifference(noncgs_rpc.Fc, 1.189435e-09) < EPS_TEST);
                REQUIRE(
                    FractionalDifference(noncgs_rpc.nu_sigma0, 2.829094e-74) < EPS_TEST);
                REQUIRE(FractionalDifference(noncgs_rpc.gA, -1.23) < EPS_TEST);
            }

            THEN("We can convert meaningfully into and out of funny units")
            {
                int n_wrong_h = 0;
#ifdef PORTABILITY_STRATEGY_KOKKOS
                Kokkos::View<int, atomic_view> n_wrong_d("wrong");
#else
                PortableMDArray<int> n_wrong_d(&n_wrong_h, 1);
#endif

                portableFor(
                    "emissivities in funny units", 0, 100, PORTABLE_LAMBDA(const int& i) {
                        Real jnu_funny = funny_units.EmissivityPerNuOmega(
                            rho / rho_unit, temp / temp_unit, nu * time_unit);
                        Real jnu = opac.EmissivityPerNuOmega(rho, temp, nu);
                        if (FractionalDifference(jnu, jnu_funny * j_unit) > EPS_TEST) {
                            n_wrong_d() += 1;
                        }
                    });
#ifdef PORTABILITY_STRATEGY_KOKKOS
                Kokkos::deep_copy(n_wrong_h, n_wrong_d);
#endif
                REQUIRE(n_wrong_h == 0);
            }
        }

        THEN("We can fill an indexer allocated in a cell")
        {
            int n_wrong_h = 0;
#ifdef PORTABILITY_STRATEGY_KOKKOS
            Kokkos::View<int, atomic_view> n_wrong_d("wrong");
#else
            PortableMDArray<int> n_wrong_d(&n_wrong_h, 1);
#endif
            constexpr int nbins = 100;
            constexpr int ntemps = 100;

            constexpr Real lnu_min = 8;
            constexpr Real lnu_max = 10;
            Real nu_min = std::pow(10, lnu_min);
            Real nu_max = std::pow(10, lnu_max);
            Real dnu = (lnu_max - lnu_min) / (Real)(nbins - 1);

            constexpr Real lt_min = 2;
            constexpr Real lt_max = 4;
            Real dt = (lt_max - lt_min) / (Real)(ntemps - 1);

            Real* nu_bins = (Real*)PORTABLE_MALLOC(nbins * sizeof(Real));
            Real* temp_bins = (Real*)PORTABLE_MALLOC(ntemps * sizeof(Real));
            DataBox loglin_bins(Spiner::AllocationTarget::Device, ntemps, nbins);

            portableFor(
                "set nu bins", 0, nbins, PORTABLE_LAMBDA(const int& i) {
                    nu_bins[i] = std::pow(10, lnu_min + dnu * i);
                });
            portableFor(
                "set temp bins", 0, ntemps, PORTABLE_LAMBDA(const int& i) {
                    temp_bins[i] = std::pow(10, lt_min + dt * i);
                });

            portableFor(
                "Fill the indexers", 0, ntemps, PORTABLE_LAMBDA(const int& i) {
                    Real temp = temp_bins[i];
                    auto bins = loglin_bins.slice(i);
                    indexers::LogLinear J_log(bins, nu_min, nu_max, nbins);
                    opac.EmissivityPerNu(rho, temp, nu_bins, J_log, nbins);
                    Real Jtrue = opac.EmissivityPerNu(rho, temp, nu);
                    if (FractionalDifference(Jtrue, J_log(nu)) > EPS_TEST) {
                        n_wrong_d() += 1;
                    }
                });

#ifdef PORTABILITY_STRATEGY_KOKKOS
            Kokkos::deep_copy(n_wrong_h, n_wrong_d);
#endif
            REQUIRE(n_wrong_h == 0);

            PORTABLE_FREE(nu_bins);
            PORTABLE_FREE(temp_bins);
            free(loglin_bins);
        }

        opac.Finalize();
    }
}

TEST_CASE("Tophat Opacities", "[TopHat]")
{
    WHEN("We initialize a tophat neutrino opacity")
    {
        neutrinos::Opacity opac = neutrinos::Tophat(1, 1e-2, 1e2);
    }
}
