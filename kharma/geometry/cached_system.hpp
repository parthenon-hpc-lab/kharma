// © 2021. Triad National Security, LLC. All rights reserved.  This
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

#ifndef GEOMETRY_CACHED_SYSTEM_HPP_
#define GEOMETRY_CACHED_SYSTEM_HPP_

// C++ includes
#include <array>
#include <cmath>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

// Parthenon includes
#include <coordinates/coordinates.hpp>
#include <globals.hpp>
#include <kokkos_abstraction.hpp>
#include <parthenon/package.hpp>
#include <utils/error_checking.hpp>

// phoebus includes
#include "geometry/geometry_defaults.hpp"
#include "geometry/geometry_utils.hpp"
#include "phoebus_utils/cell_locations.hpp"
#include "phoebus_utils/phoebus_interpolation.hpp"
#include "phoebus_utils/robust.hpp"

using namespace parthenon::package::prelude;

namespace Geometry
{

// A geometry type that "caches" an analytic coordinate system on the
// grid. Quantities such as the metric, connection coefficients, etc.,
// are stored at cell centers and cell faces. If a request is made for
// a node or for off-grid, the original analytic system is used.
const std::vector<std::string> GEOMETRY_CACHED_VARS = {"g.c.alpha", "g.c.dalpha",
    "g.c.bcon", "g.c.gcov", "g.c.gamcon", "g.c.detgam", "g.c.dg", "g.c.coord",

    "g.f1.alpha", "g.f1.bcon", "g.f1.gcov", "g.f1.gamcon", "g.f1.detgam",

    "g.f2.alpha", "g.f2.bcon", "g.f2.gcov", "g.f2.gamcon", "g.f2.detgam",

    "g.f3.alpha", "g.f3.bcon", "g.f3.gcov", "g.f3.gamcon", "g.f3.detgam",

    "g.n.coord"};
const std::vector<std::string> GEOMETRY_LOC_NAMES = {"c", "f1", "f2", "f3"};

// Support class. Accesses a pack on cells or faces as appropriate
namespace Impl
{
struct GeomPackIndices
{
    int alpha, dalpha, bcon, gcov, gamcon, detgam, dg;
};
template<typename T>
class LocArray
{
  public:
    KOKKOS_INLINE_FUNCTION
    T& operator[](int i) { return a_[i]; }
    KOKKOS_INLINE_FUNCTION
    T& operator[](CellLocation loc) { return (*this)[icast_(loc)]; }
    KOKKOS_INLINE_FUNCTION
    const T& operator[](int i) const { return a_[i]; }
    KOKKOS_INLINE_FUNCTION
    const T& operator[](CellLocation loc) const { return (*this)[icast_(loc)]; }

  private:
    KOKKOS_INLINE_FUNCTION
    int icast_(CellLocation loc) const { return static_cast<int>(loc); }
    T a_[NUM_CELL_LOCATIONS];
    // std::array<T, NUM_CELL_LOCATIONS> a_;
};
} // namespace Impl

// TODO(JMM): Store corners when available if needed
template<typename Pack, typename System>
class Cached
{
  public:
    Cached() = default;
    template<typename Data>
    Cached(
        Data* rc, const System& s, bool axisymmetric = false, bool time_dependent = false)
        : s_(s)
        , axisymmetric_(axisymmetric)
        , time_dependent_(time_dependent)
        , nvar_deriv_(time_dependent ? NDFULL : NDSPACE)
    {
        PackIndexMap imap;
        pack_ = rc->PackVariables(GEOMETRY_CACHED_VARS, imap);
        int i = 0;
        // These variables exist everywhere
        for (auto& loc : GEOMETRY_LOC_NAMES) {
            PARTHENON_DEBUG_REQUIRE(
                imap["g." + loc + ".alpha"].second >= 0, "variable exists");
            PARTHENON_DEBUG_REQUIRE(
                imap["g." + loc + ".alpha"].second - imap["g." + loc + ".alpha"].first ==
                    0,
                "variable shape correct");
            PARTHENON_DEBUG_REQUIRE(
                imap["g." + loc + ".bcon"].second >= 0, "variable exists");
            PARTHENON_DEBUG_REQUIRE(imap["g." + loc + ".bcon"].second -
                                            imap["g." + loc + ".bcon"].first + 1 ==
                                        3,
                "variable shape correct");
            PARTHENON_DEBUG_REQUIRE(
                imap["g." + loc + ".gcov"].second >= 0, "variable exists");
            PARTHENON_DEBUG_REQUIRE(
                imap["g." + loc + ".gamcon"].second >= 0, "variable exists");
            PARTHENON_DEBUG_REQUIRE(
                imap["g." + loc + ".detgam"].second >= 0, "variable exists");
            idx_[i].alpha = imap["g." + loc + ".alpha"].first;
            idx_[i].bcon = imap["g." + loc + ".bcon"].first;
            idx_[i].gcov = imap["g." + loc + ".gcov"].first;
            idx_[i].gamcon = imap["g." + loc + ".gamcon"].first;
            idx_[i].detgam = imap["g." + loc + ".detgam"].first;
            i++;
        }
        // These variables only exist at cell centers
        PARTHENON_DEBUG_REQUIRE(imap["g.c.dalpha"].second >= 0, "variable exists");
        PARTHENON_DEBUG_REQUIRE(
            imap["g.c.dalpha"].second - imap["g.c.dalpha"].first + 1 == nvar_deriv_,
            "variable shape correct");
        idx_[CellLocation::Cent].dalpha = imap["g.c.dalpha"].first;
        PARTHENON_DEBUG_REQUIRE(imap["g.c.dg"].second >= 0, "variable exists");
        idx_[CellLocation::Cent].dg = imap["g.c.dg"].first;
        icoord_c_ = imap["g.c.coord"].first;
        // This variable only exists at cell corners
        icoord_n_ = imap["g.n.coord"].first;
    }
    KOKKOS_INLINE_FUNCTION
    Real Lapse(int b, Real X0, Real X1, Real X2, Real X3) const
    {
        return LCInterp::Do(b, X1, X2, X3, pack_, idx_[cent_].alpha);
    }
    KOKKOS_INLINE_FUNCTION
    Real Lapse(Real X0, Real X1, Real X2, Real X3) const
    {
        return Lapse(0, X0, X1, X2, X3);
    }
    KOKKOS_INLINE_FUNCTION
    Real Lapse(CellLocation loc, int b, int k, int j, int i) const
    {
        PARTHENON_DEBUG_REQUIRE(
            loc != CellLocation::Corn, "Node-centered values not available.");
        k *= (!axisymmetric_); // maps k -> 0 for axisymmetric spacetimes. Otherwise no-op
        return pack_(b, idx_[loc].alpha, k, j, i);
    }
    KOKKOS_INLINE_FUNCTION
    Real Lapse(CellLocation loc, int k, int j, int i) const
    {
        return Lapse(loc, b_, k, j, i);
    }
    KOKKOS_INLINE_FUNCTION
    void ContravariantShift(
        int b, Real X0, Real X1, Real X2, Real X3, Real beta[NDSPACE]) const
    {
        SPACELOOP(d)
        {
            beta[d] = LCInterp::Do(b, X1, X2, X3, pack_, idx_[cent_].bcon + d);
        }
    }
    KOKKOS_INLINE_FUNCTION
    void ContravariantShift(Real X0, Real X1, Real X2, Real X3, Real beta[NDSPACE]) const
    {
        return ContravariantShift(0, X0, X1, X2, X3, beta);
    }
    KOKKOS_INLINE_FUNCTION
    void ContravariantShift(
        CellLocation loc, int b, int k, int j, int i, Real beta[NDSPACE]) const
    {
        PARTHENON_DEBUG_REQUIRE(
            loc != CellLocation::Corn, "Node-centered values not available.");
        k *= (!axisymmetric_); // maps k -> 0 for axisymmetric spacetimes. Otherwise no-op
        SPACELOOP(d) { beta[d] = pack_(b, idx_[loc].bcon + d, k, j, i); }
    }
    KOKKOS_INLINE_FUNCTION
    void ContravariantShift(
        CellLocation loc, int k, int j, int i, Real beta[NDSPACE]) const
    {
        ContravariantShift(loc, b_, k, j, i, beta);
    }
    KOKKOS_INLINE_FUNCTION
    void Metric(
        int b, Real X0, Real X1, Real X2, Real X3, Real gamma[NDSPACE][NDSPACE]) const
    {
        SPACELOOP2(m, n)
        {
            int offst = Utils::Flatten2(m + 1, n + 1, NDFULL); // gamma_{ij} = g_{ij}
            gamma[m][n] = LCInterp::Do(b, X1, X2, X3, pack_, idx_[cent_].gcov + offst);
        }
    }
    KOKKOS_INLINE_FUNCTION
    void Metric(Real X0, Real X1, Real X2, Real X3, Real gamma[NDSPACE][NDSPACE]) const
    {
        return Metric(0, X0, X1, X2, X3, gamma);
    }
    KOKKOS_INLINE_FUNCTION
    void Metric(
        CellLocation loc, int b, int k, int j, int i, Real gamma[NDSPACE][NDSPACE]) const
    {
        PARTHENON_DEBUG_REQUIRE(
            loc != CellLocation::Corn, "Node-centered values not available.");
        k *= (!axisymmetric_); // maps k -> 0 for axisymmetric spacetimes. Otherwise no-op
        SPACELOOP2(m, n)
        { // gamma_{ij} = g_{ij}
            int offst = Utils::Flatten2(m + 1, n + 1, NDFULL);
            gamma[m][n] = pack_(b, idx_[loc].gcov + offst, k, j, i);
        }
    }
    KOKKOS_INLINE_FUNCTION
    void Metric(CellLocation loc, int k, int j, int i, Real gamma[NDSPACE][NDSPACE]) const
    {
        Metric(loc, b_, k, j, i, gamma);
    }
    KOKKOS_INLINE_FUNCTION
    void MetricInverse(
        int b, Real X0, Real X1, Real X2, Real X3, Real gamma[NDSPACE][NDSPACE]) const
    {
        SPACELOOP2(m, n)
        {
            int offst = Utils::Flatten2(m, n, NDSPACE);
            gamma[m][n] = LCInterp::Do(b, X1, X2, X3, pack_, idx_[cent_].gamcon + offst);
        }
    }
    KOKKOS_INLINE_FUNCTION
    void MetricInverse(
        Real X0, Real X1, Real X2, Real X3, Real gamma[NDSPACE][NDSPACE]) const
    {
        return MetricInverse(0, X0, X1, X2, X3, gamma);
    }
    KOKKOS_INLINE_FUNCTION
    void MetricInverse(
        CellLocation loc, int b, int k, int j, int i, Real gamma[NDSPACE][NDSPACE]) const
    {
        PARTHENON_DEBUG_REQUIRE(
            loc != CellLocation::Corn, "Node-centered values not available.");
        k *= (!axisymmetric_); // maps k -> 0 for axisymmetric spacetimes. Otherwise no-op
        SPACELOOP2(m, n)
        {
            int offst = Utils::Flatten2(m, n, NDSPACE);
            gamma[m][n] = pack_(b, idx_[loc].gamcon + offst, k, j, i);
        }
    }
    KOKKOS_INLINE_FUNCTION
    void MetricInverse(
        CellLocation loc, int k, int j, int i, Real gamma[NDSPACE][NDSPACE]) const
    {
        MetricInverse(loc, b_, k, j, i, gamma);
    }
    KOKKOS_INLINE_FUNCTION
    void SpacetimeMetric(
        int b, Real X0, Real X1, Real X2, Real X3, Real g[NDFULL][NDFULL]) const
    {
        SPACETIMELOOP2(mu, nu)
        {
            int offst = Utils::Flatten2(mu, nu, NDFULL);
            g[mu][nu] = LCInterp::Do(b, X1, X2, X3, pack_, idx_[cent_].gcov + offst);
        }
    }
    KOKKOS_INLINE_FUNCTION
    void SpacetimeMetric(Real X0, Real X1, Real X2, Real X3, Real g[NDFULL][NDFULL]) const
    {
        return SpacetimeMetric(0, X0, X1, X2, X3, g);
    }
    KOKKOS_INLINE_FUNCTION
    void SpacetimeMetric(
        CellLocation loc, int b, int k, int j, int i, Real g[NDFULL][NDFULL]) const
    {
        PARTHENON_DEBUG_REQUIRE(
            loc != CellLocation::Corn, "Node-centered values not available.");
        k *= (!axisymmetric_); // maps k -> 0 for axisymmetric spacetimes. Otherwise no-op
        SPACETIMELOOP2(mu, nu)
        {
            int offst = Utils::Flatten2(mu, nu, NDFULL);
            g[mu][nu] = pack_(b, idx_[loc].gcov + offst, k, j, i);
        }
    }
    KOKKOS_INLINE_FUNCTION
    void SpacetimeMetric(
        CellLocation loc, int k, int j, int i, Real g[NDFULL][NDFULL]) const
    {
        SpacetimeMetric(loc, b_, k, j, i, g);
    }
    KOKKOS_INLINE_FUNCTION
    void SpacetimeMetricInverse(
        int b, Real X0, Real X1, Real X2, Real X3, Real g[NDFULL][NDFULL]) const
    {
        using robust::ratio;
        auto& idx = idx_[cent_];
        Real alpha2 = Lapse(b, X1, X2, X3);
        alpha2 *= alpha2;
        g[0][0] = -robust::ratio(1, alpha2);
        SPACELOOP(mu)
        {
            Real num = LCInterp::Do(b, X1, X2, X3, pack_, idx.bcon + mu);
            g[mu + 1][0] = g[0][mu + 1] = ratio(num, alpha2);
        }
        SPACELOOP2(mu, nu)
        {
            int offst = Utils::Flatten2(mu, nu, NDSPACE);
            g[mu + 1][nu + 1] = LCInterp::Do(b, X1, X2, X3, pack_, idx.gamcon + offst);
            g[mu + 1][nu + 1] -= g[0][mu + 1] * g[0][nu + 1] * alpha2;
        }
    }
    KOKKOS_INLINE_FUNCTION
    void SpacetimeMetricInverse(
        Real X0, Real X1, Real X2, Real X3, Real g[NDFULL][NDFULL]) const
    {
        return SpacetimeMetricInverse(b_, X0, X1, X2, X3, g);
    }
    KOKKOS_INLINE_FUNCTION
    void SpacetimeMetricInverse(
        CellLocation loc, int b, int k, int j, int i, Real g[NDFULL][NDFULL]) const
    {
        using robust::ratio;
        PARTHENON_DEBUG_REQUIRE(
            loc != CellLocation::Corn, "Node-centered values not available.");
        k *= (!axisymmetric_); // maps k -> 0 for axisymmetric spacetimes. Otherwise no-op
        auto& idx = idx_[loc];
        Real alpha2 = Lapse(loc, b, k, j, i);
        alpha2 *= alpha2;
        g[0][0] = -ratio(1, alpha2);
        SPACELOOP(mu)
        {
            g[mu + 1][0] = g[0][mu + 1] = ratio(pack_(b, idx.bcon + mu, k, j, i), alpha2);
        }
        SPACELOOP2(mu, nu)
        {
            int offst = Utils::Flatten2(mu, nu, NDSPACE);
            g[mu + 1][nu + 1] = pack_(b, idx.gamcon + offst, k, j, i);
            g[mu + 1][nu + 1] -= g[0][mu + 1] * g[0][nu + 1] * alpha2;
        }
    }
    KOKKOS_INLINE_FUNCTION
    void SpacetimeMetricInverse(
        CellLocation loc, int k, int j, int i, Real g[NDFULL][NDFULL]) const
    {
        SpacetimeMetricInverse(loc, b_, k, j, i, g);
    }
    KOKKOS_INLINE_FUNCTION
    Real DetGamma(int b, Real X0, Real X1, Real X2, Real X3) const
    {
        return LCInterp::Do(b, X1, X2, X3, pack_, idx_[cent_].detgam);
    }
    KOKKOS_INLINE_FUNCTION
    Real DetGamma(Real X0, Real X1, Real X2, Real X3) const
    {
        return DetGamma(0, X0, X1, X2, X3);
    }
    KOKKOS_INLINE_FUNCTION
    Real DetGamma(CellLocation loc, int b, int k, int j, int i) const
    {
        PARTHENON_DEBUG_REQUIRE(
            loc != CellLocation::Corn, "Node-centered values not available.");
        k *= (!axisymmetric_); // maps k -> 0 for axisymmetric spacetimes. Otherwise no-op
        return pack_(b, idx_[loc].detgam, k, j, i);
    }
    KOKKOS_INLINE_FUNCTION
    Real DetGamma(CellLocation loc, int k, int j, int i) const
    {
        return DetGamma(loc, b_, k, j, i);
    }
    template<typename... Args>
    KOKKOS_INLINE_FUNCTION Real DetG(Args... args) const
    {
        return Lapse(std::forward<Args>(args)...) * DetGamma(std::forward<Args>(args)...);
    }
    KOKKOS_INLINE_FUNCTION
    void ConnectionCoefficient(int b, Real X0, Real X1, Real X2, Real X3,
        Real Gamma[NDFULL][NDFULL][NDFULL]) const
    {
        Real dg[NDFULL][NDFULL][NDFULL];
        MetricDerivative(b, X0, X1, X2, X3, dg);
        Utils::SetConnectionCoeffFromMetricDerivs(dg, Gamma);
    }
    KOKKOS_INLINE_FUNCTION
    void ConnectionCoefficient(
        Real X0, Real X1, Real X2, Real X3, Real Gamma[NDFULL][NDFULL][NDFULL]) const
    {
        return ConnectionCoefficient(0, X0, X1, X2, X3, Gamma);
    }
    KOKKOS_INLINE_FUNCTION
    void ConnectionCoefficient(
        CellLocation loc, int k, int j, int i, Real Gamma[NDFULL][NDFULL][NDFULL]) const
    {
        PARTHENON_DEBUG_REQUIRE(
            loc != CellLocation::Corn, "Node-centered values not available.");
        k *= (!axisymmetric_); // maps k -> 0 for axisymmetric spacetimes. Otherwise no-op
        Utils::SetConnectionCoeffByFD(*this, Gamma, loc, k, j, i);
    }
    KOKKOS_INLINE_FUNCTION
    void ConnectionCoefficient(CellLocation loc, int bid, int k, int j, int i,
        Real Gamma[NDFULL][NDFULL][NDFULL]) const
    {
        PARTHENON_DEBUG_REQUIRE(
            loc != CellLocation::Corn, "Node-centered values not available.");
        k *= (!axisymmetric_); // maps k -> 0 for axisymmetric spacetimes. Otherwise no-op
        Utils::SetConnectionCoeffByFD(*this, Gamma, loc, bid, k, j, i);
    }
    KOKKOS_INLINE_FUNCTION
    void MetricDerivative(
        int b, Real X0, Real X1, Real X2, Real X3, Real dg[NDFULL][NDFULL][NDFULL]) const
    {
        const int offset = !time_dependent_;
        SPACETIMELOOP2(mu, nu)
        {
            const int flat =
                nvar_deriv_ * Utils::Flatten2(mu, nu, NDFULL) + idx_[cent_].dg - offset;
            dg[mu][nu][0] = 0.0; // gets overwritten if time-dependent metric
            for (int sigma = offset; sigma < NDFULL; sigma++) {
                dg[mu][nu][sigma] = LCInterp::Do(b, X1, X2, X3, pack_, flat + sigma);
            }
        }
    }
    KOKKOS_INLINE_FUNCTION
    void MetricDerivative(
        Real X0, Real X1, Real X2, Real X3, Real dg[NDFULL][NDFULL][NDFULL]) const
    {
        return MetricDerivative(0, X0, X1, X2, X3, dg);
    }
    KOKKOS_INLINE_FUNCTION
    void MetricDerivative(CellLocation loc, int b, int k, int j, int i,
        Real dg[NDFULL][NDFULL][NDFULL]) const
    {
        PARTHENON_DEBUG_REQUIRE(
            loc != CellLocation::Corn, "Node-centered values not available.");
        k *= (!axisymmetric_); // maps k -> 0 for axisymmetric spacetimes. Otherwise no-op
        const int offset = !time_dependent_;
        SPACETIMELOOP2(mu, nu)
        {
            const int flat =
                nvar_deriv_ * Utils::Flatten2(mu, nu, NDFULL) + idx_[loc].dg - offset;
            dg[mu][nu][0] = 0.0; // gets overwritten if time-dependent metric
            for (int sigma = offset; sigma < NDFULL; sigma++) {
                dg[mu][nu][sigma] = pack_(b, flat + sigma, k, j, i);
            }
        }
    }
    KOKKOS_INLINE_FUNCTION
    void MetricDerivative(
        CellLocation loc, int k, int j, int i, Real dg[NDFULL][NDFULL][NDFULL]) const
    {
        MetricDerivative(loc, b_, k, j, i, dg);
    }
    KOKKOS_INLINE_FUNCTION
    void GradLnAlpha(int b, Real X0, Real X1, Real X2, Real X3, Real da[NDFULL]) const
    {
        const int offset = !time_dependent_;
        da[0] = 0; // gets overwritten if time-dependent metric
        for (int d = offset; d < NDFULL; ++d) {
            da[d] = LCInterp::Do(b, X1, X2, X3, pack_, idx_[cent_].dalpha + d - offset);
        }
    }
    KOKKOS_INLINE_FUNCTION
    void GradLnAlpha(Real X0, Real X1, Real X2, Real X3, Real da[NDFULL]) const
    {
        return GradLnAlpha(0, X0, X1, X2, X3, da);
    }
    KOKKOS_INLINE_FUNCTION
    void GradLnAlpha(CellLocation loc, int b, int k, int j, int i, Real da[NDFULL]) const
    {
        PARTHENON_DEBUG_REQUIRE(
            loc != CellLocation::Corn, "Node-centered values not available.");
        k *= (!axisymmetric_); // maps k -> 0 for axisymmetric spacetimes. Otherwise no-op
        const int offset = !time_dependent_;
        da[0] = 0; // gets overwritten if time-dependent metric
        for (int d = offset; d < NDFULL; ++d) {
            da[d] = pack_(b, idx_[loc].dalpha + d - offset, k, j, i);
        }
    }
    KOKKOS_INLINE_FUNCTION
    void GradLnAlpha(CellLocation loc, int k, int j, int i, Real da[NDFULL]) const
    {
        return GradLnAlpha(loc, b_, k, j, i, da);
    }
    KOKKOS_INLINE_FUNCTION
    void Coords(int b, Real X0, Real X1, Real X2, Real X3, Real C[NDFULL]) const
    {
        C[0] = X0_;
        SPACELOOP(d)
        { // coords never axisymmetric
            C[d + 1] = LCInterp::Do(b, X1, X2, X3, pack_, icoord_c_ + d);
        }
    }
    KOKKOS_INLINE_FUNCTION
    void Coords(Real X0, Real X1, Real X2, Real X3, Real C[NDFULL]) const
    {
        return Coords(0, X0, X1, X2, X3, C);
    }
    KOKKOS_INLINE_FUNCTION
    void Coords(CellLocation loc, int b, int k, int j, int i, Real C[NDFULL]) const
    {
        if ((loc == CellLocation::Cent) || (loc == CellLocation::Corn)) {
            C[0] = X0_; // TODO(JMM): Should these be updated for time-dep metrics?
            int icoord = (loc == CellLocation::Cent) ? icoord_c_ : icoord_n_;
            SPACELOOP(d) { C[d + 1] = pack_(b, icoord + d, k, j, i); }
        } else {
            s_.Coords(loc, b, k, j, i, C);
        }
    }
    KOKKOS_INLINE_FUNCTION
    void Coords(CellLocation loc, int k, int j, int i, Real C[NDFULL]) const
    {
        if ((loc == CellLocation::Cent) || (loc == CellLocation::Corn)) {
            C[0] = X0_; // TODO(JMM): Should these be updated for time-dep metrics?
            int icoord = (loc == CellLocation::Cent) ? icoord_c_ : icoord_n_;
            SPACELOOP(d) { C[d + 1] = pack_(b_, icoord + d, k, j, i); }
        } else {
            s_.Coords(loc, k, j, i, C);
        }
    }

  private:
    bool axisymmetric_ = false;
    bool time_dependent_ = false;
    int nvar_deriv_ = 3;
    System s_;
    int icoord_c_;
    int icoord_n_;
    Pack pack_;
    Impl::LocArray<Impl::GeomPackIndices> idx_;
    static constexpr Real X0_ = 0;
    static constexpr int b_ = 0;
    static constexpr CellLocation cent_ = CellLocation::Cent;
};

template<typename System>
using CachedOverMeshBlock = Cached<parthenon::VariablePack<Real>, System>;
template<typename System>
using CachedOverMesh = Cached<parthenon::MeshBlockVarPack<Real>, System>;

template<typename System>
void InitializeCachedCoordinateSystem(ParameterInput* pin, StateDescriptor* geometry)
{
    using parthenon::MetadataFlag;

    // Initialize underlying system
    Initialize<System>(pin, geometry);

    // Get params
    Params& params = geometry->AllParams();

    // Geometries which are time-dependent and wish to hook in to the
    // caching machinery are expected to set a parameter, "time_dependent",
    // in their params during initialization.
    // If it's available it will be read out and returned here.
    // Otherwise it is set to false.
    const bool time_dependent = params.Get("time_dependent", false);
    const int nvar_deriv = time_dependent ? NDFULL : NDSPACE;

    const bool axisymmetric = pin->GetOrAddBoolean("geometry", "axisymmetric", false);
    params.Add("axisymmetric", axisymmetric);

    // This tells the goemetry infrastructure not to call
    // SetGeometryDefault, because the cached geometry will handle it.
    const bool do_defaults = false;
    params.Add("do_defaults", do_defaults);

    // Optionally use finite differences on the grid to set the metric
    // derivatives. If the geometry we're caching specifies one way or
    // the other, we defer to the geometry we're caching, and
    // warn. Otherwise, we defer to the input deck.
    bool do_fd_on_grid;
    bool do_fd_on_grid_pin = pin->GetOrAddBoolean("geometry", "do_fd_on_grid", false);
    if (params.hasKey("do_fd_on_grid")) {
        do_fd_on_grid = params.Get<bool>("do_fd_on_grid");
        if ((do_fd_on_grid != do_fd_on_grid_pin) && parthenon::Globals::my_rank == 0) {
            std::stringstream ss;
            ss << "do_fd_on_grid set by input deck as " << do_fd_on_grid_pin
               << " but set by the geometry class to be " << do_fd_on_grid
               << ". The geometry class takes priority." << std::endl;
            PARTHENON_WARN(ss);
        }
    } else {
        do_fd_on_grid = do_fd_on_grid_pin;
        params.Add("do_fd_on_grid", do_fd_on_grid);
    }

    // Request fields for cache
    Utils::MeshBlockShape dims(pin);

    std::vector<int> shape;
    Metadata m;
    std::vector<MetadataFlag> flags_c = {
        Metadata::Cell, Metadata::Derived, Metadata::OneCopy};
    std::vector<MetadataFlag> flags_o = {Metadata::Derived, Metadata::OneCopy};

    // JMM: Coords arrays are always provided by the geometry machinery,
    // for use with 3D vis software. No special logic in caching
    // required.
    std::vector<int> var_sizes = {
        1, NDSPACE, Utils::SymSize<NDFULL>(), Utils::SymSize<NDSPACE>(), 1};
    std::vector<std::string> var_names = {"alpha", "bcon", "gcov", "gamcon", "detgam"};
    std::vector<int> cent_only_sizes = {
        nvar_deriv, nvar_deriv * Utils::SymSize<NDFULL>()};
    std::vector<std::string> cent_only_names = {"dalpha", "dg"};
    PARTHENON_REQUIRE_THROWS(
        var_sizes.size() == var_names.size(), "Same number of variables as sizes");
    PARTHENON_REQUIRE_THROWS(cent_only_sizes.size() == cent_only_names.size(),
        "Same number of variables as sizes");
    // Cell variables
    for (int i = 0; i < var_sizes.size(); ++i) {
        if (axisymmetric) {
            shape = {dims.nx1, dims.nx2, 1, var_sizes[i]};
            m = Metadata(flags_o, shape);
        } else {
            shape = {var_sizes[i]};
            m = Metadata(flags_c, shape);
        }
        geometry->AddField("g.c." + var_names[i], m);
    }
    // Do metric derivative and lapse derivative separately, as
    // they're only available on cell centers
    for (int i = 0; i < cent_only_sizes.size(); ++i) {
        if (axisymmetric) {
            shape = {dims.nx1, dims.nx2, 1, cent_only_sizes[i]};
            m = Metadata(flags_o, shape);
        } else {
            shape = {cent_only_sizes[i]};
            m = Metadata(flags_c, shape);
        }
        geometry->AddField("g.c." + cent_only_names[i], m);
    }
    // face variables
    for (int d = 1; d <= 3; ++d) {
        for (int i = 0; i < var_sizes.size(); ++i) {
            shape = {dims.nx1, dims.nx2, dims.nx3, var_sizes[i]};
            shape[d - 1] += 1;
            if (axisymmetric) shape[2] = 1;
            m = Metadata(flags_o, shape);
            geometry->AddField("g." + GEOMETRY_LOC_NAMES[d] + "." + var_names[i], m);
        }
    }
}

template<typename System>
CachedOverMeshBlock<System> GetCachedCoordinateSystem(MeshBlockData<Real>* rc)
{
    auto system = GetCoordinateSystem<System>(rc);
    auto& pkg = rc->GetMeshPointer()->packages.Get("geometry");
    bool axisymmetric = pkg->Param<bool>("axisymmetric");
    bool time_dependent = pkg->Param<bool>("time_dependent");
    return CachedOverMeshBlock<System>(rc, system, axisymmetric, time_dependent);
}
template<typename System>
CachedOverMesh<System> GetCachedCoordinateSystem(MeshData<Real>* rc)
{
    auto system = GetCoordinateSystem<System>(rc);
    auto& pkg = rc->GetMeshPointer()->packages.Get("geometry");
    bool axisymmetric = pkg->Param<bool>("axisymmetric");
    bool time_dependent = pkg->Param<bool>("time_dependent");
    return CachedOverMesh<System>(rc, system, axisymmetric, time_dependent);
}

template<typename System, typename Data>
void SetCachedCoordinateSystem(Data* rc)
{
    Mesh* pparent = rc->GetMeshPointer();
    auto system = GetCoordinateSystem<System>(rc);

    auto& pkg = pparent->packages.Get("geometry");
    parthenon::Params& params = pkg->AllParams();
    bool axisymmetric = params.Get<bool>("axisymmetric");
    bool time_dependent = params.Get<bool>("time_dependent");
    bool do_fd_on_grid = params.Get<bool>("do_fd_on_grid");
    const int nvar_deriv = time_dependent ? NDFULL : NDSPACE;

    Impl::LocArray<Impl::GeomPackIndices> idx;
    PackIndexMap imap;
    auto pack = rc->PackVariables(GEOMETRY_CACHED_VARS, imap);
    int i = 0;
    for (auto& loc : GEOMETRY_LOC_NAMES) {
        PARTHENON_DEBUG_REQUIRE(
            imap["g." + loc + ".alpha"].second >= 0, "Variable exists");
        PARTHENON_DEBUG_REQUIRE(
            imap["g." + loc + ".bcon"].second >= 0, "Variable exists");
        PARTHENON_DEBUG_REQUIRE(
            imap["g." + loc + ".gcov"].second >= 0, "Variable exists");
        PARTHENON_DEBUG_REQUIRE(
            imap["g." + loc + ".gamcon"].second >= 0, "Variable exists");
        PARTHENON_DEBUG_REQUIRE(
            imap["g." + loc + ".detgam"].second >= 0, "Variable exists");
        idx[i].alpha = imap["g." + loc + ".alpha"].first;
        idx[i].bcon = imap["g." + loc + ".bcon"].first;
        idx[i].gcov = imap["g." + loc + ".gcov"].first;
        idx[i].gamcon = imap["g." + loc + ".gamcon"].first;
        idx[i].detgam = imap["g." + loc + ".detgam"].first;
        i++;
    }
    PARTHENON_DEBUG_REQUIRE(imap["g.c.dalpha"].second >= 0, "Variable exists");
    PARTHENON_DEBUG_REQUIRE(imap["g.c.dg"].second >= 0, "Variable exists");
    idx[CellLocation::Cent].dg = imap["g.c.dg"].first;
    idx[CellLocation::Cent].dalpha = imap["g.c.dalpha"].first;

    auto lamb = KOKKOS_LAMBDA(
        const int b, const int k, const int j, const int i, const CellLocation loc)
    {
        pack(b, idx[loc].alpha, k, j, i) = system.Lapse(loc, b, k, j, i);
        pack(b, idx[loc].detgam, k, j, i) = system.DetGamma(loc, b, k, j, i);

        Real beta[NDSPACE];
        system.ContravariantShift(loc, b, k, j, i, beta);
        SPACELOOP(d) { pack(b, idx[loc].bcon + d, k, j, i) = beta[d]; }

        Real gcov[NDFULL][NDFULL];
        system.SpacetimeMetric(loc, b, k, j, i, gcov);
        for (int mu = 0; mu < NDFULL; ++mu) {
            for (int nu = mu; nu < NDFULL; ++nu) {
                int offst = idx[loc].gcov + Utils::Flatten2(mu, nu, NDFULL);
                pack(b, offst, k, j, i) = gcov[mu][nu];
            }
        }

        Real gamcon[NDSPACE][NDSPACE];
        system.MetricInverse(loc, b, k, j, i, gamcon);
        for (int mu = 0; mu < NDSPACE; ++mu) {
            for (int nu = mu; nu < NDSPACE; ++nu) {
                int offst = idx[loc].gamcon + Utils::Flatten2(mu, nu, NDSPACE);
                pack(b, offst, k, j, i) = gamcon[mu][nu];
            }
        }
    };

    // Derivatives
    auto lamb_derivs = KOKKOS_LAMBDA(const int b, const int k, const int j, const int i)
    {
        auto loc = CellLocation::Cent;
        {
            int offset = !time_dependent;
            Real da[NDFULL];
            system.GradLnAlpha(loc, b, k, j, i, da);
            for (int d = offset; d < NDFULL; ++d) {
                pack(b, idx[loc].dalpha + d - offset, k, j, i) = da[d];
            }
        }
        Real dg[NDFULL][NDFULL][NDFULL];
        system.MetricDerivative(loc, b, k, j, i, dg);
        {
            int offset = !time_dependent;
            for (int sigma = offset; sigma < NDFULL; ++sigma) {
                for (int mu = 0; mu < NDFULL; ++mu) {
                    for (int nu = mu; nu < NDFULL; ++nu) {
                        int flat = idx[loc].dg +
                                   nvar_deriv * Utils::Flatten2(mu, nu, NDFULL) + sigma -
                                   offset;
                        pack(b, flat, k, j, i) = dg[mu][nu][sigma];
                    }
                }
            }
        }
    };
    // Compute derivatives by finite differences
    // If time-derivatives present, use this to overwrite non time-derivative calls
    // If time-derivatives not present, just do this.
    // Strategy is to do second-order finite differences by diffing cell
    // faces to compute cell centered values.
    auto lamb_derivs_fd =
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i)
    {
        const auto loc = CellLocation::Cent;
        const auto& coords = pack.GetCoords(b);
        const Real alpha = system.Lapse(loc, b, k, j, i);
        Real alpha_fp, alpha_fm;
        Real g_fp[NDFULL][NDFULL];
        Real g_fm[NDFULL][NDFULL];
        const CellLocation floc[] = {
            CellLocation::Face1, CellLocation::Face2, CellLocation::Face3};
        SPACELOOP(d)
        {
            int idx_fp[] = {k, j, i};
            idx_fp[NDSPACE - 1 - d] += 1;
            alpha_fp = system.Lapse(floc[d], b, idx_fp[0], idx_fp[1], idx_fp[2]);
            alpha_fm = system.Lapse(floc[d], b, k, j, i);
            system.SpacetimeMetric(floc[d], b, idx_fp[0], idx_fp[1], idx_fp[2], g_fp);
            system.SpacetimeMetric(floc[d], b, k, j, i, g_fm);
            // Recall that we're setting GradLnAlpha, not GradAlpha;
            const Real da = robust::ratio(
                alpha_fp - alpha_fm, alpha * coords.FaceArea(d + 1, k, j, i));
            // if time_dependent, then we don't want to fill the d/dt, mu = 0 terms
            pack(b, idx[loc].dalpha + d + time_dependent, k, j, i) = da;
            // 2d loop for metric
            const Real sigma = d + time_dependent;
            for (int mu = 0; mu < NDFULL; ++mu) {
                for (int nu = mu; nu < NDFULL; ++nu) {
                    // if time_dependent, then we don't want to fill the d/dt, sigma = 0
                    // terms
                    const int flat = idx[loc].dg +
                                     nvar_deriv * Utils::Flatten2(mu, nu, NDFULL) + sigma;
                    const Real dg = robust::ratio(
                        g_fp[mu][nu] - g_fm[mu][nu], coords.FaceArea(d + 1, k, j, i));
                    pack(b, flat, k, j, i) = dg;
                }
            }
        }
    };

    IndexRange ib = rc->GetBoundsI(IndexDomain::entire);
    IndexRange jb = rc->GetBoundsJ(IndexDomain::entire);
    IndexRange kb = rc->GetBoundsK(IndexDomain::entire);
    int kbs = axisymmetric ? 0 : kb.s;
    int kbe = axisymmetric ? 0 : kb.e;
    parthenon::par_for(DEFAULT_LOOP_PATTERN, "SetGeometry::Set Cached data, Cent",
        DevExecSpace(), 0, pack.GetDim(5) - 1, kbs, kbe, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i)
        {
            lamb(b, k, j, i, CellLocation::Cent);
        });
    // Skip if do_fd_on_grid and not time_dependent, since these can be
    // all set by FD.
    if (time_dependent || !do_fd_on_grid) {
        parthenon::par_for(DEFAULT_LOOP_PATTERN, "SetGeometry::Set Cached data, derivs",
            DevExecSpace(), 0, pack.GetDim(5) - 1, kbs, kbe, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i)
            {
                lamb_derivs(b, k, j, i);
            });
    }
    parthenon::par_for(DEFAULT_LOOP_PATTERN, "SetGeometry::Set Cached data, Face1",
        DevExecSpace(), 0, pack.GetDim(5) - 1, kbs, kbe, jb.s, jb.e, ib.s, ib.e + 1,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i)
        {
            lamb(b, k, j, i, CellLocation::Face1);
        });
    parthenon::par_for(DEFAULT_LOOP_PATTERN, "SetGeometry::Set Cached data, Face2",
        DevExecSpace(), 0, pack.GetDim(5) - 1, kbs, kbe, jb.s, jb.e + 1, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i)
        {
            lamb(b, k, j, i, CellLocation::Face2);
        });
    if (!axisymmetric) kbe = kb.e + 1;
    parthenon::par_for(DEFAULT_LOOP_PATTERN, "SetGeometry::Set Cached data, Face3",
        DevExecSpace(), 0, pack.GetDim(5) - 1, kbs, kbe, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i)
        {
            lamb(b, k, j, i, CellLocation::Face3);
        });
    if (do_fd_on_grid) {
        kbe = axisymmetric ? 0 : kb.e;
        parthenon::par_for(DEFAULT_LOOP_PATTERN,
            "SetGeometry::Set Cached data, derivs via FD", DevExecSpace(), 0,
            pack.GetDim(5) - 1, kbs, kbe, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i)
            {
                lamb_derivs_fd(b, k, j, i);
            });
    }
    // Call SetGeometryDefault to set the coords objects.
    impl::SetGeometryDefault(rc, system);
}

} // namespace Geometry

#endif // GEOMETRY_CACHED_SYSTEM_HPP_
