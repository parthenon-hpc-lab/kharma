/*
 *  File: coordinate_embedding.hpp
 *
 *  BSD 3-Clause License
 *
 *  Copyright (c) 2020, AFD Group at UIUC
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *
 *  1. Redistributions of source code must retain the above copyright notice, this
 *     list of conditions and the following disclaimer.
 *
 *  2. Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 *
 *  3. Neither the name of the copyright holder nor the names of its
 *     contributors may be used to endorse or promote products derived from
 *     this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 *  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 *  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 *  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 *  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#pragma once

#include "decs.hpp"

#include "coordinate_systems.hpp"
#include "coordinate_utils.hpp"
#include "matrix.hpp"

// std::variant requires C++ exceptions,
// so it will never be SYCL-ready.
// Instead we use mpark's reimplementation,
// patched to never throw exceptions.
// Because who needs those?
// TODO(CEP) try to switch to std:: unless using SYCL
#include <ports-of-call/variant.hpp>
// #include <variant>
// namespace mpark = std;

/**
 * Coordinates in HARM are logically Cartesian -- that is, in some coordinate system, here
 * dubbed "native" coordinates, each cell is a rectangular prism of exactly the same shape
 * as all the others. However, working in GR allows us to define that "native" coordinate
 * system arbitrarily in relation to the "base" or "embedding" coordinates, which are
 * usually Spherical Kerr-Schild coordinates.
 *
 * That is, as long as we have a bijective map of base<->transformed coordinates, we can
 * define the latter arbitrarily, which is great for putting resolution where we need and
 * not where we don't.
 *
 * This class keeps track of the base coordinates and the map, defining generic functions
 * guaranteed to return something in the native or embedding coordinate system, given
 * something else -- e.g. covariant metric "gcov" at a native coordinate location Xnative.
 *
 * Each system or transform must be a class defining a basic interface:
 * see coordinate_systems.hpp for the current examples.
 *
 * Specifically, the BaseCoords class must implement:
 * * gcov_embed
 * And the Transform class must implement:
 * * coord_to_embed
 * * coord_to_native
 * * dxdX_to_embed
 * * dxdX_to_native
 *
 * Each possible class is added to a couple of PortsOfCall::variant containers, and then to the
 * chains of if statements below.
 *
 * TODO convenience functions.  Intelligent r/th/phi, x/y/z, KS and BL, a, etc by
 * auto-translating contents
 */
class CoordinateEmbedding
{
  public:
    SomeBaseCoords base;
    SomeTransform transform;

    // Common code for constructors
#pragma hd_warning_disable
    KOKKOS_FUNCTION void EmplaceSystems(
        const SomeBaseCoords& base_in, const SomeTransform& transform_in)
    {
        // Isn't there some more elegant way to say "yeah the types are fine just copy da
        // bits"?
        if (PortsOfCall::holds_alternative<SphMinkowskiCoords>(base_in)) {
            base.emplace<SphMinkowskiCoords>(PortsOfCall::get<SphMinkowskiCoords>(base_in));
        } else if (PortsOfCall::holds_alternative<CartMinkowskiCoords>(base_in)) {
            base.emplace<CartMinkowskiCoords>(PortsOfCall::get<CartMinkowskiCoords>(base_in));
        } else if (PortsOfCall::holds_alternative<SphBLCoords>(base_in)) {
            base.emplace<SphBLCoords>(PortsOfCall::get<SphBLCoords>(base_in));
        } else if (PortsOfCall::holds_alternative<SphKSCoords>(base_in)) {
            base.emplace<SphKSCoords>(PortsOfCall::get<SphKSCoords>(base_in));
        } else if (PortsOfCall::holds_alternative<SphKSExtG>(base_in)) {
            base.emplace<SphKSExtG>(PortsOfCall::get<SphKSExtG>(base_in));
        } else if (PortsOfCall::holds_alternative<SphBLExtG>(base_in)) {
            base.emplace<SphBLExtG>(PortsOfCall::get<SphBLExtG>(base_in));
        }

        if (PortsOfCall::holds_alternative<NullTransform>(transform_in)) {
            transform.emplace<NullTransform>(PortsOfCall::get<NullTransform>(transform_in));
        } else if (PortsOfCall::holds_alternative<ExponentialTransform>(transform_in)) {
            transform.emplace<ExponentialTransform>(
                PortsOfCall::get<ExponentialTransform>(transform_in));
        } else if (PortsOfCall::holds_alternative<SuperExponentialTransform>(transform_in)) {
            transform.emplace<SuperExponentialTransform>(
                PortsOfCall::get<SuperExponentialTransform>(transform_in));
        } else if (PortsOfCall::holds_alternative<ModifyTransform>(transform_in)) {
            transform.emplace<ModifyTransform>(PortsOfCall::get<ModifyTransform>(transform_in));
        } else if (PortsOfCall::holds_alternative<FunkyTransform>(transform_in)) {
            transform.emplace<FunkyTransform>(PortsOfCall::get<FunkyTransform>(transform_in));
        } else if (PortsOfCall::holds_alternative<WidepoleTransform>(transform_in)) {
            transform.emplace<WidepoleTransform>(
                PortsOfCall::get<WidepoleTransform>(transform_in));
        }
    }

    // Constructors
#pragma hd_warning_disable
    CoordinateEmbedding() = default;
#pragma hd_warning_disable
    CoordinateEmbedding(parthenon::ParameterInput* pin)
    {
        const std::string base_str = pin->GetString("coordinates", "base");
        const std::string transform_str =
            pin->GetOrAddString("coordinates", "transform", "null");

        // Parse names.  See coordinate_systems.hpp for details
        if (base_str == "spherical_minkowski") {
            base.emplace<SphMinkowskiCoords>(SphMinkowskiCoords());
        } else if (base_str == "cartesian_minkowski" || base_str == "minkowski") {
            base.emplace<CartMinkowskiCoords>(CartMinkowskiCoords());
        } else if (base_str == "spherical_ks" || base_str == "ks" ||
                   base_str == "spherical_ks_extg" || base_str == "ks_extg") {
            GReal a = pin->GetReal("coordinates", "a");
            bool ext_g = pin->GetOrAddBoolean("coordinates", "ext_g", false);
            if (ext_g || base_str == "spherical_ks_extg" || base_str == "ks_extg") {
                if (a > 0)
                    throw std::invalid_argument(
                        "Transform is for spherical coordinates!");
                base.emplace<SphKSExtG>(SphKSExtG(a));
            } else {
                base.emplace<SphKSCoords>(SphKSCoords(a));
            }
        } else if (base_str == "spherical_bl" || base_str == "bl" ||
                   base_str == "spherical_bl_extg" || base_str == "bl_extg") {
            GReal a = pin->GetReal("coordinates", "a");
            bool ext_g = pin->GetOrAddBoolean("coordinates", "ext_g", false);
            if (ext_g || base_str == "spherical_bl_extg" || base_str == "bl_extg") {
                if (a > 0)
                    throw std::invalid_argument(
                        "Transform is for spherical coordinates!");
                base.emplace<SphBLExtG>(SphBLExtG(a));
            } else {
                base.emplace<SphBLCoords>(SphBLCoords(a));
            }
        } else {
            throw std::invalid_argument("Unsupported base coordinates!");
        }

        bool spherical = is_spherical();

        if (transform_str == "null" || transform_str == "none") {
            if (spherical) {
                transform.emplace<SphNullTransform>(SphNullTransform());
            } else {
                transform.emplace<NullTransform>(NullTransform());
            }
        } else if (transform_str == "exponential" || transform_str == "exp" ||
                   transform_str == "eks") {
            if (!spherical)
                throw std::invalid_argument("Transform is for spherical coordinates!");
            transform.emplace<ExponentialTransform>(ExponentialTransform());
        } else if (transform_str == "superexponential" || transform_str == "superexp") {
            if (!spherical)
                throw std::invalid_argument("Transform is for spherical coordinates!");
            GReal r_br = pin->GetOrAddReal("coordinates", "r_br", 1000.);
            GReal npow = pin->GetOrAddReal("coordinates", "npow", 1.0);
            GReal cpow = pin->GetOrAddReal("coordinates", "cpow", 4.0);
            transform.emplace<SuperExponentialTransform>(
                SuperExponentialTransform(r_br, npow, cpow));
        } else if (transform_str == "modified" || transform_str == "mks") {
            if (!spherical)
                throw std::invalid_argument("Transform is for spherical coordinates!");
            GReal hslope = pin->GetOrAddReal("coordinates", "hslope", 0.3);
            transform.emplace<ModifyTransform>(ModifyTransform(hslope));
        } else if (transform_str == "funky" || transform_str == "fmks") {
            if (!spherical)
                throw std::invalid_argument("Transform is for spherical coordinates!");
            GReal hslope = pin->GetOrAddReal("coordinates", "hslope", 0.3);
            GReal mks_smooth = pin->GetOrAddReal("coordinates", "mks_smooth", 0.5);
            GReal poly_xt = pin->GetOrAddReal("coordinates", "poly_xt", 0.82);
            GReal poly_alpha = pin->GetOrAddReal("coordinates", "poly_alpha", 14.0);
            // Set fmks to use x1min from our system for compatibility. Note THIS WILL
            // CHANGE
            GReal startx1 = 0.; // Default for temporary coordinate construction before
                                // mesh, future general default
            if (pin->DoesParameterExist("coordinates", "fmks_zero_point")) {
                startx1 = pin->GetReal("coordinates", "fmks_zero_point");
            } else if (pin->DoesParameterExist("parthenon/mesh", "x1min")) {
                std::cout << "KHARMA WARNING: Constructing FMKS coordinates using mesh "
                             "x1min is deprecated."
                          << std::endl
                          << "Set coordinates/fmks_zero_point for consistent behavior."
                          << std::endl;
                startx1 = pin->GetReal("parthenon/mesh", "x1min");
            }
            transform.emplace<FunkyTransform>(
                FunkyTransform(startx1, hslope, mks_smooth, poly_xt, poly_alpha));
        } else if (transform_str == "widepole" || transform_str == "wks") {
            if (!spherical)
                throw std::invalid_argument("Transform is for spherical coordinates!");
            GReal lin_frac = pin->GetOrAddReal("coordinates", "lin_frac", 0.6);
            GReal smoothness = pin->GetOrAddReal("coordinates", "smoothness", -1.0);
            GReal nx2 = pin->GetReal("parthenon/mesh", "nx2");
            GReal nx3 = pin->GetReal("parthenon/mesh", "nx3");
            transform.emplace<WidepoleTransform>(
                WidepoleTransform(lin_frac, smoothness, nx2, nx3));
        } else {
            throw std::invalid_argument("Unsupported coordinate transform!");
        }
    }
#pragma hd_warning_disable
    KOKKOS_FUNCTION CoordinateEmbedding(
        SomeBaseCoords& base_in, SomeTransform& transform_in)
        : base(base_in)
        , transform(transform_in)
    {}
#pragma hd_warning_disable
    KOKKOS_FUNCTION CoordinateEmbedding(const CoordinateEmbedding& src)
        : base(src.base)
        , transform(src.transform)
    {}
#pragma hd_warning_disable
    KOKKOS_FUNCTION const CoordinateEmbedding& operator=(const CoordinateEmbedding& src)
    {
        // CoordinateEmbedding copy(src);
        // base.swap(copy.base);
        // transform.swap(copy.transform);
        EmplaceSystems(src.base, src.transform);
        return *this;
    }
    // Convenience functions to get common things:
    // Names (host only)
#pragma hd_warning_disable
    KOKKOS_INLINE_FUNCTION std::string variant_names() const
    {
        std::string basename(PortsOfCall::visit(
            [&](const auto& self)
            {
                return self.name;
            },
            base));

        std::string transformname(PortsOfCall::visit(
            [&](const auto& self)
            {
                return self.name;
            },
            transform));

        return basename + " " + transformname;
    }

    // Properties (host or device)
    KOKKOS_INLINE_FUNCTION bool is_spherical() const
    {
        return PortsOfCall::visit(
            [&](const auto& self)
            {
                return self.spherical;
            },
            base);
    }
    KOKKOS_INLINE_FUNCTION bool is_transformed() const
    {
        return !PortsOfCall::holds_alternative<NullTransform>(transform);
    }
    KOKKOS_INLINE_FUNCTION GReal get_horizon() const
    {
        if (PortsOfCall::holds_alternative<SphKSCoords>(base) ||
            PortsOfCall::holds_alternative<SphBLCoords>(base) ||
            PortsOfCall::holds_alternative<SphKSExtG>(base) ||
            PortsOfCall::holds_alternative<SphBLExtG>(base)) {
            const GReal a = get_a();
            return 1 + m::sqrt(1 - a * a);
        } else {
            return 0.0;
        }
    }
    KOKKOS_INLINE_FUNCTION GReal get_a() const
    {
        return PortsOfCall::visit(
            [&](const auto& self)
            {
                return self.a;
            },
            base);
    }
    GReal startx(int dir) const
    {
        return PortsOfCall::visit(
            [&](const auto& self)
            {
                return self.startx[dir - 1];
            },
            transform);
    }
    GReal stopx(int dir) const
    {
        return PortsOfCall::visit(
            [&](const auto& self)
            {
                return self.stopx[dir - 1];
            },
            transform);
    }

    KOKKOS_INLINE_FUNCTION bool is_ks() const
    {
        return PortsOfCall::holds_alternative<SphKSCoords>(base);
    }
    KOKKOS_INLINE_FUNCTION bool is_cart_minkowski() const
    {
        return PortsOfCall::holds_alternative<CartMinkowskiCoords>(base) &&
               PortsOfCall::holds_alternative<NullTransform>(transform);
    }

    // Note this is the one thing we need from BaseCoords
    KOKKOS_INLINE_FUNCTION void gcov_embed(
        const GReal Xembed[GR_DIM], Real gcov[GR_DIM][GR_DIM]) const
    {
        PortsOfCall::visit(
            [&Xembed, &gcov](const auto& self)
            {
                self.gcov_embed(Xembed, gcov);
            },
            base);
    }
    // All the quantities we can derive from that
    KOKKOS_INLINE_FUNCTION Real gcon_from_gcov(
        const Real gcov[GR_DIM][GR_DIM], Real gcon[GR_DIM][GR_DIM]) const
    {
        Real gdet = invert(&gcov[0][0], &gcon[0][0]);
        return m::sqrt(m::abs(gdet));
    }
    KOKKOS_INLINE_FUNCTION Real gcon_embed(
        const GReal Xembed[GR_DIM], Real gcon[GR_DIM][GR_DIM]) const
    {
        GReal gcov[GR_DIM][GR_DIM];
        gcov_embed(Xembed, gcov);
        return gcon_from_gcov(gcov, gcon);
    }
    KOKKOS_INLINE_FUNCTION Real gdet_embed(const GReal Xembed[GR_DIM]) const
    {
        GReal gcon[GR_DIM][GR_DIM];
        return gcon_embed(Xembed, gcon);
    }

    // Now, everything we take from CoordinateTransform
    KOKKOS_INLINE_FUNCTION void coord_to_embed(
        const GReal Xnative[GR_DIM], GReal Xembed[GR_DIM]) const
    {
        PortsOfCall::visit(
            [&Xnative, &Xembed](const auto& self)
            {
                self.coord_to_embed(Xnative, Xembed);
            },
            transform);
    }
    KOKKOS_INLINE_FUNCTION void coord_to_native(
        const GReal Xembed[GR_DIM], GReal Xnative[GR_DIM]) const
    {
        PortsOfCall::visit(
            [&Xnative, &Xembed](const auto& self)
            {
                self.coord_to_native(Xembed, Xnative);
            },
            transform);
    }
    KOKKOS_INLINE_FUNCTION void dxdX(
        const GReal Xnative[GR_DIM], Real dxdX[GR_DIM][GR_DIM]) const
    {
        PortsOfCall::visit(
            [&Xnative, &dxdX](const auto& self)
            {
                self.dxdX(Xnative, dxdX);
            },
            transform);
    }
    KOKKOS_INLINE_FUNCTION void dXdx(
        const GReal Xnative[GR_DIM], Real dXdx[GR_DIM][GR_DIM]) const
    {
        PortsOfCall::visit(
            [&Xnative, &dXdx](const auto& self)
            {
                self.dXdx(Xnative, dXdx);
            },
            transform);
    }

    // Coordinate convenience functions:
    // transform the radial coordinate alone without len-4 arrays
    // ...at least not for the user.  These are not fast
    KOKKOS_INLINE_FUNCTION GReal r_to_native(const GReal r) const
    {
        const GReal Xembed[GR_DIM] = {0., r, 0., 0.};
        GReal Xnative[GR_DIM];
        PortsOfCall::visit(
            [&Xembed, &Xnative](const auto& self)
            {
                self.coord_to_native(Xembed, Xnative);
            },
            transform);
        return Xnative[1];
    }
    KOKKOS_INLINE_FUNCTION GReal X1_to_embed(const GReal X1) const
    {
        const GReal Xnative[GR_DIM] = {0., X1, 0., 0.};
        GReal Xembed[GR_DIM];
        PortsOfCall::visit(
            [&Xnative, &Xembed](const auto& self)
            {
                self.coord_to_embed(Xnative, Xembed);
            },
            transform);
        return Xembed[1];
    }

    // Get a particular coordinate from an array
    // note these *aren't faster* or less memory, just convenient
    KOKKOS_INLINE_FUNCTION GReal r_of(const GReal Xnative[GR_DIM]) const
    {
        GReal Xembed[GR_DIM];
        PortsOfCall::visit(
            [&Xnative, &Xembed](const auto& self)
            {
                self.coord_to_embed(Xnative, Xembed);
            },
            transform);
        if (is_spherical()) {
            return Xembed[1];
        } else {
            return m::sqrt(SQR(Xembed[1]) + SQR(Xembed[2]) + SQR(Xembed[3]));
        }
    }
    KOKKOS_INLINE_FUNCTION GReal th_of(const GReal Xnative[GR_DIM]) const
    {
        GReal Xembed[GR_DIM];
        PortsOfCall::visit(
            [&Xnative, &Xembed](const auto& self)
            {
                self.coord_to_embed(Xnative, Xembed);
            },
            transform);
        if (is_spherical()) {
            return Xembed[2];
        } else {
            return m::atan2(m::sqrt(SQR(Xembed[1]) + SQR(Xembed[2])), Xembed[3]);
        }
    }
    KOKKOS_INLINE_FUNCTION GReal phi_of(const GReal Xnative[GR_DIM]) const
    {
        GReal Xembed[GR_DIM];
        PortsOfCall::visit(
            [&Xnative, &Xembed](const auto& self)
            {
                self.coord_to_embed(Xnative, Xembed);
            },
            transform);
        if (is_spherical()) {
            return Xembed[3];
        } else {
            return m::atan2(Xembed[2], Xembed[1]);
        }
    }
    KOKKOS_INLINE_FUNCTION GReal x_of(const GReal Xnative[GR_DIM]) const
    {
        GReal Xembed[GR_DIM];
        PortsOfCall::visit(
            [&Xnative, &Xembed](const auto& self)
            {
                self.coord_to_embed(Xnative, Xembed);
            },
            transform);
        if (!is_spherical()) {
            return Xembed[1];
        } else {
            return Xembed[1] * m::sin(Xembed[2]) * m::cos(Xembed[3]);
        }
    }
    KOKKOS_INLINE_FUNCTION GReal y_of(const GReal Xnative[GR_DIM]) const
    {
        GReal Xembed[GR_DIM];
        PortsOfCall::visit(
            [&Xnative, &Xembed](const auto& self)
            {
                self.coord_to_embed(Xnative, Xembed);
            },
            transform);
        if (!is_spherical()) {
            return Xembed[2];
        } else {
            return Xembed[1] * m::sin(Xembed[2]) * m::sin(Xembed[3]);
        }
    }
    KOKKOS_INLINE_FUNCTION GReal z_of(const GReal Xnative[GR_DIM]) const
    {
        GReal Xembed[GR_DIM];
        PortsOfCall::visit(
            [&Xnative, &Xembed](const auto& self)
            {
                self.coord_to_embed(Xnative, Xembed);
            },
            transform);
        if (!is_spherical()) {
            return Xembed[3];
        } else {
            return Xembed[1] * m::cos(Xembed[2]);
        }
    }

    // VECTOR TRANSFORMS
    // Contravariant vectors:
    KOKKOS_INLINE_FUNCTION void con_vec_to_embed(const GReal Xnative[GR_DIM],
        const GReal vcon_native[GR_DIM], GReal vcon_embed[GR_DIM]) const
    {
        Real dxdX_temp[GR_DIM][GR_DIM];
        dxdX(Xnative, dxdX_temp);
        DLOOP1 {
            vcon_embed[mu] = 0;
            for (int nu = 0; nu < GR_DIM; ++nu)
                vcon_embed[mu] += dxdX_temp[mu][nu] * vcon_native[nu];
        }
    }
    KOKKOS_INLINE_FUNCTION void con_vec_to_native(const GReal Xnative[GR_DIM],
        const GReal vcon_embed[GR_DIM], GReal vcon_native[GR_DIM]) const
    {
        Real dXdx_temp[GR_DIM][GR_DIM];
        dXdx(Xnative, dXdx_temp);
        DLOOP1 { // TODO is this faster, or DLOOP1/2?
            vcon_native[mu] = 0;
            for (int nu = 0; nu < GR_DIM; ++nu)
                vcon_native[mu] += dXdx_temp[mu][nu] * vcon_embed[nu];
        }
    }
    // Covariant are opposite
    KOKKOS_INLINE_FUNCTION void cov_vec_to_native(const GReal Xnative[GR_DIM],
        const GReal vcov_embed[GR_DIM], GReal vcov_native[GR_DIM]) const
    {
        con_vec_to_embed(Xnative, vcov_embed, vcov_native);
    }
    KOKKOS_INLINE_FUNCTION void cov_vec_to_embed(const GReal Xnative[GR_DIM],
        const GReal vcov_native[GR_DIM], GReal vcov_embed[GR_DIM]) const
    {
        con_vec_to_native(Xnative, vcov_native, vcov_embed);
    }

    // TENSOR TRANSFORMS
    // Covariant first
    KOKKOS_INLINE_FUNCTION void cov_tensor_to_embed(const GReal Xnative[GR_DIM],
        const GReal tcov_native[GR_DIM][GR_DIM], GReal tcov_embed[GR_DIM][GR_DIM]) const
    {
        Real dXdx_temp[GR_DIM][GR_DIM];
        dXdx(Xnative, dXdx_temp);

        DLOOP2 {
            tcov_embed[mu][nu] = 0;
            for (int lam = 0; lam < GR_DIM; ++lam) {
                for (int kap = 0; kap < GR_DIM; ++kap) {
                    tcov_embed[mu][nu] +=
                        tcov_native[lam][kap] * dXdx_temp[lam][mu] * dXdx_temp[kap][nu];
                }
            }
        }
    }
    KOKKOS_INLINE_FUNCTION void cov_tensor_to_native(const GReal Xnative[GR_DIM],
        const GReal tcov_embed[GR_DIM][GR_DIM], GReal tcov_native[GR_DIM][GR_DIM]) const
    {
        Real dxdX_temp[GR_DIM][GR_DIM];
        dxdX(Xnative, dxdX_temp);

        DLOOP2 {
            tcov_native[mu][nu] = 0;
            for (int lam = 0; lam < GR_DIM; lam++) {
                for (int kap = 0; kap < GR_DIM; kap++) {
                    tcov_native[mu][nu] +=
                        tcov_embed[lam][kap] * dxdX_temp[lam][mu] * dxdX_temp[kap][nu];
                }
            }
        }
    }
    // Con are opposite
    KOKKOS_INLINE_FUNCTION void con_tensor_to_embed(const GReal Xnative[GR_DIM],
        const GReal tcon_native[GR_DIM][GR_DIM], GReal tcon_embed[GR_DIM][GR_DIM]) const
    {
        cov_tensor_to_native(Xnative, tcon_native, tcon_embed);
    }
    KOKKOS_INLINE_FUNCTION void con_tensor_to_native(const GReal Xnative[GR_DIM],
        const GReal tcon_embed[GR_DIM][GR_DIM], GReal tcon_native[GR_DIM][GR_DIM]) const
    {
        cov_tensor_to_embed(Xnative, tcon_embed, tcon_native);
    }

    // And then derived metric properties
    KOKKOS_INLINE_FUNCTION void gcov_native(
        const GReal Xnative[GR_DIM], Real gcov[GR_DIM][GR_DIM]) const
    {
        Real gcov_em[GR_DIM][GR_DIM];
        GReal Xembed[GR_DIM];
        // Get coordinates in embedding system
        coord_to_embed(Xnative, Xembed);

        // Get metric in embedding coordinates
        gcov_embed(Xembed, gcov_em);

        // Transform to native coordinates
        cov_tensor_to_native(Xnative, gcov_em, gcov);
    }
    KOKKOS_INLINE_FUNCTION Real gcon_native(
        const GReal X[GR_DIM], Real gcon[GR_DIM][GR_DIM]) const
    {
        Real gcov[GR_DIM][GR_DIM];
        gcov_native(X, gcov);
        return gcon_from_gcov(gcov, gcon);
    }
    KOKKOS_INLINE_FUNCTION Real gdet_native(const GReal X[GR_DIM]) const
    {
        Real gcon[GR_DIM][GR_DIM];
        return gcon_native(X, gcon);
    }

    KOKKOS_INLINE_FUNCTION void conn_native(
        const GReal X[GR_DIM], const GReal delta, Real conn[GR_DIM][GR_DIM][GR_DIM]) const
    {
        GReal tmp[GR_DIM][GR_DIM][GR_DIM];
        GReal gcon[GR_DIM][GR_DIM];
        GReal Xh[GR_DIM], Xl[GR_DIM];
        GReal gh[GR_DIM][GR_DIM];
        GReal gl[GR_DIM][GR_DIM];

        for (int nu = 0; nu < GR_DIM; nu++) {
            DLOOP1
                Xl[mu] = X[mu] - delta * (mu == nu);
            DLOOP1
                Xh[mu] = X[mu] + delta * (mu == nu);
            gcov_native(Xh, gh);
            gcov_native(Xl, gl);

            for (int lam = 0; lam < GR_DIM; lam++) {
                for (int kap = 0; kap < GR_DIM; kap++) {
                    conn[lam][kap][nu] =
                        (gh[lam][kap] - gl[lam][kap]) / (Xh[nu] - Xl[nu]);
                }
            }
        }

        // Rearrange to find \Gamma_{lam nu mu}
        for (int lam = 0; lam < GR_DIM; lam++) {
            for (int nu = 0; nu < GR_DIM; nu++) {
                for (int mu = 0; mu < GR_DIM; mu++) {
                    tmp[lam][nu][mu] =
                        0.5 * (conn[nu][lam][mu] + conn[mu][lam][nu] - conn[mu][nu][lam]);
                }
            }
        }

        // Need gcon for raising index
        gcon_native(X, gcon);

        // Raise index to get \Gamma^lam_{nu mu}
        for (int lam = 0; lam < GR_DIM; lam++) {
            for (int nu = 0; nu < GR_DIM; nu++) {
                for (int mu = 0; mu < GR_DIM; mu++) {
                    conn[lam][nu][mu] = 0.;

                    for (int kap = 0; kap < GR_DIM; kap++)
                        conn[lam][nu][mu] += gcon[lam][kap] * tmp[kap][nu][mu];
                }
            }
        }
    }

    /**
     * Takes a velocity in Boyer-Lindquist coordinates (optionally without time component)
     * and converts it to KS, and then to native coordinates. Not guaranteed to be fast.
     */
    KOKKOS_INLINE_FUNCTION void bl_fourvel_to_native(const Real Xnative[GR_DIM],
        const Real ucon_bl[GR_DIM], Real ucon_native[GR_DIM]) const
    {
        GReal Xembed[GR_DIM];
        coord_to_embed(Xnative, Xembed);

        // Set u^t to make u a velocity 4-vector in BL
        GReal gcov_bl[GR_DIM][GR_DIM];
        if (PortsOfCall::holds_alternative<SphKSCoords>(base) ||
            PortsOfCall::holds_alternative<SphBLCoords>(base)) {
            SphBLCoords(get_a()).gcov_embed(Xembed, gcov_bl);
        } else if (PortsOfCall::holds_alternative<SphKSExtG>(base) ||
                   PortsOfCall::holds_alternative<SphBLExtG>(base)) {
            SphBLExtG(get_a()).gcov_embed(Xembed, gcov_bl);
        }

        Real ucon_bl_fourv[GR_DIM];
        DLOOP1
            ucon_bl_fourv[mu] = ucon_bl[mu];
        set_ut(gcov_bl, ucon_bl_fourv);

        // Then transform that 4-vector to KS (or not, if we're using BL base coords)
        Real ucon_base[GR_DIM];
        if (PortsOfCall::holds_alternative<SphKSCoords>(base)) {
            PortsOfCall::get<SphKSCoords>(base).vec_from_bl(Xembed, ucon_bl_fourv, ucon_base);
        } else if (PortsOfCall::holds_alternative<SphKSExtG>(base)) {
            PortsOfCall::get<SphKSExtG>(base).vec_from_bl(Xembed, ucon_bl_fourv, ucon_base);
        } else if (PortsOfCall::holds_alternative<SphBLCoords>(base) ||
                   PortsOfCall::holds_alternative<SphBLExtG>(base)) {
            DLOOP1
                ucon_base[mu] = ucon_bl_fourv[mu];
        }
        // Finally, apply any transform to native coordinates
        con_vec_to_native(Xnative, ucon_base, ucon_native);
    }
};
